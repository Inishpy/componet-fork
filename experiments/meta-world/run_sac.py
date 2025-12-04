#!/usr/bin/env python3
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
import pathlib
from torch.utils.tensorboard import SummaryWriter
from typing import Literal, Optional, Tuple
from models import shared, SimpleAgent, CompoNetAgent, PackNetAgent, ProgressiveNetAgent,DiffusionAgent
# from models.sequential_merge import SequentialMergeAgent
from tasks import get_task, get_task_name
from stable_baselines3.common.buffers import ReplayBuffer
import logging
from queue import deque

class StateBuffer:
    """
    Buffer to collect and store state samples for policy-aware merging.
    Stores diverse states encountered during training.
    """
    def __init__(self, max_size=100000, obs_dim=None):
        self.max_size = max_size
        self.obs_dim = obs_dim
        self.buffer = deque(maxlen=max_size)
        
    def add(self, state):
        """Add a single state to the buffer"""
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        self.buffer.append(state)
    
    def add_batch(self, states):
        """Add a batch of states to the buffer"""
        if isinstance(states, np.ndarray):
            states = torch.from_numpy(states).float()
        for state in states:
            self.buffer.append(state)
    
    def sample(self, batch_size):
        """Sample a batch of states"""
        if len(self.buffer) == 0:
            return None
        
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        samples = [self.buffer[i] for i in indices]
        return torch.stack(samples)
    
    def get_all(self):
        """Get all states in the buffer"""
        if len(self.buffer) == 0:
            return None
        return torch.stack(list(self.buffer))
    
    def __len__(self):
        return len(self.buffer)
    
    def save(self, filepath):
        """Save buffer to disk"""
        if len(self.buffer) > 0:
            states = torch.stack(list(self.buffer))
            torch.save(states, filepath)
            print(f"Saved {len(self.buffer)} states to {filepath}")
    
    @staticmethod
    def load(filepath):
        """Load buffer from disk"""
        if os.path.exists(filepath):
            states = torch.load(filepath)
            buffer = StateBuffer(max_size=len(states))
            for state in states:
                buffer.add(state)
            print(f"Loaded {len(buffer)} states from {filepath}")
            return buffer
        return None


@dataclass
class Args:
    model_type: Literal["simple", "finetune", "componet", "packnet", "prognet", "sequential-merge", "diffusion"]
    """The name of the NN model to use for the agent"""
    save_dir: Optional[str] = None
    """If provided, the model will be saved in the given directory"""
    prev_units: Tuple[pathlib.Path, ...] = ()
    """Paths to the previous models. Not required when model_type is `simple` or `packnet` or `prognet`"""

    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cw-sac"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    num_envs: int = 1
    """Number of parallel environments to use"""

    # Algorithm specific arguments
    task_id: int = 0
    """ID number of the task"""
    eval_every: int = 10_000
    """Evaluate the agent in determinstic mode every X timesteps"""
    num_evals: int = 10
    """Number of times to evaluate the agent"""
    total_timesteps: int = int(1e6)
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 128
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5_000
    """timestep to start learning"""
    random_actions_end: int = 10_000
    """timesteps to take actions randomly"""
    policy_lr: float = 1e-3
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = False
    """automatic tuning of the entropy coefficient"""


    # State buffer arguments for enhanced DOP merge
    state_buffer_size: int = 5000
    """Maximum number of states to store for policy-aware merging"""
    state_buffer_sample_freq: int = 10
    """Sample states every N steps"""
    use_enhanced_merge: bool = True
    """Use enhanced DOP merge with policy and Fisher terms"""
    
    # DOP merge hyperparameters
    dop_K: int = 50
    """Number of optimization iterations for DOP merge"""
    dop_r: int = 16
    """Rank for SVD approximation"""
    dop_beta: float = 0.95
    """Momentum parameter for alpha smoothing"""
    dop_eta: float = 1e-3
    """Learning rate for DOP optimization"""
    dop_lambda_kl: float = 0.1
    """Weight for KL divergence term"""
    dop_lambda_f: float = 0.01
    """Weight for Fisher information term"""


def make_env(task_id):
    def thunk():
        env = get_task(task_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.fc = shared(
            np.array(envs.single_observation_space.shape).prod()
            + np.prod(envs.single_action_space.shape)
        )
        self.fc_out = nn.Linear(256, 1)

    def forward(self, x, a):
        x = torch.cat([x, a], 1)
        x = self.fc(x)
        x = self.fc_out(x)
        return x


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class Actor(nn.Module):
    def __init__(self, envs, model):
        super().__init__()
        self.model = model

        # action rescaling
        self.register_buffer(
            "action_scale",
            torch.tensor(
                (envs.single_action_space.high - envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )
        self.register_buffer(
            "action_bias",
            torch.tensor(
                (envs.single_action_space.high + envs.single_action_space.low) / 2.0,
                dtype=torch.float32,
            ),
        )

    def forward(self, x, **kwargs):
        mean, log_std = self.model(x, **kwargs)
        log_std = torch.tanh(log_std)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            log_std + 1
        )  # From SpinUp / Denis Yarats

        return mean, log_std

    def get_action(self, x, **kwargs):
        if isinstance(self.model, DiffusionAgent):
            # diffusion returns raw actions (already scaled if you passed envs)
            action = self.model.get_action(x, temperature=1.0)
            # Check for NaN, Inf, or huge values in diffusion output
            if torch.isnan(action).any() or torch.isinf(action).any() or (action.abs() > 1e6).any():
                print("[ERROR] Invalid action detected in DiffusionAgent.get_action: NaN, Inf, or huge value")
                print(f"action: {action}")
                # Optionally, clip or zero out invalid actions
                action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
                action = torch.clamp(action, -1e6, 1e6)
            return action, None, None
        else:
            # old Gaussian path
            mean, log_std = self(x, **kwargs)
            std = log_std.exp()
            # Debug: log mean and std for nan/inf/huge
            if (
                torch.isnan(mean).any() or torch.isnan(std).any() or
                torch.isinf(mean).any() or torch.isinf(std).any() or
                (mean.abs() > 1e6).any() or (std.abs() > 1e6).any()
            ):
                print("[ERROR] Invalid mean or std in get_action: NaN, Inf, or huge value")
                print(f"mean: {mean}")
                print(f"log_std: {log_std}")
                print(f"std: {std}")
                # Optionally, clip or zero out invalid values
                mean = torch.nan_to_num(mean, nan=0.0, posinf=0.0, neginf=0.0)
                std = torch.nan_to_num(std, nan=1.0, posinf=1.0, neginf=1.0)
                mean = torch.clamp(mean, -1e6, 1e6)
                std = torch.clamp(std, 1e-6, 1e6)
            normal = torch.distributions.Normal(mean, std)
            x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
            y_t = torch.tanh(x_t)
            action = y_t * self.action_scale + self.action_bias
            # Check for NaN, Inf, or huge values in action
            if torch.isnan(action).any() or torch.isinf(action).any() or (action.abs() > 1e6).any():
                print("[ERROR] Invalid action sampled: NaN, Inf, or huge value")
                print(f"action: {action}")
                action = torch.nan_to_num(action, nan=0.0, posinf=0.0, neginf=0.0)
                action = torch.clamp(action, -1e6, 1e6)
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            mean = torch.tanh(mean) * self.action_scale + self.action_bias
            return action, log_prob, mean


@torch.no_grad()
def eval_agent(agent, test_env, num_evals, global_step, writer, device):
    obs, _ = test_env.reset()
    avg_ep_ret = 0
    avg_success = 0
    ep_ret = 0
    for _ in range(num_evals):
        while True:
            obs = torch.Tensor(obs).to(device).unsqueeze(0)
            # Handle diffusion and non-diffusion agents correctly
            if isinstance(agent.model, DiffusionAgent):
                action = agent.get_action(obs)
            else:
                action, _, _ = agent.get_action(obs)
            obs, reward, termination, truncation, info = test_env.step(
                action[0].cpu().numpy()
            )

            ep_ret += reward

            if termination or truncation:
                avg_success += info["success"]
                avg_ep_ret += ep_ret
                # resets
                obs, _ = test_env.reset()
                ep_ret = 0
                break
    avg_ep_ret /= num_evals
    avg_success /= num_evals
    print(f"\nTEST: ep_ret={avg_ep_ret}, success={avg_success}\n")
    writer.add_scalar("charts/test_episodic_return", avg_ep_ret, global_step)
    writer.add_scalar("charts/test_success", avg_success, global_step)
    return avg_ep_ret, avg_success


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"task_{args.task_id}__{args.model_type}__{args.exp_name}__{args.seed}"
    print(f"\n*** Run name: {run_name} ***\n")
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    print(f"*** Device: {device}")

    # env setup
    num_envs = args.num_envs
    envs = gym.vector.AsyncVectorEnv([make_env(args.task_id) for _ in range(num_envs)])
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    max_action = float(envs.single_action_space.high[0])

    # select the model to use as the agent
    obs_dim = np.array(envs.single_observation_space.shape).prod()
    # print(obs_dim, "obs dim", envs.single_observation_space)
    act_dim = np.prod(envs.single_action_space.shape)



    # Initialize state buffer for current task
    state_buffer_current = StateBuffer(max_size=10000, obs_dim=obs_dim)
    state_buffers_prev = []

    # Load all previous state buffers if available
    if args.task_id > 0 and len(args.prev_units) > 0 and args.use_enhanced_merge:
        for prev_unit in args.prev_units:
            prev_buffer_path = f"{prev_unit}/state_buffer.pt"
            if os.path.exists(prev_buffer_path):
                buf = StateBuffer.load(prev_buffer_path)
                if buf is not None and len(buf) > 0:
                    state_buffers_prev.append(buf)
                    print(f"Loaded previous state buffer with {len(buf)} states from {prev_buffer_path}")
                else:
                    print(f"Warning: Previous state buffer at {prev_buffer_path} is empty or failed to load")
            else:
                print(f"Warning: Previous state buffer not found at {prev_buffer_path}")



    print(f"*** Loading model `{args.model_type}` ***")
    if args.model_type in ["finetune", "componet"]:
        assert (
            len(args.prev_units) > 0
        ), f"Model type {args.model_type} requires at least one previous unit"

    if args.model_type == "simple":
        model = SimpleAgent(obs_dim=obs_dim, act_dim=act_dim).to(device)

    elif args.model_type == "finetune":
        model = SimpleAgent.load(
            args.prev_units[0], map_location=device, reset_heads=True
        ).to(device)

    elif args.model_type == "componet":
        model = CompoNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            prev_paths=args.prev_units,
            map_location=device,
        ).to(device)
    elif args.model_type == "packnet":
        packnet_retrain_start = args.total_timesteps - int(args.total_timesteps * 0.2)
        if len(args.prev_units) == 0:
            model = PackNetAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                task_id=args.task_id,
                total_task_num=20,
                device=device,
            ).to(device)
        else:
            model = PackNetAgent.load(
                args.prev_units[0],
                task_id=args.task_id + 1,
                restart_heads=True,
                freeze_bias=True,
                map_location=device,
            ).to(device)
    elif args.model_type == "prognet":
        model = ProgressiveNetAgent(
            obs_dim=obs_dim,
            act_dim=act_dim,
            prev_paths=args.prev_units,
            map_location=device,
        ).to(device)
    elif args.model_type == "sequential-merge":
        if args.task_id > 0 and len(args.prev_units) > 0:
            try:
                model = SimpleAgent.load(args.prev_units[0], map_location=device, reset_heads=False).to(device)
                print(f"Loaded previous model from {args.prev_units[0]}")
            except Exception as e:
                logging.info(f"Error loading previous model: {e}")
                model = SimpleAgent(obs_dim=obs_dim, act_dim=act_dim).to(device)
        else:
            model = SimpleAgent(obs_dim=obs_dim, act_dim=act_dim).to(device)

    elif args.model_type == "diffusion":
            model = DiffusionAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dim=256,
                depth=4,
                embed_dim=128,
                T=50,               # you can reduce to 5-10 for faster sampling
            ).to(device)

    actor = Actor(envs, model).to(device)
    qf1 = SoftQNetwork(envs).to(device)
    qf2 = SoftQNetwork(envs).to(device)
    qf1_target = SoftQNetwork(envs).to(device)
    qf2_target = SoftQNetwork(envs).to(device)
    qf1_target.load_state_dict(qf1.state_dict())
    qf2_target.load_state_dict(qf2.state_dict())
    q_optimizer = optim.Adam(
        list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr
    )
    actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)

    # Automatic entropy tuning
    if args.autotune:
        target_entropy = -torch.prod(
            torch.Tensor(envs.single_action_space.shape).to(device)
        ).item()
        log_alpha = torch.zeros(1, requires_grad=True, device=device)
        alpha = log_alpha.exp().item()
        a_optimizer = optim.Adam([log_alpha], lr=args.q_lr)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32
    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        n_envs=num_envs,
        handle_timeout_termination=False,
    )

    start_time = time.time()

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    global_step = 0
    total_update_steps = 0

    # --- Early stopping based on consecutive successes ---
    consecutive_successes = 0
    required_consecutive_successes = 5  # Stop after 5 consecutive successes
    early_stop_triggered = False

    while global_step < args.total_timesteps:
        # ALGO LOGIC: put action logic here
        if global_step < args.random_actions_end:
            actions = np.array(
                [envs.single_action_space.sample() for _ in range(num_envs)]
            )
        else:
            if args.model_type == "componet" and global_step % 1000 == 0:
                actions, _, _ = actor.get_action(
                    torch.Tensor(obs).to(device),
                    writer=writer,
                    global_step=global_step,
                )
            else:
                actions, _, _ = actor.get_action(torch.Tensor(obs).to(device))
            actions = actions.detach().cpu().numpy()
            actions = np.array(actions)
            actions = np.atleast_1d(actions)
        # Collect states for state buffer (periodically)
        if global_step % args.state_buffer_sample_freq == 0:
            state_buffer_current.add_batch(obs)
            
        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        global_step += num_envs

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs
        
        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    print(
                        f"global_step={global_step}, episodic_return={info['episode']['r']}, success={info['success']}"
                    )
                    writer.add_scalar(
                        "charts/episodic_return", info["episode"]["r"], global_step
                    )
                    writer.add_scalar(
                        "charts/episodic_length", info["episode"]["l"], global_step
                    )
                    writer.add_scalar("charts/success", info["success"], global_step)
                    # --- Early stopping logic: check for consecutive successes ---
                    if info.get("success", 0) == 1:
                        consecutive_successes += 1
                        if consecutive_successes >= required_consecutive_successes:
                            print(f"Early stopping: {required_consecutive_successes} consecutive successes achieved at global_step={global_step}")
                            early_stop_triggered = True
                            break
                    else:
                        consecutive_successes = 0
            if early_stop_triggered:
                break

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            for gradient_step in range(num_envs):
                total_update_steps += 1
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    
                    if isinstance(actor.model, DiffusionAgent):
                        next_state_actions = actor.model.get_action(data.next_observations, temperature=1.0)
                        next_state_log_pi = torch.zeros((data.rewards.shape[0], 1), device=device)
                    else:
                        next_state_actions, next_state_log_pi, _ = actor.get_action(data.next_observations)

                    qf1_next_target = qf1_target(data.next_observations, next_state_actions)
                    qf2_next_target = qf2_target(data.next_observations, next_state_actions)
                    min_qf_next_target = (
                        torch.min(qf1_next_target, qf2_next_target)
                        - alpha * next_state_log_pi
                    )
                    next_q_value = data.rewards.flatten() + (
                        1 - data.dones.flatten()
                    ) * args.gamma * (min_qf_next_target).view(-1)

                qf1_a_values = qf1(data.observations, data.actions).view(-1)
                qf2_a_values = qf2(data.observations, data.actions).view(-1)
                qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                qf_loss = qf1_loss + qf2_loss

                # optimize the model
                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if total_update_steps % args.policy_frequency == 0:
                    for _ in range(args.policy_frequency):
                        if isinstance(actor.model, DiffusionAgent):
                            # ------------------- Diffusion denoising loss -------------------
                            batch_size = data.actions.shape[0]
                            t = torch.randint(0, actor.model.T, (batch_size,), device=device)          # random timestep
                            noise = torch.randn_like(data.actions)

                            # Forward diffusion: q(a_t | a_0)
                            sqrt_alph_cumprod_t = actor.model.alpha_hats[t].sqrt()[:, None]
                            sqrt_one_minus_alph_cumprod_t = (1.0 - actor.model.alpha_hats[t]).sqrt()[:, None]
                            noisy_actions = sqrt_alph_cumprod_t * data.actions + sqrt_one_minus_alph_cumprod_t * noise

                            # Predict noise
                            eps_pred = actor.model(noisy_actions, data.observations, t.float() / actor.model.T)

                            actor_loss = F.mse_loss(eps_pred, noise)
                            # ------------------------------------------------------------------
                        else:
                            # Original Gaussian SAC loss (unchanged)
                            pi, log_pi, _ = actor.get_action(data.observations)
                            qf1_pi = qf1(data.observations, pi)
                            qf2_pi = qf2(data.observations, pi)
                            min_qf_pi = torch.min(qf1_pi, qf2_pi)
                            actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                        actor_optimizer.zero_grad()
                        actor_loss.backward()
                        if args.model_type == "packnet":
                            if global_step >= packnet_retrain_start:
                                # can be called multiple times, only the first counts
                                actor.model.start_retraining()
                            actor.model.before_update()
                        actor_optimizer.step()

                        if args.autotune:
                            with torch.no_grad():
                                _, log_pi, _ = actor.get_action(data.observations)

                            if log_pi is not None:  # Gaussian policy → normal SAC autotune
                                alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                                a_optimizer.zero_grad()
                                alpha_loss.backward()
                                a_optimizer.step()
                                alpha = log_alpha.exp().item()
                            else:  # Diffusion policy → no analytical log_prob exists
                                # Keep alpha fixed (diffusion already stochastic via sampling temperature)
                                alpha = 0.0
                        else:
                            alpha = args.alpha

                        # Force alpha = 0 for diffusion even if autotune is off (optional but recommended)
                        if isinstance(actor.model, DiffusionAgent):
                            alpha = 0.0

                # update the target networks
                if total_update_steps % args.target_network_frequency == 0:
                    for param, target_param in zip(
                        qf1.parameters(), qf1_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )
                    for param, target_param in zip(
                        qf2.parameters(), qf2_target.parameters()
                    ):
                        target_param.data.copy_(
                            args.tau * param.data + (1 - args.tau) * target_param.data
                        )

            if global_step % 100 == 0:
                writer.add_scalar(
                    "losses/qf1_values", qf1_a_values.mean().item(), global_step
                )
                writer.add_scalar(
                    "losses/qf2_values", qf2_a_values.mean().item(), global_step
                )
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS",
                    int(global_step / (time.time() - start_time)),
                    global_step,
                )
                if args.autotune:
                    writer.add_scalar(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )

    # --- Sequential-merge post-training merge logic ---
    # if args.model_type == "sequential-merge" and args.task_id > 0 and len(args.prev_units) > 0:
    #     from models.sequential_merge import dop_merge_simple
    #     from models.simple import SimpleAgent

    #     print("[SEQUENTIAL-MERGE] Loading previous model for merging ...", args.prev_units)
    #     prev_model = SimpleAgent.load(args.prev_units[0], map_location=device, reset_heads=False).to(device)
    #     print("[SEQUENTIAL-MERGE] Merging previous and current (trained) model with DOP merge ...")
    #     # Theta0: you need a base pretrained model. If not available use prev_model as a proxy.
    #     Theta0 = prev_model  # replace with actual base pretrained if you have it
    #     merged_model = dop_merge_simple(Theta0, prev_model, actor.model, K=50, r=16, beta=0.95, eta=1e-3)
    #     # Optionally: retrain merged_model here if desired
    #     actor.model.load_state_dict(merged_model.state_dict())
    # --- Enhanced DOP merge post-training logic ---
    if args.model_type == "sequential-merge" and args.task_id > 0 and len(args.prev_units) > 0:
        print("\n" + "="*60)
        print("[ENHANCED DOP MERGE] Starting merge process...")
        print("="*60)
        
        # Import the enhanced merge function
        from models.sequential_merge import dop_merge_enhanced
        
        # Load previous model
        print(f"[MERGE] Loading previous model from {args.prev_units[0]}...")
        prev_model = SimpleAgent.load(args.prev_units[0], map_location=device, reset_heads=False).to(device)
        
        # Always load the first trained model of the first task as Theta0, where the first task is determined by start_mode
        import glob
        import re
        first_task_model_path = None
        start_mode_id = None
        if args.save_dir is not None:
            # Find all task directories in save_dir matching the experiment naming convention
            pattern = f"{args.save_dir}/task_*__*__{args.exp_name}__{args.seed}"
            candidates = glob.glob(pattern)
            # Extract task ids
            task_ids = []
            for c in candidates:
                m = re.search(r"task_(\d+)__", c)
                if m:
                    task_ids.append(int(m.group(1)))
            if task_ids:
                start_mode_id = min(task_ids)
                # Find the path for the minimum task id
                for c in candidates:
                    if f"task_{start_mode_id}__" in c:
                        first_task_model_path = c
                        break
        if not first_task_model_path and len(args.prev_units) > 0:
            # Fallback: try to infer from prev_units path
            prev_path = str(args.prev_units[0])
            m = re.search(r"task_(\d+)__", prev_path)
            if m:
                # Try to replace with the minimum task id found above, else fallback to 0
                min_id = start_mode_id if start_mode_id is not None else 0
                first_task_model_path = re.sub(r"task_\d+__", f"task_{min_id}__", prev_path)
            else:
                first_task_model_path = prev_path
        print(f"[MERGE] Loading first task model as Theta0 from {first_task_model_path} ...")
        Theta0 = SimpleAgent.load(first_task_model_path, map_location=device, reset_heads=False).to(device)
        
        # Get state samples
        states_old = None
        states_new = None
        
        if True: #args.use_enhanced_merge:
            print(f"[MERGE] Preparing state samples for policy-aware merging...")
            
            # Sample states from previous task buffer
            if state_buffers_prev is not None and len(state_buffers_prev) > 0:
                # Sample 1000 from each previous buffer and concatenate
                sampled_states = []
                for buf in state_buffers_prev:
                    if len(buf) > 0:
                        s = buf.sample(min(2000, len(buf)))
                        if s is not None:
                            sampled_states.append(s)
                if len(sampled_states) > 0:
                    states_old = torch.cat(sampled_states, dim=0).to(device)
                    print(f"[MERGE] Sampled {states_old.shape[0]} states from all previous task buffers")
                else:
                    states_old = None
                    print("[MERGE] Warning: No previous task states available")
            
            # Sample states from current task buffer
            if len(state_buffer_current) > 0:
                states_new = state_buffer_current.sample(min(2000, len(state_buffer_current)))
                states_new = states_new.to(device)
                print(f"[MERGE] Sampled {states_new.shape[0]} states from current task")
            else:
                print("[MERGE] Warning: No current task states available")
        
        # Perform enhanced DOP merge
        print(f"[MERGE] Performing enhanced DOP merge with:")
        print(f"  - K={args.dop_K} iterations")
        print(f"  - r={args.dop_r} rank")
        print(f"  - λ_kl={args.dop_lambda_kl}")
        print(f"  - λ_f={args.dop_lambda_f}")
        
        merged_model = dop_merge_enhanced(
            Theta0=Theta0,
            Theta_old=prev_model,
            Theta_new=actor.model,
            states_old=states_old,
            states_new=states_new,
            K=args.dop_K,
            r=args.dop_r,
            beta=args.dop_beta,
            eta=args.dop_eta,
            lambda_kl=args.dop_lambda_kl if args.use_enhanced_merge else 0.0,
            lambda_f=args.dop_lambda_f if args.use_enhanced_merge else 0.0,
            use_policy_terms=args.use_enhanced_merge,
        )
        
        print("[MERGE] Merge complete! Loading merged weights into actor...")
        actor.model.load_state_dict(merged_model.state_dict())
        print("="*60 + "\n")
        # Debug: check for NaNs in merged model output
        test_states = None
        if states_new is not None:
            test_states = states_new[:5]
        elif states_old is not None:
            test_states = states_old[:5]
        if test_states is not None:
            with torch.no_grad():
                mean, log_std = actor.model(test_states)
                if torch.isnan(mean).any() or torch.isnan(log_std).any():
                    print("[DEBUG] NaN detected in merged model output after merge")
                    print(f"mean: {mean}")
                    print(f"log_std: {log_std}")

    # Evaluate on all previous tasks and print/average results
    all_returns = []
    all_successes = []
    print("\n=== Evaluation on all previous tasks ===")
    for tid in range(args.task_id+1):
        test_env = make_env(tid)()
        ep_ret, success = eval_agent(actor, test_env, args.num_evals, global_step, writer, device)
        print(f"Task {tid}: ep_ret={ep_ret:.2f}, success={success:.2f}")
        all_returns.append(ep_ret)
        all_successes.append(success)
        test_env.close()
    avg_ret = np.mean(all_returns)
    avg_success = np.mean(all_successes)
    print(f"\nAverage over all tasks: ep_ret={avg_ret:.2f}, success={avg_success:.2f}\n")

    envs.close()
    writer.close()

    # Save model and state buffer
    if args.save_dir is not None:
        save_path = f"{args.save_dir}/{run_name}"
        print(f"[SAVE] Saving trained agent to {save_path}")
        actor.model.save(dirname=save_path)
        
        # Save state buffer for future merging
        if args.use_enhanced_merge and len(state_buffer_current) > 0:
            buffer_path = f"{save_path}/state_buffer.pt"
            state_buffer_current.save(buffer_path)
            print(f"[SAVE] Saved state buffer with {len(state_buffer_current)} states")


    # Log training duration
    end_time = time.time()
    elapsed = end_time - start_time
    print(f"Training took {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")