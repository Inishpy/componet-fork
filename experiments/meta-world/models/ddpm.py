import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import os

def vp_beta_schedule(timesteps: int):
    t = torch.arange(1, timesteps + 1)
    T = timesteps
    b_max = 10.0
    b_min = 0.1
    alpha = torch.exp(-b_min / T - 0.5 * (b_max - b_min) * (2 * t - 1) / T ** 2)
    betas = 1 - alpha
    return betas.clamp(0, 0.999)

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class ResBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.Mish(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, x):
        return self.net(x) + x


class DiffusionAgent(nn.Module):
    """
    Drop-in replacement for SimpleAgent that uses a conditional DDPM denoiser (noise-prediction model)
    very close to the architecture in your JAX psec_pretrain.py.

    Differences from SimpleAgent:
    - forward() now predicts noise (eps) instead of (mean, log_std)
    - get_action() performs iterative denoising (sampling) instead of sampling from a Gaussian
    - There is no closed-form log_prob, so you will need to change the actor loss to denoising MSE loss (score matching / diffusion BC)
    - Sampling is fast when T is small (default T=50, but you can set T=5-10 like in the JAX code for ~10× speedup)

    How to use it in run_sac.py:
    1. Add "diffusion" to the Literal list in Args.model_type
 2. In the model selection block add:
        elif args.model_type == "diffusion":
            model = DiffusionAgent(
                obs_dim=obs_dim,
                act_dim=act_dim,
                hidden_dim=256,
                depth=4,
                embed_dim=128,
                T=50,               # you can reduce to 5-10 for faster sampling
            ).to(device)
 3. Change Actor.get_action() to detect diffusion models:
        def get_action(self, x, **kwargs):
            if isinstance(self.model, DiffusionAgent):
                # diffusion returns raw actions (already scaled if you passed envs)
                return self.model.get_action(x, temperature=1.0), None, None
            else:
                # old Gaussian path
                ...
 4. Change the actor update block to denoising loss:
        # inside the training loop, replace the Gaussian actor_loss with:
        if isinstance(actor.model, DiffusionAgent):
            t = torch.randint(0, actor.model.T, (data.actions.shape[0],), device=device)
            noise = torch.randn_like(data.actions)
            a_hat = torch.sqrt(actor.model.alpha_hats[t])[:, None] * data.actions + \
                    torch.sqrt(1 - actor.model.alpha_hats[t])[:, None] * noise
            eps_pred = actor.model(a_hat, data.observations, t.float() / actor.model.T)
            actor_loss = F.mse_loss(eps_pred, noise)
        else:
            # old Gaussian loss
            ...

    The critic update can stay exactly the same — just sample next_state_actions with model.get_action(data.next_observations)
    (you already do that for the Gaussian case).
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dim: int = 256,
        depth: int = 4,
        embed_dim: int = 128,
        T: int = 50,
        use_sinusoidal_embed: bool = True,
    ):
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.embed_dim = embed_dim
        self.T = T

        if use_sinusoidal_embed:
            self.time_embed = nn.Sequential(
                SinusoidalEmbedding(embed_dim),
                nn.Linear(embed_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
        else:
            # learned Gaussian Fourier features like in your JAX code
            self.time_embed = nn.Linear(embed_dim // 2, embed_dim)

        input_dim = act_dim + obs_dim + hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        self.blocks = nn.ModuleList([ResBlock(hidden_dim) for _ in range(depth)])

        self.out_proj = nn.Linear(hidden_dim, act_dim)

        self.act = nn.Mish()

        # VP schedule to exactly match your JAX code
        betas = vp_beta_schedule(T)
        self.register_buffer("betas", betas)
        alphas = 1.0 - betas
        self.register_buffer("alphas", alphas)
        alpha_hats = torch.cumprod(alphas, dim=0)
        self.register_buffer("alpha_hats", alpha_hats)

    def forward(self, noisy_actions: torch.Tensor, obs: torch.Tensor, t: torch.Tensor):
        """
        Predict noise.
        noisy_actions: (B, act_dim)
        obs:           (B, obs_dim)
        t:             (B,) float in [0, T)  — we normalize t/T for sinusoidal embedding
        """
        if not self.training:
            t = t * self.T  # in eval we receive normalized t, convert back to int index if needed

        cond = self.time_embed(t)

        x = torch.cat([noisy_actions, obs, cond], dim=1)
        x = self.input_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.act(x)
        return self.out_proj(x)

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        """
        Sample action(s) from the diffusion policy.
        obs: (B, obs_dim) or (obs_dim,)
        Returns: (B, act_dim) actions in the original action space (tanh-squashed if you want)
        """
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        batch_size = obs.shape[0]
        device = obs.device

        # start from pure noise (temperature controls exploration)
        x = torch.randn(batch_size, self.act_dim, device=device) * temperature

        for t_int in range(self.T - 1, -1, -1):
            t_norm = torch.full((batch_size,), t_int / self.T, device=device)

            eps = self.forward(x, obs, t_norm)

            alpha = self.alphas[t_int]
            alpha_hat = self.alpha_hats[t_int]

            # DDPM reverse step (noise prediction model) — exactly the formula used in your JAX code
            x = (1 / torch.sqrt(alpha)) * (x - (1 - alpha) / torch.sqrt(1 - alpha_hat) * eps)

            if t_int > 0:
                noise = torch.randn_like(x)
                x = x + torch.sqrt(self.betas[t_int]) * noise * temperature

        # Optional: enforce bounds with tanh (common in SAC)
        # If you trained on raw actions, remove tanh
        x = torch.tanh(x)

        return x.squeeze(0) if batch_size == 1 else x

    def save(self, dirname: str):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.state_dict(), f"{dirname}/model.pt")

    @classmethod
    def load(cls, dirname: str, obs_dim: int, act_dim: int, map_location=None, **kwargs):
        agent = cls(obs_dim=obs_dim, act_dim=act_dim, **kwargs)
        agent.load_state_dict(torch.load(f"{dirname}/model.pt", map_location=map_location))
        return agent