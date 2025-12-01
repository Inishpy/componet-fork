import torch
import torch.nn as nn
import os
from .shared_arch import shared

def layerwise_merge(model_a, model_b, eval_fn, data_loader, device, verbose=True):
    """
    Merge two models by selecting, for each layer, the one that performs best on eval_fn.
    Assumes both models have the same architecture.
    """
    merged = SimpleAgent(model_a.obs_dim, model_a.act_dim).to(device)
    merged.load_state_dict(model_a.state_dict())  # start with model_a

    # For each layer, try both versions and pick the one with better eval_fn score
    for name, param in merged.named_parameters():
        param_a = model_a.state_dict()[name]
        param_b = model_b.state_dict()[name]
        # Set to param_a, evaluate
        merged.state_dict()[name].copy_(param_a)
        score_a = eval_fn(merged, data_loader, device)
        # Set to param_b, evaluate
        merged.state_dict()[name].copy_(param_b)
        score_b = eval_fn(merged, data_loader, device)
        # Choose best
        if verbose:
            print(f"[MERGE] Layer: {name} | Score A: {score_a:.4f} | Score B: {score_b:.4f} | Selected: {'A' if score_a >= score_b else 'B'}")
        if score_a >= score_b:
            merged.state_dict()[name].copy_(param_a)
        else:
            merged.state_dict()[name].copy_(param_b)
    if verbose:
        print("[MERGE] Layerwise merge complete.")
    return merged

# Weighted average merge: new_model gets 60%, main model gets 40%

def weighted_average_merge(model_a, model_b, weight_new=0.6):
    """
    Merge two models by weighted averaging their parameters.
    Args:
        model_a: main model (previously merged)
        model_b: new task model
        weight_new: float, weight for new model (default 0.6)
    Returns:
        merged: new SimpleAgent with weighted parameters
    """
    merged = SimpleAgent(model_a.obs_dim, model_a.act_dim).to(model_a.fc_mean.weight.device)
    # merged.reset_heads()  # Ensure heads are initialized as in simple.py
    state_a = model_a.state_dict()
    state_b = model_b.state_dict()
    merged_state = merged.state_dict()
    for name in merged_state:
        merged_state[name].copy_(
            (1 - weight_new) * state_a[name] + weight_new * state_b[name]
        )
    merged.load_state_dict(merged_state)
    return merged



# models/sequential_merge.py
import copy
import torch
import torch.nn as nn

# def _top_r_svd(mat, r):
#     if torch.norm(mat) == 0:
#         m, n = mat.shape
#         return (torch.zeros((m, min(r, n)), device=mat.device, dtype=mat.dtype),
#                 torch.zeros((min(r, n),), device=mat.device, dtype=mat.dtype),
#                 torch.zeros((n, min(r, n)), device=mat.device, dtype=mat.dtype))
#     U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
#     r = choose_rank_by_energy(S)
#     r = min(r, S.numel())
#     return U[:, :r], S[:r], Vh[:r, :].T

# def choose_rank_by_energy(S, energy_thresh=0.9, r_max=None):
#     # S is 1D tensor of singular values
#     energy = (S**2)
#     cumsum = energy.cumsum(0)
#     total = cumsum[-1].clamp_min(1e-12)
#     frac = cumsum / total
#     r = int((frac >= energy_thresh).nonzero(as_tuple=False)[0].item()) + 1
#     if r_max is not None:
#         r = min(r, r_max)
#     return r

# def _project_onto_svd_subspace(X, U, V):
#     # U: (m,r), V: (n,r), X: (m,n)
#     if U.numel() == 0 or V.numel() == 0:
#         return torch.zeros_like(X)
#     coeffs = torch.einsum('ir,ij,jr->r', U, X, V)
#     proj = torch.zeros_like(X)
#     for i in range(U.shape[1]):
#         ui = U[:, i].unsqueeze(1)
#         vi = V[:, i].unsqueeze(1)
#         proj = proj + coeffs[i] * (ui @ vi.T)
#     return proj

# def dop_merge_simple(
#     Theta0,         # base pretrained model (nn.Module)
#     Theta_old,      # previously merged / main model (SimpleAgent)
#     Theta_new,      # new trained model (SimpleAgent)
#     K=20,
#     r=8,
#     beta=0.9,
#     eta=1e-2,
#     clip_eps=1e-8,
# ):
#     """
#     Dual Orthogonal Projection merge tailored to SimpleAgent-like models.
#     Returns a SimpleAgent instance with merged parameters.
#     """
#     # device/dtype
#     device = next(Theta_old.parameters()).device
#     dtype = next(Theta_old.parameters()).dtype

#     state0 = Theta0.state_dict()
#     state_old = Theta_old.state_dict()
#     state_new = Theta_new.state_dict()

#     merged = type(Theta_old)(Theta_old.obs_dim, Theta_old.act_dim).to(device)
#     # merged.reset_heads()  # uncomment if you want freshly initialized heads

#     merged_state = merged.state_dict()

#     for name in merged_state.keys():
#         W0 = state0.get(name, None)
#         Wold = state_old[name].to(device=device, dtype=dtype)
#         Wnew = state_new.get(name, None)
#         if Wnew is None:
#             merged_state[name] = Wold.clone()
#             continue
#         Wnew = Wnew.to(device=device, dtype=dtype)

#         # treat 2D tensors as linear weights eligible for SVD-based merge
#         if Wold.ndim == 2 and Wnew.ndim == 2 and (W0 is not None and getattr(W0, "ndim", 0) == 2):
#             W0 = W0.to(device=device, dtype=dtype)

#             tau_old = (Wold - W0).detach()
#             tau_new = (Wnew - W0).detach()

#             Uo, So, Vo = _top_r_svd(tau_old, r)
#             Un, Sn, Vn = _top_r_svd(tau_new, r)

#             Wstar = ((Wold + Wnew) / 2.0).clone().to(device=device, dtype=dtype)

#             alpha_s_prev = 0.5
#             alpha_p_prev = 0.5

#             for k in range(K):
#                 Wstar.requires_grad_(True)

#                 delta_old = Wstar - Wold
#                 delta_new = Wstar - Wnew

#                 proj_old = _project_onto_svd_subspace(delta_old, Uo, Vo)
#                 proj_new = _project_onto_svd_subspace(delta_new, Un, Vn)

#                 Ls = 0.5 * torch.sum((delta_old - proj_old) ** 2)
#                 Lp = 0.5 * torch.sum((delta_new - proj_new) ** 2)

#                 grad_Ls = torch.autograd.grad(Ls, Wstar, retain_graph=False, create_graph=False)[0]
#                 grad_Lp = torch.autograd.grad(Lp, Wstar, retain_graph=False, create_graph=False)[0]

#                 g_s = grad_Ls.detach().view(-1)
#                 g_p = grad_Lp.detach().view(-1)
#                 diff = g_s - g_p
#                 denom = (diff @ diff).clamp_min(clip_eps)
#                 numer = ((g_p - g_s) @ g_p)
#                 alpha_k = float((numer / denom).clamp(0.0, 1.0))

#                 alpha_s_k = beta * alpha_s_prev + (1.0 - beta) * alpha_k
#                 alpha_p_k = beta * alpha_p_prev + (1.0 - beta) * (1.0 - alpha_k)

#                 gk = alpha_s_k * grad_Ls + alpha_p_k * grad_Lp

#                 with torch.no_grad():
#                     Wstar = (Wstar - eta * gk).detach().to(device=device, dtype=dtype)

#                 alpha_s_prev = alpha_s_k
#                 alpha_p_prev = alpha_p_k

#             merged_state[name] = Wstar.clone().detach()

#         else:
#             # fallback: simple average for biases and non-2D params
#             merged_state[name] = ((Wold + Wnew) / 2.0).clone().to(device=device, dtype=dtype)

#     # load merged state
#     merged.load_state_dict(merged_state)
#     return merged


  



def _top_r_svd(mat, r):
    if torch.norm(mat) == 0:
        m, n = mat.shape
        return (torch.zeros((m, min(r, n)), device=mat.device, dtype=mat.dtype),
                torch.zeros((min(r, n),), device=mat.device, dtype=mat.dtype),
                torch.zeros((n, min(r, n)), device=mat.device, dtype=mat.dtype))
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    r = min(r, S.numel())
    return U[:, :r], S[:r], Vh[:r, :].T

def choose_rank_by_energy(S, energy_thresh=0.9, r_max=None):
    # S is 1D tensor of singular values
    energy = (S**2)
    cumsum = energy.cumsum(0)
    total = cumsum[-1].clamp_min(1e-12)
    frac = cumsum / total
    indices = (frac >= energy_thresh).nonzero(as_tuple=False)
    if indices.numel() == 0:
        # If no singular values meet the threshold, use all available
        r = S.numel()
    else:
        r = int(indices[0].item()) + 1
    if r_max is not None:
        r = min(r, r_max)
    return r

def _project_onto_svd_subspace(X, U, V, S=None):
    """
    Project X onto subspace spanned by U and V.
    Optionally weight by singular values S.
    U: (m,r), V: (n,r), X: (m,n), S: (r,) optional
    """
    if U.numel() == 0 or V.numel() == 0:
        return torch.zeros_like(X)
    
    # Compute coefficients
    coeffs = torch.einsum('ir,ij,jr->r', U, X, V)
    
    # Weight by singular values if provided
    if S is not None:
        coeffs = coeffs * S
    
    # Reconstruct projection
    proj = torch.zeros_like(X)
    for i in range(U.shape[1]):
        ui = U[:, i].unsqueeze(1)
        vi = V[:, i].unsqueeze(1)
        c = coeffs[i] if S is None else coeffs[i] / S[i]  # Undo weighting for reconstruction
        proj = proj + c * (ui @ vi.T)
    return proj

def _compute_weighted_projection_loss(delta, U, V, S):
    """
    Compute: 0.5 * || delta - proj(delta) ||^2 weighted by singular values
    """
    proj = _project_onto_svd_subspace(delta, U, V, S)
    residual = delta - proj
    # Weight the loss by singular value importance
    loss = 0.5 * torch.sum(residual ** 2)
    return loss

def _estimate_kl_divergence(model_star, model_ref, states, act_dim):
    """
    Estimate KL(π_star || π_ref) over given states.
    Assumes model outputs (mean, log_std).
    """
    model_star.eval()
    model_ref.eval()
    
    with torch.no_grad():
        mean_star, log_std_star = model_star(states)
        mean_ref, log_std_ref = model_ref(states)
        
        # KL divergence for Gaussian distributions
        std_star = torch.exp(log_std_star)
        std_ref = torch.exp(log_std_ref)
        var_star = std_star ** 2
        var_ref = std_ref ** 2
        
        # KL(N(μ1,σ1²) || N(μ2,σ2²)) = log(σ2/σ1) + (σ1² + (μ1-μ2)²)/(2σ2²) - 1/2
        kl = (log_std_ref - log_std_star + 
              (var_star + (mean_star - mean_ref) ** 2) / (2 * var_ref + 1e-8) - 0.5)
        kl = torch.sum(kl)  # Sum over actions and batch
    
    return kl

def _compute_fisher_term(model, states, act_dim):
    """
    Compute Fisher information approximation: E[||∇log π||²]
    Using empirical samples from states.
    """
    model.eval()
    fisher_loss = 0.0
    
    for state in states:
        state = state.unsqueeze(0).requires_grad_(True)
        mean, log_std = model(state)
        
        # Sample action from policy
        std = torch.exp(log_std)
        action = mean + std * torch.randn_like(mean)
        
        # Compute log probability
        log_prob = -0.5 * ((action - mean) / (std + 1e-8)) ** 2 - log_std - 0.5 * torch.log(2 * torch.tensor(3.14159))
        log_prob = torch.sum(log_prob)
        
        # Compute gradient
        grad = torch.autograd.grad(log_prob, state, retain_graph=False)[0]
        fisher_loss += torch.sum(grad ** 2)
    
    return fisher_loss / len(states)

def dop_merge_enhanced(
    Theta0,         # base pretrained model (nn.Module)
    Theta_old,      # previously merged / main model (SimpleAgent)
    Theta_new,      # new trained model (SimpleAgent)
    states_old=None,  # states from old task for policy KL
    states_new=None,  # states from new task for policy KL
    K=20,
    r=8,
    beta=0.9,
    eta=1e-2,
    lambda_kl=0.1,   # weight for KL divergence term
    lambda_f=0.01,   # weight for Fisher information term
    energy_thresh=0.9,
    clip_eps=1e-8,
    use_policy_terms=True,  # Flag to enable/disable policy and Fisher terms
):
    """
    Enhanced Dual Orthogonal Projection merge with:
    - Singular value weighting in projections
    - Policy-based KL divergence loss
    - Fisher information regularization
    """
    device = next(Theta_old.parameters()).device
    dtype = next(Theta_old.parameters()).dtype

    state0 = Theta0.state_dict()
    state_old = Theta_old.state_dict()
    state_new = Theta_new.state_dict()

    merged = type(Theta_old)(Theta_old.obs_dim, Theta_old.act_dim).to(device)
    merged_state = merged.state_dict()

    # Prepare state samples if provided
    if states_old is not None:
        states_old = states_old.to(device=device, dtype=dtype)
    if states_new is not None:
        states_new = states_new.to(device=device, dtype=dtype)

    for name in merged_state.keys():
        W0 = state0.get(name, None)
        Wold = state_old[name].to(device=device, dtype=dtype)
        Wnew = state_new.get(name, None)
        
        # Debug: print initial values for problematic layers
        if name in ["fc.0.weight", "fc.2.weight"]:
            print(f"[DEBUG] Initial values for {name}:")
            print(f"Wold: {Wold}")
            print(f"Wnew: {Wnew}")
            print(f"W0: {W0}")
        
        if Wnew is None:
            merged_state[name] = Wold.clone()
            continue
        
        Wnew = Wnew.to(device=device, dtype=dtype)

        # Process 2D tensors (linear layers)
        if Wold.ndim == 2 and Wnew.ndim == 2 and (W0 is not None and getattr(W0, "ndim", 0) == 2):
            W0 = W0.to(device=device, dtype=dtype)

            # Compute task vectors
            tau_old = (Wold - W0).detach()
            tau_new = (Wnew - W0).detach()

            # Debug: print tau values for problematic layers
            if name in ["fc.0.weight", "fc.2.weight"]:
                print(f"[DEBUG] tau_old for {name}: {tau_old}")
                print(f"[DEBUG] tau_new for {name}: {tau_new}")

            # SVD decomposition
            Uo, So, Vo = _top_r_svd(tau_old, r)
            Un, Sn, Vn = _top_r_svd(tau_new, r)

            # Debug: print SVD outputs for problematic layers
            if name in ["fc.0.weight", "fc.2.weight"]:
                print(f"[DEBUG] SVD outputs for {name}:")
                print(f"Uo: {Uo}")
                print(f"So: {So}")
                print(f"Vo: {Vo}")
                print(f"Un: {Un}")
                print(f"Sn: {Sn}")
                print(f"Vn: {Vn}")

            # If SVD outputs are all zeros or contain NaN/Inf, skip merge for this layer and use simple average
            if (
                (torch.all(So == 0) or torch.all(Sn == 0)) or
                torch.isnan(So).any() or torch.isnan(Sn).any() or
                torch.isinf(So).any() or torch.isinf(Sn).any()
            ):
                print(f"[DEBUG] SVD singular values are invalid for {name}, using simple average for merge.")
                merged_state[name] = ((Wold + Wnew) / 2.0).clone().to(device=device, dtype=dtype)
                continue

            # Apply energy-based rank selection
            r_old = choose_rank_by_energy(So, energy_thresh)
            r_new = choose_rank_by_energy(Sn, energy_thresh)
            
            Uo_r, So_r, Vo_r = Uo[:, :r_old], So[:r_old], Vo[:, :r_old]
            Un_r, Sn_r, Vn_r = Un[:, :r_new], Sn[:r_new], Vn[:, :r_new]

            # Initialize merge point (weighted average)
            Wstar = ((Wold + Wnew) / 2.0).clone().to(device=device, dtype=dtype)

            alpha_s_prev = 0.5
            alpha_p_prev = 0.5

            for k in range(K):
                # Create temporary model for policy evaluation
                temp_state = merged_state.copy()
                temp_state[name] = Wstar.clone()
                temp_model = type(Theta_old)(Theta_old.obs_dim, Theta_old.act_dim).to(device)
                temp_model.load_state_dict(temp_state)

                Wstar.requires_grad_(True)

                # Compute deltas
                delta_old = Wstar - Wold
                delta_new = Wstar - Wnew

                # Weighted projection losses
                Ls_proj = _compute_weighted_projection_loss(delta_old, Uo_r, Vo_r, So_r)
                Lp_proj = _compute_weighted_projection_loss(delta_new, Un_r, Vn_r, Sn_r)

                # Policy KL divergence terms
                Ls_policy = 0.0
                Lp_policy = 0.0
                if use_policy_terms and states_old is not None:
                    Ls_policy = _estimate_kl_divergence(temp_model, Theta_old, states_old, Theta_old.act_dim)
                if use_policy_terms and states_new is not None:
                    Lp_policy = _estimate_kl_divergence(temp_model, Theta_new, states_new, Theta_new.act_dim)

                # Fisher information terms
                fisher_old = 0.0
                fisher_new = 0.0
                if use_policy_terms and states_old is not None:
                    fisher_old = _compute_fisher_term(temp_model, states_old, Theta_old.act_dim)
                if use_policy_terms and states_new is not None:
                    fisher_new = _compute_fisher_term(temp_model, states_new, Theta_new.act_dim)

                # Total losses
                Ls_total = Ls_proj + lambda_kl * Ls_policy + lambda_f * fisher_old
                Lp_total = Lp_proj + lambda_kl * Lp_policy + lambda_f * fisher_new

                # Compute gradients
                grad_Ls = torch.autograd.grad(Ls_total, Wstar, retain_graph=True, create_graph=False)[0]
                grad_Lp = torch.autograd.grad(Lp_total, Wstar, retain_graph=False, create_graph=False)[0]

                # Compute adaptive alpha
                g_s = grad_Ls.detach().view(-1)
                g_p = grad_Lp.detach().view(-1)
                diff = g_s - g_p
                denom = (diff @ diff).clamp_min(clip_eps)
                numer = ((g_p - g_s) @ g_p)
                alpha_k = float((numer / denom).clamp(0.0, 1.0))

                # Smooth alphas with momentum
                alpha_s_k = beta * alpha_s_prev + (1.0 - beta) * alpha_k
                alpha_p_k = beta * alpha_p_prev + (1.0 - beta) * (1.0 - alpha_k)

                # Combined gradient
                gk = alpha_s_k * grad_Ls + alpha_p_k * grad_Lp

                # Update Wstar
                with torch.no_grad():
                    Wstar = (Wstar - eta * gk).detach().to(device=device, dtype=dtype)
                    # Debug: check for NaNs in Wstar after each update
                    if torch.isnan(Wstar).any():
                        print(f"[DEBUG] NaN detected in Wstar during merge at iteration {k} for layer {name}")
                        print(f"Wstar: {Wstar}")

                alpha_s_prev = alpha_s_k
                alpha_p_prev = alpha_p_k

            # Debug: check for NaNs in Wstar after all iterations
            if torch.isnan(Wstar).any():
                print(f"[DEBUG] NaN detected in Wstar after merge for layer {name}")
                print(f"Wstar: {Wstar}")

            merged_state[name] = Wstar.clone().detach()

        else:
            # Fallback: weighted average for biases and non-2D params
            merged_state[name] = ((Wold + Wnew) / 2.0).clone().to(device=device, dtype=dtype)

    # Load merged state
    # Debug: check for NaNs in all merged parameters before loading
    for pname, pval in merged_state.items():
        if torch.isnan(pval).any():
            print(f"[DEBUG] NaN detected in merged parameter {pname} before loading into model")
            print(f"Parameter value: {pval}")
    merged.load_state_dict(merged_state)
    return merged


def merge_with_critic_correction(
    actor_old, actor_new, critic_old, critic_new, 
    states_batch, actions_batch, rewards_batch, next_states_batch,
    Theta0_actor, Theta0_critic,
    gamma=0.99, td_steps=5, td_lr=1e-3,
    **merge_kwargs
):
    """
    Merge actors and critics, then perform TD correction on critic.
    
    Args:
        actor_old, actor_new: Actor networks
        critic_old, critic_new: Critic networks
        states_batch, actions_batch, rewards_batch, next_states_batch: Transition data
        Theta0_actor, Theta0_critic: Base pretrained models
        gamma: Discount factor
        td_steps: Number of TD correction steps
        td_lr: Learning rate for TD correction
        **merge_kwargs: Arguments for dop_merge_enhanced
    """
    # Merge actors
    merged_actor = dop_merge_enhanced(
        Theta0_actor, actor_old, actor_new,
        **merge_kwargs
    )
    
    # Merge critics similarly (without policy terms)
    merged_critic = dop_merge_enhanced(
        Theta0_critic, critic_old, critic_new,
        states_old=None, states_new=None,  # No policy loss for critic
        **{k: v for k, v in merge_kwargs.items() if k not in ['states_old', 'states_new']}
    )
    
    # TD correction pass
    optimizer = torch.optim.Adam(merged_critic.parameters(), lr=td_lr)
    merged_critic.train()
    
    for step in range(td_steps):
        # Compute current Q-values
        q_values = merged_critic(states_batch)
        q_values = q_values.gather(1, actions_batch.unsqueeze(1)).squeeze()
        
        # Compute target Q-values
        with torch.no_grad():
            next_actions, _ = merged_actor(next_states_batch)
            next_q = merged_critic(next_states_batch).max(1)[0]
            target_q = rewards_batch + gamma * next_q
        
        # TD loss
        td_loss = F.mse_loss(q_values, target_q)
        
        optimizer.zero_grad()
        td_loss.backward()
        optimizer.step()
    
    return merged_actor, merged_critic


class SimpleAgent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.fc1 = nn.Linear(obs_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc_mean = nn.Linear(256, self.act_dim)
        self.fc_logstd = nn.Linear(256, self.act_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        return mean, log_std

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self.state_dict(), f"{dirname}/model.pt")

    @staticmethod
    def load(dirname, obs_dim, act_dim, map_location=None):
        model = SimpleAgent(obs_dim, act_dim)
        model.load_state_dict(torch.load(f"{dirname}/model.pt", map_location=map_location))
        return model


# Usage example
# if __name__ == "__main__":
#     obs_dim, act_dim = 10, 4
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
#     # Create models
#     Theta0 = SimpleAgent(obs_dim, act_dim).to(device)
#     Theta_old = SimpleAgent(obs_dim, act_dim).to(device)
#     Theta_new = SimpleAgent(obs_dim, act_dim).to(device)
    
#     # Sample states for policy evaluation
#     states_old = torch.randn(100, obs_dim).to(device)
#     states_new = torch.randn(100, obs_dim).to(device)
    
#     # Perform enhanced merge
#     merged_model = dop_merge_enhanced(
#         Theta0, Theta_old, Theta_new,
#         states_old=states_old,
#         states_new=states_new,
#         K=20,
#         lambda_kl=0.1,
#         lambda_f=0.01
#     )
    
#     print("Enhanced DOP merge complete!")

















# class SimpleAgent(nn.Module):
#     def __init__(self, obs_dim, act_dim):
#         super().__init__()
#         self.act_dim = act_dim
#         self.obs_dim = obs_dim
#         self.fc = shared(input_dim=obs_dim)
#         self.fc_mean = nn.Linear(256, self.act_dim)
#         self.fc_logstd = nn.Linear(256, self.act_dim)

#     def forward(self, x):
#         x = self.fc(x)
#         mean = self.fc_mean(x)
#         log_std = self.fc_logstd(x)
#         return mean, log_std

#     def save(self, dirname):
#         os.makedirs(dirname, exist_ok=True)
#         torch.save(self, f"{dirname}/model.pt")

#     @staticmethod
#     def load(dirname, map_location=None):
#         return torch.load(f"{dirname}/model.pt", map_location=map_location)

# class SequentialMergeAgent:
#     """
#     Orchestrates sequential training and layerwise merging.
#     """
#     def __init__(self, obs_dim, act_dim, device, eval_fn, data_loaders):
#         self.device = device
#         self.obs_dim = obs_dim
#         self.act_dim = act_dim
#         self.eval_fn = eval_fn
#         self.data_loaders = data_loaders  # list of data loaders for each task
#         self.model = SimpleAgent(obs_dim, act_dim).to(device)

#     def train_on_task(self, task_id, train_fn):
#         print(f"[TRAIN] Training on task {task_id} ...")
#         # train_fn should train and return a model
#         model = train_fn(self.obs_dim, self.act_dim, self.device, task_id)
#         print(f"[TRAIN] Finished training on task {task_id}.")
#         return model

#     def sequential_train_and_merge(self, train_fn, verbose=True):
#         for task_id, data_loader in enumerate(self.data_loaders):
#             print(f"[SEQUENTIAL] Starting task {task_id} ...")
#             # Train on new task
#             new_model = self.train_on_task(task_id, train_fn)
#             # Weighted average merge: main model (self.model) and new_model
#             print(f"[SEQUENTIAL] Weighted merging model from task {task_id} ...")
#             merged = weighted_average_merge(self.model, new_model, weight_new=0.6)
#             # Copy merged, train on new task again
#             print(f"[SEQUENTIAL] Training copy of merged model on task {task_id} ...")
#             merged_copy = SimpleAgent(self.obs_dim, self.act_dim).to(self.device)
            
#             merged_copy.load_state_dict(merged.state_dict())
#             retrained_model = train_fn(self.obs_dim, self.act_dim, self.device, task_id, init_model=merged_copy)
#             # Merge again with same weighting
#             print(f"[SEQUENTIAL] Weighted merging retrained model from task {task_id} ...")
#             self.model = weighted_average_merge(merged, retrained_model, weight_new=0.6)
#             print(f"[SEQUENTIAL] Merge after task {task_id} complete.")
#         print("[SEQUENTIAL] All tasks processed and merged.")

#     def evaluate(self, eval_fn=None, verbose=True):
#         # Evaluate on all tasks and return average
#         eval_fn = eval_fn or self.eval_fn
#         scores = []
#         for i, data_loader in enumerate(self.data_loaders):
#             score = eval_fn(self.model, data_loader, self.device)
#             scores.append(score)
#             if verbose:
#                 print(f"[EVAL] Task {i} score: {score:.4f}")
#         avg = sum(scores) / len(scores) if scores else 0.0
#         if verbose:
#             print(f"[EVAL] Average score across all tasks: {avg:.4f}")
#         return avg