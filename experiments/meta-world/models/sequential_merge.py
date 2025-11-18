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

def _top_r_svd(mat, r):
    if torch.norm(mat) == 0:
        m, n = mat.shape
        return (torch.zeros((m, min(r, n)), device=mat.device, dtype=mat.dtype),
                torch.zeros((min(r, n),), device=mat.device, dtype=mat.dtype),
                torch.zeros((n, min(r, n)), device=mat.device, dtype=mat.dtype))
    U, S, Vh = torch.linalg.svd(mat, full_matrices=False)
    r = min(r, S.numel())
    return U[:, :r], S[:r], Vh[:r, :].T

def _project_onto_svd_subspace(X, U, V):
    # U: (m,r), V: (n,r), X: (m,n)
    if U.numel() == 0 or V.numel() == 0:
        return torch.zeros_like(X)
    coeffs = torch.einsum('ir,ij,jr->r', U, X, V)
    proj = torch.zeros_like(X)
    for i in range(U.shape[1]):
        ui = U[:, i].unsqueeze(1)
        vi = V[:, i].unsqueeze(1)
        proj = proj + coeffs[i] * (ui @ vi.T)
    return proj

def dop_merge_simple(
    Theta0,         # base pretrained model (nn.Module)
    Theta_old,      # previously merged / main model (SimpleAgent)
    Theta_new,      # new trained model (SimpleAgent)
    K=20,
    r=8,
    beta=0.9,
    eta=1e-2,
    clip_eps=1e-8,
):
    """
    Dual Orthogonal Projection merge tailored to SimpleAgent-like models.
    Returns a SimpleAgent instance with merged parameters.
    """
    # device/dtype
    device = next(Theta_old.parameters()).device
    dtype = next(Theta_old.parameters()).dtype

    state0 = Theta0.state_dict()
    state_old = Theta_old.state_dict()
    state_new = Theta_new.state_dict()

    merged = type(Theta_old)(Theta_old.obs_dim, Theta_old.act_dim).to(device)
    # merged.reset_heads()  # uncomment if you want freshly initialized heads

    merged_state = merged.state_dict()

    for name in merged_state.keys():
        W0 = state0.get(name, None)
        Wold = state_old[name].to(device=device, dtype=dtype)
        Wnew = state_new.get(name, None)
        if Wnew is None:
            merged_state[name] = Wold.clone()
            continue
        Wnew = Wnew.to(device=device, dtype=dtype)

        # treat 2D tensors as linear weights eligible for SVD-based merge
        if Wold.ndim == 2 and Wnew.ndim == 2 and (W0 is not None and getattr(W0, "ndim", 0) == 2):
            W0 = W0.to(device=device, dtype=dtype)

            tau_old = (Wold - W0).detach()
            tau_new = (Wnew - W0).detach()

            Uo, So, Vo = _top_r_svd(tau_old, r)
            Un, Sn, Vn = _top_r_svd(tau_new, r)

            Wstar = ((Wold + Wnew) / 2.0).clone().to(device=device, dtype=dtype)

            alpha_s_prev = 0.5
            alpha_p_prev = 0.5

            for k in range(K):
                Wstar.requires_grad_(True)

                delta_old = Wstar - Wold
                delta_new = Wstar - Wnew

                proj_old = _project_onto_svd_subspace(delta_old, Uo, Vo)
                proj_new = _project_onto_svd_subspace(delta_new, Un, Vn)

                Ls = 0.5 * torch.sum((delta_old - proj_old) ** 2)
                Lp = 0.5 * torch.sum((delta_new - proj_new) ** 2)

                grad_Ls = torch.autograd.grad(Ls, Wstar, retain_graph=False, create_graph=False)[0]
                grad_Lp = torch.autograd.grad(Lp, Wstar, retain_graph=False, create_graph=False)[0]

                g_s = grad_Ls.detach().view(-1)
                g_p = grad_Lp.detach().view(-1)
                diff = g_s - g_p
                denom = (diff @ diff).clamp_min(clip_eps)
                numer = ((g_p - g_s) @ g_p)
                alpha_k = float((numer / denom).clamp(0.0, 1.0))

                alpha_s_k = beta * alpha_s_prev + (1.0 - beta) * alpha_k
                alpha_p_k = beta * alpha_p_prev + (1.0 - beta) * (1.0 - alpha_k)

                gk = alpha_s_k * grad_Ls + alpha_p_k * grad_Lp

                with torch.no_grad():
                    Wstar = (Wstar - eta * gk).detach().to(device=device, dtype=dtype)

                alpha_s_prev = alpha_s_k
                alpha_p_prev = alpha_p_k

            merged_state[name] = Wstar.clone().detach()

        else:
            # fallback: simple average for biases and non-2D params
            merged_state[name] = ((Wold + Wnew) / 2.0).clone().to(device=device, dtype=dtype)

    # load merged state
    merged.load_state_dict(merged_state)
    return merged


  




















class SimpleAgent(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.act_dim = act_dim
        self.obs_dim = obs_dim
        self.fc = shared(input_dim=obs_dim)
        self.fc_mean = nn.Linear(256, self.act_dim)
        self.fc_logstd = nn.Linear(256, self.act_dim)

    def forward(self, x):
        x = self.fc(x)
        mean = self.fc_mean(x)
        log_std = self.fc_logstd(x)
        return mean, log_std

    def save(self, dirname):
        os.makedirs(dirname, exist_ok=True)
        torch.save(self, f"{dirname}/model.pt")

    @staticmethod
    def load(dirname, map_location=None):
        return torch.load(f"{dirname}/model.pt", map_location=map_location)

class SequentialMergeAgent:
    """
    Orchestrates sequential training and layerwise merging.
    """
    def __init__(self, obs_dim, act_dim, device, eval_fn, data_loaders):
        self.device = device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.eval_fn = eval_fn
        self.data_loaders = data_loaders  # list of data loaders for each task
        self.model = SimpleAgent(obs_dim, act_dim).to(device)

    def train_on_task(self, task_id, train_fn):
        print(f"[TRAIN] Training on task {task_id} ...")
        # train_fn should train and return a model
        model = train_fn(self.obs_dim, self.act_dim, self.device, task_id)
        print(f"[TRAIN] Finished training on task {task_id}.")
        return model

    def sequential_train_and_merge(self, train_fn, verbose=True):
        for task_id, data_loader in enumerate(self.data_loaders):
            print(f"[SEQUENTIAL] Starting task {task_id} ...")
            # Train on new task
            new_model = self.train_on_task(task_id, train_fn)
            # Weighted average merge: main model (self.model) and new_model
            print(f"[SEQUENTIAL] Weighted merging model from task {task_id} ...")
            merged = weighted_average_merge(self.model, new_model, weight_new=0.6)
            # Copy merged, train on new task again
            print(f"[SEQUENTIAL] Training copy of merged model on task {task_id} ...")
            merged_copy = SimpleAgent(self.obs_dim, self.act_dim).to(self.device)
            
            merged_copy.load_state_dict(merged.state_dict())
            retrained_model = train_fn(self.obs_dim, self.act_dim, self.device, task_id, init_model=merged_copy)
            # Merge again with same weighting
            print(f"[SEQUENTIAL] Weighted merging retrained model from task {task_id} ...")
            self.model = weighted_average_merge(merged, retrained_model, weight_new=0.6)
            print(f"[SEQUENTIAL] Merge after task {task_id} complete.")
        print("[SEQUENTIAL] All tasks processed and merged.")

    def evaluate(self, eval_fn=None, verbose=True):
        # Evaluate on all tasks and return average
        eval_fn = eval_fn or self.eval_fn
        scores = []
        for i, data_loader in enumerate(self.data_loaders):
            score = eval_fn(self.model, data_loader, self.device)
            scores.append(score)
            if verbose:
                print(f"[EVAL] Task {i} score: {score:.4f}")
        avg = sum(scores) / len(scores) if scores else 0.0
        if verbose:
            print(f"[EVAL] Average score across all tasks: {avg:.4f}")
        return avg