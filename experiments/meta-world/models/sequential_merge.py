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