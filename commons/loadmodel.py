import torch
from torch import nn
from pathlib import Path


def load_weights(path_to_weights: str, device: torch.device):
    """
    Load the saved state dict of the model.
    """
    path = Path(path_to_weights)
    if not path.exists():
        raise FileNotFoundError(f"Weights '{path}' doesn't exist.")
    # now load the weights
    checkpoint = torch.load(path, map_location=device)
    return checkpoint


def load_for_resume(model: nn.Module, optimizer: nn.Module, scheduler: nn.Module, path_to_weights: str, device: torch.device):
    """
    Load the model, optimizer and scheduler to resume the training.
    Returns the state dict of the saved model.
    """
    checkpoint = load_weights(path_to_weights, device)
    # now set back the state dicts
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
    return checkpoint


def load_for_finetune(model: nn.Module, path_to_weights: str, device: torch.device):
    """
    Load the model for finetuning. Optimizer and scheduler are ignored.
    Returns the state dict of the saved model.
    """
    checkpoint = load_weights(path_to_weights, device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
