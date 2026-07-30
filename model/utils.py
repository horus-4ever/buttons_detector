from torch import nn, Tensor
import torch.nn.functional as F
import torch
from pathlib import Path


class FFN(nn.Module):
    def __init__(
            self,
            d_model: int,
            d_hidden: int,
            dropout: float = 0.1,
            activation: str = "relu"
    ):
        super().__init__()
        # network layers
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_hidden, d_model)
        # activation function
        self.activation = _get_activation_fn(activation)

    def forward(self, input):
        result = self.linear2(self.dropout(self.activation(self.linear1(input))))
        return result


class AddNorm(nn.Module):
    def __init__(
            self,
            d_model: int,
            dropout: float = 0.1
    ):
        super().__init__()
        self.norm_layer = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, add_target):
        add_target = self.dropout(add_target)
        return self.norm_layer(input + add_target)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers):
            in_dim = input_dim if i == 0 else hidden_dim
            out_dim = output_dim if i == num_layers - 1 else hidden_dim
            layers.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


def _get_activation_fn(name: str):
    match name:
        case "relu":
            return F.relu
        case "gelu":
            return F.gelu
        case _:
            return F.relu
        

def inverse_sigmoid(x, eps=1e-6):
    """
    Converts a probability back into a real number.
    Value is clamped to avoid overflows.
    """
    x = x.clamp(min=eps, max=1.0 - eps)
    return torch.log(x / (1.0 - x))


def load_weights(model, weights: Path, device):
    """
    Loads the model weigths into the model.
    """
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    checkpoint = torch.load(weights, map_location=device)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)


def compute_intersect(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Computes the intersection area between two boxes.
    - box1: [B, 4] (cx, cy, w, h)
    - box2: [B, 4] (cx, cy, w, h)
    """
    # computes the intersection area between two boxes
    x1 = torch.max(box1[..., 0] - box1[..., 2] / 2, box2[..., 0] - box2[..., 2] / 2)
    y1 = torch.max(box1[..., 1] - box1[..., 3] / 2, box2[..., 1] - box2[..., 3] / 2)
    x2 = torch.min(box1[..., 0] + box1[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2)
    y2 = torch.min(box1[..., 1] + box1[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2)
    inter_area = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0)
    return inter_area

def compute_union(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Computes the union area between two boxes.
    - box1: [B, 4] (cx, cy, w, h)
    - box2: [B, 4] (cx, cy, w, h)
    """
    # computes the union area between two boxes
    area1 = box1[..., 2] * box1[..., 3]
    area2 = box2[..., 2] * box2[..., 3]
    inter_area = compute_intersect(box1, box2)
    union_area = area1 + area2 - inter_area
    return union_area


def compute_iou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Computes the Intersection over Union between two boxes.
    - box1: [B, 4] (cx, cy, w, h)
    - box2: [B, 4] (cx, cy, w, h)
    """
    intersection = compute_intersect(box1, box2)
    union = compute_union(box1, box2)
    iou = intersection / (union + 1e-6) # add a small epsilon to avoid zero divisions
    return iou


def compute_giou(box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
    """
    Compute the Generalized Intersection over Union between two boxes.
    - box1: [B, 4] (cx, cy, w, h)
    - box2: [B, 4] (cx, cy, w, h)
    """
    # first we need to compute the large box that contains both boxes
    # we can do that by taking the min and max of the corners of the boxes
    x1 = torch.min(box1[..., 0] - box1[..., 2] / 2, box2[..., 0] - box2[..., 2] / 2)
    y1 = torch.min(box1[..., 1] - box1[..., 3] / 2, box2[..., 1] - box2[..., 3] / 2)
    x2 = torch.max(box1[..., 0] + box1[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2)
    y2 = torch.max(box1[..., 1] + box1[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2)
    # compute the IoU then the GIoU
    iou = compute_iou(box1, box2)
    union = compute_union(box1, box2)
    enclosing_width = torch.clamp(x2 - x1, min=0)
    enclosing_height = torch.clamp(y2 - y1, min=0)
    enclosing_area = enclosing_width * enclosing_height
    giou = iou - (enclosing_area - union) / (enclosing_area + 1e-6)
    return giou


def compute_pair_iou(b1: torch.Tensor, gt1: torch.Tensor, b2: torch.Tensor, gt2: torch.Tensor):
    """
    Compute the IoU of a pair of predictions.
    Boxes are all in format: [B, 4] (cx, cy, w, h)
    """
    left = compute_iou(b1, gt1)
    right = compute_iou(b2, gt2)
    return torch.min(left, right)