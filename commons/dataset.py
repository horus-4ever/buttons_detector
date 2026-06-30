import torch
from torch.utils.data import Dataset, DataLoader, Subset
from pathlib import Path


def load_dataset(path: Path, check: bool = True):
    images_path = path / "images"
    annotations_path = path / "annotations"
    # get all annotations paths
    annotations_files = annotations_path.glob("*.json")
    