from .dataformat import *
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from typing import Callable
from PIL import Image
from abc import ABC, abstractmethod


class Transform(ABC):
    @abstractmethod
    def __call__(self, image: Image.Image, annotations: Annotation):
        raise NotImplemented


class EmptyTransform(Transform):
    def __call__(self, image: Image.Image, annotations: Annotation):
        return image, annotations


class PairDataset(Dataset):
    def __init__(self, root: "DatasetConfig", annotations: list[Annotation], transform: Transform):
        self.root = root
        self.annotations = annotations
        self.transform = transform

    def __len__(self):
        """
        Returns the length of the dataset.
        Also, with `__getitem__`, defines the len / getitem iterable protocol.
        """
        return len(self.annotations)

    def __getitem__(self, index) -> tuple[Image.Image, Annotation]:
        """
        Get the image and annotation at the given index.
        """
        annotation: Annotation = self.annotations[index]
        # open the image and get the annotations as torch tensors
        image_path = annotation.image.url
        image = Image.open(image_path)
        # transform the image if any transformation needs to be applied
        if self.transform:
            image, annotation = self.transform(image, annotation)
        return image, annotation


@dataclass
class DatasetConfig:
    """
    Represents the dataset configuration.
    """
    dataset_root: Path
    train_paths: list[Path]
    val_paths: list[Path]
    test_paths: list[Path]

    def to_json(self) -> dict:
        return {
            "root": self.dataset_root,
            "train": self.train_paths,
            "validation": self.val_paths,
            "test": self.test_paths
        }

    @classmethod
    def from_json(cls, json_data: dict) -> "DatasetConfig":
        dataset_root = Path(json_data["root"])
        train_paths = [
            dataset_root / garment_name for garment_name in json_data["train"]
        ]
        val_paths = [
            dataset_root / garment_name for garment_name in json_data["validation"]
        ]
        test_paths = [
            dataset_root / garment_name for garment_name in json_data["test"]
        ]
        return cls(
            dataset_root=dataset_root,
            train_paths=train_paths,
            val_paths=val_paths,
            test_paths=test_paths
        )
    
    @classmethod
    def open(cls, config_path: Path) -> "DatasetConfig":
        """
        Opens a dataset from a JSON file.
        """
        with open(config_path, "r") as f:
            json_data = json.load(f)
        return cls.from_json(json_data)

    def _was_loaded(self):
        return hasattr(self, "_train_annotations")

    @property
    def train_annotations(self) -> list[Annotation]:
        if not self._was_loaded():
            raise ValueError("Dataset not loaded. Call `load()` first.")
        return self._train_annotations

    @property
    def validation_annotations(self) -> list[Annotation]:
        if not self._was_loaded():
            raise ValueError("Dataset not loaded. Call `load()` first.")
        return self._validation_annotations

    @property
    def test_annotations(self) -> list[Annotation]:
        if not self._was_loaded():
            raise ValueError("Dataset not loaded. Call `load()` first.")
        return self._test_annotations

    def _load_annotations(self, path: Path):
        images_directory = path / "images"
        annotations_directory = path / "annotations"
        annotations = []
        for annotation_file in annotations_directory.glob("*.json"):
            with open(annotation_file, "r") as file:
                json_data = json.load(file)
            annotation = Annotation.from_json(json_data)
            # now change the image url
            annotation.image.url = str(images_directory / annotation.image.url)
            annotations.append(annotation)
        return annotations

    def load(self):
        """
        Loads the dataset.
        """
        train_annotations = []
        validation_annotations = []
        test_annotations = []
        for train_path in self.train_paths:
            train_path = self.dataset_root / train_path
            annotations = self._load_annotations(train_path)
            train_annotations.extend(annotations)
        for val_path in self.val_paths:
            val_path = self.dataset_root / val_path
            annotations = self._load_annotations(val_path)
            validation_annotations.extend(annotations)
        for test_path in self.test_paths:
            test_path = self.dataset_root / test_path
            annotations = self._load_annotations(test_path)
            test_annotations.extend(annotations)
        self._train_annotations = train_annotations
        self._validation_annotations = validation_annotations
        self._test_annotations = test_annotations
        return self

    def to_torch_dataset(self):
        """
        Returns the training, validation and test dataset as torch `Dataset` objects.
        """
        train_dataset = PairDataset(self, self.train_annotations, transform=EmptyTransform())
        val_dataset = PairDataset(self, self.validation_annotations, transform=EmptyTransform())
        test_dataset = PairDataset(self, self.test_annotations, transform=EmptyTransform())
        return train_dataset, val_dataset, test_dataset
