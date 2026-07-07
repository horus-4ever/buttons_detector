
from .dataformat import *
import json
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from typing import Callable
from PIL import Image


class PairDataset(Dataset):
    def __init__(self, root: "DatasetConfig", annotations: list[Annotation], transform: Callable | None = None):
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
        image_path = self.root.images_dir / annotation.image.url
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
    images_dir: Path
    annotations_dir: Path
    data_split: DataSplit
    train_indices: list[int] = field(default_factory=list)
    validation_indices: list[int] = field(default_factory=list)
    test_indices: list[int] = field(default_factory=list)

    def to_json(self) -> dict:
        return {
            "dataset_root": self.dataset_root,
            "images_dir": self.images_dir,
            "annotations_dir": self.annotations_dir,
            "data_split": self.data_split
        }

    @classmethod
    def from_json(cls, json_data: dict) -> "DatasetConfig":
        dataset_root = Path(json_data["root"])
        images_dir = dataset_root / json_data["images_dir"]
        annotations_dir = dataset_root / json_data["annotations_dir"]

        return cls(
            dataset_root=dataset_root,
            images_dir=images_dir,
            annotations_dir=annotations_dir,
            data_split=DataSplit.from_json(json_data["data_split"]),
        )
    
    @classmethod
    def open(cls, config_path: Path) -> "DatasetConfig":
        """
        Opens a dataset from a JSON file.
        """
        with open(config_path, "r") as f:
            json_data = json.load(f)
        return cls.from_json(json_data)
    
    def _has_split_cache(self) -> bool:
        """
        Checks if the dataset has a split cache.
        """
        return Path(self.dataset_root / "dataset.cache").exists()
    
    def _create_split_cache(self):
        """
        Creates a split cache for the dataset
        """
        cache_path = self.dataset_root / "dataset.cache"
        # now we create the split indices and save them in the cache
        annotations = np.array(list(self.annotations_dir.glob("*.json")))
        dataset_length = len(annotations)
        train_size = int(dataset_length * self.data_split.train)
        val_size = int(dataset_length * self.data_split.val)
        test_size = dataset_length - train_size - val_size
        # set the seed for reproducibility
        np.random.seed(self.data_split.seed)
        indices = np.arange(dataset_length)
        np.random.shuffle(indices)
        self.train_indices = indices[:train_size].tolist()
        self.validation_indices = indices[train_size:train_size + val_size].tolist()
        self.test_indices = indices[train_size + val_size:].tolist()
        # save the split indices in the cache
        with open(cache_path, "w") as f:
            print("train:", file=f)
            print(*annotations[self.train_indices], sep="\n", file=f)
            print("validation:", file=f)
            print(*annotations[self.validation_indices], sep="\n", file=f)
            print("test:", file=f)
            print(*annotations[self.test_indices], sep="\n", file=f)

    def _load_dataset(self):
        """
        Loads the dataset from the split cache.
        """
        cache_path = self.dataset_root / "dataset.cache"
        with open(cache_path, "r") as f:
            lines = f.readlines()
        # find the indices of the split sections
        train_start = lines.index("train:\n") + 1
        validation_start = lines.index("validation:\n") + 1
        test_start = lines.index("test:\n") + 1
        # load the train, validation and test annotations paths
        train_paths = [Path(line.strip()) for line in lines[train_start:validation_start - 1]]
        validation_paths = [Path(line.strip()) for line in lines[validation_start:test_start - 1]]
        test_paths = [Path(line.strip()) for line in lines[test_start:]]
        # now try to load the dataset from the split cache
        train_annotations = []
        validation_annotations = []
        test_annotations = []
        print("Loading dataset from split cache...")
        for path in train_paths:
            with open(path, "r") as f:
                json_data = json.load(f)
            train_annotations.append(Annotation.from_json(json_data))
        for path in validation_paths:
            with open(path, "r") as f:
                json_data = json.load(f)
            validation_annotations.append(Annotation.from_json(json_data))
        for path in test_paths:
            with open(path, "r") as f:
                json_data = json.load(f)
            test_annotations.append(Annotation.from_json(json_data))
        print("Dataset loaded from split cache.")
        # set it as cache on the object
        self._train_annotations = train_annotations
        self._validation_annotations = validation_annotations
        self._test_annotations = test_annotations

    @property
    def train_annotations(self) -> list[Annotation]:
        if not hasattr(self, "_train_annotations"):
            raise ValueError("Dataset not loaded. Call `check_dataset()` first.")
        return self._train_annotations

    @property
    def validation_annotations(self) -> list[Annotation]:
        if not hasattr(self, "_validation_annotations"):
            raise ValueError("Dataset not loaded. Call `check_dataset()` first.")
        return self._validation_annotations

    @property
    def test_annotations(self) -> list[Annotation]:
        if not hasattr(self, "_test_annotations"):
            raise ValueError("Dataset not loaded. Call `check_dataset()` first.")
        return self._test_annotations

    def check_dataset(self):
        """
        Checks if the dataset has a split cache, and creates one if it doesn't exist.
        """
        if not self._has_split_cache():
            print("No split cache found. Creating one...")
            self._create_split_cache()
            print("Split cache created.")

    def to_torch_dataset(self):
        """
        Returns the training, validation and test dataset as torch `Dataset` objects.
        """
        train_dataset = PairDataset(self, self.train_annotations)
        val_dataset = PairDataset(self, self.validation_annotations)
        test_dataset = PairDataset(self, self.test_annotations)
        return train_dataset, val_dataset, test_dataset
