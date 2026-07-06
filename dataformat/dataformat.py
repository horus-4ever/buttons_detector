from dataclasses import dataclass, field
import json
from pathlib import Path
import numpy as np


@dataclass
class BoundingBox:
    """
    Represents a bounding box. `cx` and `cy` are the center coordinates.
    `w` and `h` are respectively the width and the height of the bounding box.
    """
    cx: float
    cy: float
    w: float
    h: float

    @classmethod
    def from_json(cls, json_data: dict) -> "BoundingBox":
        return cls(
            cx=json_data["cx_ndc"],
            cy=json_data["cy_ndc"],
            w=json_data["width_ndc"],
            h=json_data["height_ndc"]
        )
    
    def to_x1y1x2y2(self) -> tuple[float, float, float, float]:
        """
        Converts the bounding box from center coordinates to corner coordinates.
        Returns a tuple of (x1, y1, x2, y2).
        """
        x1 = self.cx - self.w / 2
        y1 = self.cy - self.h / 2
        x2 = self.cx + self.w / 2
        y2 = self.cy + self.h / 2
        return (x1, y1, x2, y2)


@dataclass
class Button:
    """
    Represents a button.
    """
    bbox: BoundingBox
    visible: bool

    @classmethod
    def from_json(cls, json_data: dict) -> "Button":
        return cls(
            bbox=BoundingBox.from_json(json_data["bbox"]),
            visible=json_data["visible"]
        )


@dataclass
class Fastener:
    """
    Represents a fastener.
    A fastener will be generally a velcro, a button hole or a snap button.
    """
    bbox: BoundingBox
    visible: bool
    type: str

    @classmethod
    def from_json(cls, json_data: dict) -> "Fastener":
        return cls(
            bbox=BoundingBox.from_json(json_data["bbox"]),
            visible=json_data["visible"],
            type=json_data["type"]
        )


@dataclass
class Pair:
    """
    Represents a pair between a button and a fastener.
    """
    button: Button
    fastener: Fastener

    @classmethod
    def from_json(cls, json_data: dict) -> "Pair":
        return cls(
            button=Button.from_json(json_data["button"]),
            fastener=Fastener.from_json(json_data["fastener"])
        )


@dataclass
class Cloth:
    """
    Represent a clothing item, with its pairs of <button, fastener>.
    `segmentation` represents the path to the segmentation mask of the clothing item.
    """
    type: str
    segmentation: str
    pairs: list[Pair]

    @classmethod
    def from_json(cls, json_data: dict) -> "Cloth":
        return cls(
            type=json_data["type"],
            segmentation=json_data["segmentation"],
            pairs=[Pair.from_json(pair) for pair in json_data["pairs"]]
        )


@dataclass
class ImageInfo:
    """
    Represents an image in the dataset.
    """
    url: str
    width: int
    height: int

    @classmethod
    def from_json(cls, json_data: dict) -> "ImageInfo":
        return cls(
            url=json_data["url"],
            width=json_data["width"],
            height=json_data["height"]
        )


@dataclass
class Annotation:
    """
    Represents an annotation of a clothing item in an image.
    """
    image: ImageInfo
    clothing_item: Cloth

    @classmethod
    def from_json(cls, json_data: dict) -> "Annotation":
        return cls(
            image=ImageInfo.from_json(json_data["image"]),
            clothing_item=Cloth.from_json(json_data["clothing_item"])
        )


@dataclass
class DataSplit:
    """
    Represents a split of the dataset.
    """
    seed: int
    train: float
    val: float
    test: float

    @classmethod
    def from_json(cls, json_data: dict) -> "DataSplit":
        return cls(
            seed=json_data["seed"],
            train=json_data["train"],
            val=json_data["val"],
            test=json_data["test"]
        )


@dataclass
class Dataset:
    """
    Represents the dataset.
    """
    dataset_root: Path
    images_dir: Path
    annotations_dir: Path
    data_split: DataSplit
    train_indices: list[int] = field(default_factory=list)
    validation_indices: list[int] = field(default_factory=list)
    test_indices: list[int] = field(default_factory=list)

    @classmethod
    def from_json(cls, json_data: dict) -> "Dataset":
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
    def open(cls, config_path: Path) -> "Dataset":
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
