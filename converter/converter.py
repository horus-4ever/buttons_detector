from dataclasses import dataclass
from pathlib import Path
import json
import argparse
from typing import Optional
from io import StringIO
from contextlib import redirect_stdout
import random
import shutil
from math import ceil


@dataclass
class Point:
    x: float
    y: float

    def __str__(self):
        return f"{self.x} {self.y}"


@dataclass
class BoundingBox:
    min_point: Point
    max_point: Point

    def __str__(self):
        width = abs(self.max_point.x - self.min_point.x)
        height = abs(self.max_point.y - self.min_point.y)
        return f"{self.min_point} {width} {height}"


@dataclass
class YOLO_POSE:
    class_index: int
    bbox: BoundingBox
    keypoints: list[Point]

    def __str__(self):
        result = f"{self.class_index} {self.bbox}"
        for keypoint in self.keypoints:
            result += f" {keypoint}"
        return result


@dataclass
class YOLO_DETECTION:
    class_index: int
    bbox: BoundingBox

    def __str__(self):
        result = f"{self.class_index} {self.bbox}"
        return result


@dataclass
class YOLO_LIST:
    data: list[YOLO_POSE | YOLO_DETECTION]

    def __str__(self):
        result = ""
        for element in self.data:
            result += f"{element}\n"
        return result
    
    def __iter__(self):
        return iter(self.data)


@dataclass
class DatasetSplit:
    random_seed: int
    training: int
    validation: int
    test: int

    @classmethod
    def from_str(cls, string: str, random_seed: int = 42):
        train, validate, test = map(int, string.split("/"))
        return cls(random_seed, train, validate, test)


@dataclass
class YOLODataset:
    root_path: Path
    train_path: Path
    validation_path: Path
    test_path: Optional[Path]
    classes: list[str]
    dataset_split: DatasetSplit

    def to_str(self) -> str:
        result = StringIO()
        with redirect_stdout(result):
            print(f"path: {self.root_path}\n")
            print(f"train: {self.train_path}")
            print(f"val: {self.validation_path}")
            print(f"test: {self.test_path or ''}\n")
            print(f"nc: {len(self.classes)}\n")
            print(f"names:")
            for i, class_name in enumerate(self.classes):
                print(f"    {i}: {class_name}")
        return result.getvalue()


def get_images_and_annotations(images_path: Path, annotations_path: Path):
    generator = annotations_path.glob("*.json")
    annotations_files = []
    images_files = []
    for annotation_file in generator:
        filename = annotation_file.stem
        image_file = images_path / f"{filename}.png"
        images_files.append(image_file)
        annotations_files.append(annotation_file)
    return annotations_files, images_files


def get_bounding_box(x1: float, y1: float, x2: float, y2: float) -> BoundingBox:
    """
    Get the bounding box from the two given points.
    The returned bounding box is in (ox, oy, width, height)
    """
    # first get the values in order
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    # then get the right coordinates
    origin_x, origin_y = x1, y1
    max_x, max_y = x2, y2
    width, height = (x2 - x1), (y2 - y1)
    # then simply wrap it into objects
    origin = Point(origin_x, origin_y)
    max_point = Point(max_x, max_y)
    return BoundingBox(origin, max_point)


def convert_to_pose(annotation_file: Path):
    """
    Convert to YOLO-pose data format.
    There is only one class (pair button / hole, or velcro / attach).
    """
    with open(annotation_file, "r") as file:
        json_data = json.load(file)
    results = []
    for button_data, keypoint_data in zip(json_data["buttons"], json_data["keypoints"]):
        width, height = json_data["width"], json_data["height"]
        bbox_data = button_data["bbox"]
        # extract the origin of the bounding box, and normalize
        origin_x_px, origin_y_px = bbox_data["x_min_px"], bbox_data["y_min_px"]
        origin_x_ndc, origin_y_ndc = origin_x_px / width, origin_y_px / height
        center_x_px, center_y_px = bbox_data["cx_px"], bbox_data["cy_px"]
        center_x_ndc, center_y_ndc = center_x_px / width, center_y_px / height
        # then get the position of the corresponding keypoint
        kp_x_ndc, kp_y_ndc = keypoint_data["x_ndc"], keypoint_data["y_ndc"]
        # now get the bounding box for that
        class_index = 0 # there is only one class
        bounding_box = get_bounding_box(origin_x_ndc, origin_y_ndc, kp_x_ndc, kp_y_ndc)
        keypoints = [Point(center_x_ndc, center_y_ndc), Point(kp_x_ndc, kp_y_ndc)] # two keypoints per bounding box
        converted = YOLO_POSE(class_index, bounding_box, keypoints)
        results.append(converted)
    return YOLO_LIST(results)


def convert_to_detection(annotation_file: Path):
    with open(annotation_file, "r") as file:
        json_data = json.load(file)
    results = []
    for button_data in json_data["buttons"]:
        width, height = json_data["width"], json_data["height"]
        bbox_data = button_data["bbox"]
        # extract the origin of the bounding box, and normalize
        origin_x_px, origin_y_px = bbox_data["x_min_px"], bbox_data["y_min_px"]
        origin_x_ndc, origin_y_ndc = origin_x_px / width, origin_y_px / height
        max_x_px, max_y_px = bbox_data["x_max_px"], bbox_data["y_max_px"]
        max_x_ndc, max_y_ndc = max_x_px / width, max_y_px / height
        # now get the bounding box for that
        class_index = 0 # there is only one class
        bounding_box = BoundingBox(Point(origin_x_ndc, origin_y_ndc), Point(max_x_ndc, max_y_ndc))
        converted = YOLO_DETECTION(class_index, bounding_box)
        results.append(converted)
    return YOLO_LIST(results)


def ensure_directory(path: Path):
    """
    Ensures that the directory exists.
    """
    if not path.exists():
        path.mkdir(parents=True)
        return True
    return False


def split_dataset(data, split: DatasetSplit):
    random.seed(split.random_seed)
    # now get the splits
    N = len(data) # dataset size
    train, valid, test = ceil(N * split.training / 100), ceil(N * split.validation / 100), ceil(N * split.test / 100)
    # first get the indices by shuffling 
    indices = list(range(len(data)))
    indices = random.shuffle(indices)
    # then get the right data
    train_data = data[:train]
    validation_data = data[train: train + valid]
    test_data = data[train + valid:]
    return train_data, validation_data, test_data


def save_split(data, image_out_dir: Path, annotation_out_dir: Path):
    for image_file, annotations in data:
        # first, we copy the image to the right destination
        image_name = image_file.name
        shutil.copy(image_file, image_out_dir / image_name)
        # now we get the right annotations path
        out_filename = f"{image_file.stem}.txt"
        out_file_path = annotation_out_dir / out_filename
        with open(out_file_path, "w") as out_file:
            print(annotations, file=out_file)


def convert(dataset_path: Path, mode: str, configuration: YOLODataset):
    """
    - dataset_path: Path
    - mode: str
    - configuration: YOLODataset
    """
    if configuration.test_path is None and configuration.dataset_split.test != 0:
        raise RuntimeError("The test path is not set, but the test split is not empty.")
    # first, get the annotations
    images_path = dataset_path / "images"
    annotations_path = dataset_path / "annotations"
    annotations, images = get_images_and_annotations(images_path, annotations_path)
    # then, ensure that the output directories exists
    train_dir = configuration.root_path / configuration.train_path
    validation_dir = configuration.root_path / configuration.validation_path
    test_dir = None if configuration.test_path is None else configuration.root_path / configuration.test_path
    ensure_directory(train_dir)
    ensure_directory(validation_dir)
    if test_dir is not None:
        ensure_directory(test_dir)
    # now we get all the annotations
    all_annotations = []
    for annotation_file, image_file in zip(annotations, images):
        match mode:
            case "pose":
                converted = convert_to_pose(annotation_file)
            case "detection":
                converted = convert_to_detection(annotation_file)
            case _:
                raise RuntimeError(f"This should not happen ({mode})")
        all_annotations.append((image_file, converted))
    # now split the data according
    train_split, valid_split, test_split = split_dataset(all_annotations, configuration.dataset_split)
    # get the right directories for the annotations
    train_ann_dir = configuration.root_path / "labels" / "train"
    valid_ann_dir = configuration.root_path / "labels" / "val"
    test_ann_dir = configuration.root_path / "labels" / "test"
    ensure_directory(train_ann_dir)
    ensure_directory(valid_ann_dir)
    ensure_directory(test_ann_dir)
    save_split(train_split, train_dir, train_ann_dir)
    save_split(valid_split, validation_dir, valid_ann_dir)
    if test_dir is not None:
        save_split(test_split, test_dir, test_ann_dir)
    # now write the configuration file
    dataset_name = dataset_path.stem
    config_file_name = f"{dataset_name}__{mode}.yaml"
    config_file_path = configuration.root_path / config_file_name
    with open(config_file_path, "w") as file:
        print(configuration.to_str(), file=file)


def init_arg_parser():
    parser = argparse.ArgumentParser(
        prog="Converter to YOLO-pose / YOLO-detection",
        description="This program converts the given dataset into the YOLO format for pose or detection"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use.")
    parser.add_argument("--mode", type=str, required=True, help="Either 'pose' for YOLO-pose or 'detection' for YOLO-detection.")
    parser.add_argument("--out", type=str, required=True, help="Output path to write files into.")
    parser.add_argument("--split", type=str, required=True, help="Split train/val/test in percentages. Exemple: '80/20/0'.")
    parser.add_argument("--split-seed", type=int, required=False, default=42, help="Set the random seed for the split.")
    return parser


if __name__ == "__main__":
    parser = init_arg_parser()
    arguments = parser.parse_args()
    # get the arguments
    dataset_path = Path(arguments.dataset)
    mode = arguments.mode
    out = Path(arguments.out)
    dataset_split = arguments.split
    random_seed = arguments.split_seed
    # convert to the object
    configuration = YOLODataset(
        root_path=out,
        train_path=out / "images" / "train",
        validation_path=out / "images" / "val",
        test_path=out / "images" / "test",
        classes=["button"],
        dataset_split=DatasetSplit.from_str(dataset_split, random_seed=random_seed)
    )
    convert(dataset_path, mode, configuration)
