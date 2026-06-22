from dataclasses import dataclass
from pathlib import Path
import json
import sys
import argparse
import os


@dataclass
class Point:
    x: float
    y: float

    def __str__(self):
        return f"{self.x} {self.y}"


@dataclass
class BoundingBox:
    origin: Point
    width: float
    height: float

    def __str__(self):
        return f"{self.origin} {self.width} {self.height}"


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


def get_images_and_annotations(images_path: Path, annotations_path: Path):
    generator = annotations_path.glob("*.json")
    annotations_files = []
    images_files = []
    for annotation_file in generator:
        filename = annotation_file.stem
        image_file = images_path / f"{filename}.jpg"
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
    width, height = (x2 - x1), (y2 - y1)
    # then simply wrap it into objects
    origin = Point(origin_x, origin_y)
    return BoundingBox(origin, width, height)


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
    return results


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
        width_px, height_px = bbox_data["width"], bbox_data["height"]
        width_ndc, height_ndc = width_px / width, height_px / height
        # now get the bounding box for that
        class_index = 0 # there is only one class
        bounding_box = BoundingBox(Point(origin_x_ndc, origin_y_ndc), width_ndc, height_ndc)
        converted = YOLO_DETECTION(class_index, bounding_box)
        results.append(converted)
    return results


def convert(dataset_path: Path, mode: str, out_directory: Path):
    """
    - dataset_path: Path
    - mode: str
    - out: FILE
    """
    images_path = dataset_path / "images"
    annotations_path = dataset_path / "annotations"
    annotations, images = get_images_and_annotations(images_path, annotations_path)
    for annotation_file, image_file in zip(annotations, images):
        if mode == "pose":
            converted = convert_to_pose(annotation_file)
        else:
            converted = convert_to_detection(annotation_file)
        # now get the filename for the output
        image_name = image_file.stem
        out_file = out_directory / f"{image_name}.txt"
        with open(out_file, "w") as out:
            for yolo_format in converted:
                print(yolo_format, file=out)


def init_arg_parser():
    parser = argparse.ArgumentParser(
        prog="Converter to YOLO-pose / YOLO-detection",
        description="This program converts the given dataset into the YOLO format for pose or detection"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use.")
    parser.add_argument("--mode", type=str, required=True, help="Either 'pose' for YOLO-pose or 'detection' for YOLO-detection.")
    parser.add_argument("--out", type=str, required=True, help="Output path to write files into.")
    return parser


if __name__ == "__main__":
    parser = init_arg_parser()
    arguments = parser.parse_args()
    # get the arguments
    dataset_path = Path(arguments.dataset)
    mode = arguments.mode
    out = Path(arguments.out)
    dataset_path = Path("/media/Data/Documents/Etudes/九工大/Shibata LAB/Lab Projects/DressingAssistant/Clothes_blender/out")
    convert(dataset_path, mode, out)
    # python3 converter/converter.py --dataset "/media/Data/Documents/Etudes/九工大/Shibata LAB/Lab Projects/DressingAssistant/Clothes_blender/out" --mode pose --out converter/script_tests/
