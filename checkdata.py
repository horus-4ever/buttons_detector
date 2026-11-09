from pathlib import Path
from dataformat.dataset import DatasetConfig, Annotation
import json

def check_dataset(path: Path):
    try:
        dataset_config = DatasetConfig.open(path)
        dataset_config.load()
        print(f"Dataset at {path} is valid.")
    except Exception as e:
        print(f"Dataset at {path} is invalid: {e}")


def check_annotation(path: Path):
    try:
        with path.open(encoding="utf-8") as file:
            annotation = Annotation.from_json(json.load(file))
        return True
    except Exception as e:
        return False


def check_garment_directory(path: Path):
    json_files = path / "annotations"
    for json_file in json_files.glob("*.json"):
        if not check_annotation(json_file):
            print(f"Invalid annotation found: {json_file}")
            return False
    print(f"All annotations in {path} are valid.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Check dataset and annotations.")
    parser.add_argument("path", type=Path, help="Path to the dataset.")
    args = parser.parse_args()

    if args.path.is_dir():
        for subdir in args.path.iterdir():
            if subdir.is_dir():
                check_garment_directory(subdir)
    else:
        print(f"Invalid path: {args.path}. Must be a directory.")