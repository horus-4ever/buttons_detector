from pathlib import Path
from dataclasses import dataclass
import numpy as np
import shutil
from argparse import ArgumentParser


def create_dataset_split(dataset: Path, split: int, output: Path):
    """
    Create a dataset split from 'annotations/' and 'images/' folders.
    It splits the dataset into 'split' equal parts.
    """
    images_folder = dataset / "images"
    annotations_folder = dataset / "annotations"
    images = list(Path(im.name) for im in images_folder.glob("*.png"))
    annotations = list(path.with_suffix(".json") for path in images)
    length = len(images)
    # now lets define the indices
    indices = np.arange(length)
    np.random.shuffle(indices)
    ratio = length / split
    for n in range(split):
        name = f"part_{n}"
        out_img_folder = (output / name / "images")
        out_ann_folder = (output / name / "annotations")
        out_img_folder.mkdir(parents=True, exist_ok=True)
        out_ann_folder.mkdir(parents=True, exist_ok=True)
        part_indices = indices[int(n * ratio) : int((n+1) * ratio)]
        for i in part_indices:
            shutil.copy(images_folder / images[i], out_img_folder / images[i])
            shutil.copy(annotations_folder / annotations[i], out_ann_folder / annotations[i])


def init_parser():
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--split", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    create_dataset_split(args.dataset, args.split, args.output)