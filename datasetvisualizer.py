import matplotlib.pyplot as plt
from pathlib import Path
import json
from dataformat.dataset import DatasetConfig
from dataformat.dataformat import Annotation



def plot_buttons_distribution(annotations: list[Annotation]):
    x_coords = []
    y_coords = []
    for ann in annotations:
        for pair in ann.cloth.pairs:
            x_coords.append(pair.button.bbox.cx)
            y_coords.append(pair.button.bbox.cy)
    plt.scatter(x_coords, y_coords, marker="o", s=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Distribution of the button positions")
    plt.show()


def plot_fastener_size(annotations: list[Annotation], im_size: int = 512):
    widths = []
    heights = []
    for ann in annotations:
        for pair in ann.cloth.pairs:
            widths.append(pair.fastener.bbox.w * im_size)
            heights.append(pair.fastener.bbox.h * im_size)
    plt.scatter(widths, heights, marker="o", s=10)
    plt.title(f"Distribution of the fastener sizes in pixels ({im_size}x{im_size})")
    plt.show()


def get_fastener_size_stats(annotations: list[Annotation], im_size: int = 512):
    widths = []
    heights = []
    for ann in annotations:
        for pair in ann.cloth.pairs:
            widths.append(pair.fastener.bbox.w * im_size)
            heights.append(pair.fastener.bbox.h * im_size)
    return {
        "width": {
            "min": min(widths),
            "max": max(widths),
            "mean": sum(widths) / len(widths),
            "median": sorted(widths)[len(widths) // 2]
        },
        "height": {
            "min": min(heights),
            "max": max(heights),
            "mean": sum(heights) / len(heights),
            "median": sorted(heights)[len(heights) // 2]
        }
    }


def get_button_size_stats(annotations: list[Annotation], im_size: int = 512):
    widths = []
    heights = []
    for ann in annotations:
        for pair in ann.cloth.pairs:
            widths.append(pair.button.bbox.w * im_size)
            heights.append(pair.button.bbox.h * im_size)
    return {
        "width": {
            "min": min(widths),
            "max": max(widths),
            "mean": sum(widths) / len(widths),
            "median": sorted(widths)[len(widths) // 2]
        },
        "height": {
            "min": min(heights),
            "max": max(heights),
            "mean": sum(heights) / len(heights),
            "median": sorted(heights)[len(heights) // 2]
        }
    }


if __name__ == "__main__":
    dataset_path = Path("dataset.json")
    dataset = DatasetConfig.open(config_path=dataset_path).load()
    # training dataset is representative of the dataset so let's use it
    train_dataset = dataset.train_annotations
    plot_buttons_distribution(train_dataset)
    plot_fastener_size(train_dataset)
    fastener_stats = get_fastener_size_stats(train_dataset)
    print("Fastener Size Statistics:")
    for dimension, stats in fastener_stats.items():
        print(f"  {dimension.capitalize()}:")
        for stat, value in stats.items():
            print(f"    {stat.capitalize()}: {value}")

    button_stats = get_button_size_stats(train_dataset)
    print("\nButton Size Statistics:")
    for dimension, stats in button_stats.items():
        print(f"  {dimension.capitalize()}:")
        for stat, value in stats.items():
            print(f"    {stat.capitalize()}: {value}")