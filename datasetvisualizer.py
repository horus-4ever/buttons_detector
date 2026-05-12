import matplotlib.pyplot as plt
from pathlib import Path
import json


def get_data(path: Path):
    ann_paths = path.glob("*.json")
    x_coords = []
    y_coords = []
    for ann_path in ann_paths:
        with open(ann_path, "r") as file:
            data = json.load(file)
            for button in data["buttons"]:
                x_coords.append(button["x_ndc"])
                y_coords.append(button["y_ndc"])
    return x_coords, y_coords


def plot_buttons_distribution(coords):
    x_coords, y_coords = coords
    plt.scatter(x_coords, y_coords, marker="o", s=10)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.title("Distribution of the button positions")
    plt.show()


if __name__ == "__main__":
    dataset_path = Path("dataset/annotations")
    coordinates = get_data(dataset_path)
    plot_buttons_distribution(coordinates)