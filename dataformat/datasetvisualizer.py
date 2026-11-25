import math

import matplotlib.pyplot as plt
from pathlib import Path
import json
from dataformat.dataset import DatasetConfig
from dataformat.dataformat import Annotation
from sklearn.cluster import KMeans
import numpy as np



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


def plot_counterpart_size(annotations: list[Annotation], im_size: int = 512):
    widths = []
    heights = []
    for ann in annotations:
        for pair in ann.cloth.pairs:
            widths.append(pair.counterpart.bbox.w * im_size)
            heights.append(pair.counterpart.bbox.h * im_size)
    plt.scatter(widths, heights, marker="o", s=10)
    plt.title(f"Distribution of the counterpart sizes in pixels ({im_size}x{im_size})")
    plt.show()


def get_counterpart_size_stats(annotations: list[Annotation], im_size: int = 512):
    widths = []
    heights = []
    diagonals = []
    total_number = 0
    n_fastened = 0
    n_invisible = 0
    n_visible = 0
    for ann in annotations:
        for pair in ann.cloth.pairs:
            total_number += 1
            widths.append(pair.counterpart.bbox.w * im_size)
            heights.append(pair.counterpart.bbox.h * im_size)
            diagonals.append(math.sqrt((pair.counterpart.bbox.w * im_size) ** 2 + (pair.counterpart.bbox.h * im_size) ** 2))
            if pair.fastened:
                n_fastened += 1
            elif pair.counterpart.visible:
                n_visible += 1
            elif not pair.counterpart.visible:
                n_invisible += 1
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
        },
        "diagonals": {
            "min": min(diagonals),
            "max": max(diagonals),
            "mean": sum(diagonals) / len(diagonals),
            "median": sorted(diagonals)[len(diagonals) // 2]
        },
        "visibility": {
            "fastened": f"{n_fastened / total_number * 100:.2f}",
            "visible": f"{n_visible / total_number * 100:.2f}",
            "invisible": f"{n_invisible / total_number * 100:.2f}"
        }
    }

def plot_counterpart_kmeans(annotations: list[Annotation], im_size: int = 512):
    widths = []
    heights = []
    for ann in annotations:
        for pair in ann.cloth.pairs:
            widths.append(pair.counterpart.bbox.w)
            heights.append(pair.counterpart.bbox.h)
    input = np.column_stack([np.array(widths), np.array(heights)])
    
    kmeans = KMeans(n_clusters=3, init="k-means++", n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(input)
    centroids = kmeans.cluster_centers_
    print(centroids * np.array([im_size, im_size]))
    plt.figure(figsize=(8, 6))
    # Plot data points colored by their cluster label
    plt.scatter(
        input[:, 0], input[:, 1], c=cluster_labels, s=50, cmap="viridis"
    )
    plt.show()


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
