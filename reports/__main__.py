import torch
from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt
import numpy as np
from argparse import ArgumentParser

from model.ptmodel import PTModel

def load_weights(weights: Path, device):
    """
    Loads the model weigths.
    """
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    checkpoint = PTModel.from_checkpoint(checkpoint)
    return checkpoint


def flatten(data: list[dict]):
    results = {}
    for elem in data:
        for key, value in elem.items():
            if key not in results:
                results[key] = []
            results[key].append(value)
    return results


def create_figure(x, items, fig_name):
    figure = plt.figure()
    plot = figure.add_subplot(111)
    for name, y in items.items():
        plot.plot(x, y, label=name)
    plot.legend()
    plot.set_title(fig_name)
    plot.set_xlabel("Epochs")
    return plot, figure


def visualize_distance_AP(values: list):
    result = flatten(values)
    x = np.arange(len(result["distance"]))
    distance_values = flatten(result["distance"])
    plot, figure = create_figure(x, distance_values, "Distance AP")
    plot.set_ylim(0, 1)
    plot.set_xlim(0)
    return figure

def visualize_iou_AP(values: list):
    result = flatten(values)
    x = np.arange(len(result["iou"]))
    iou_values = flatten(result["iou"])
    plot, figure = create_figure(x, iou_values, "IoU AP")
    plot.set_ylim(0, 1)
    plot.set_xlim(0)
    return figure


def visualize_evals(checkpoint: PTModel):
    by_epoch = checkpoint.evaluations
    evaluations = flatten(by_epoch) # flatten one layer
    # now visualize the distance AP and the IoU AP
    fig_val_dist_ap = visualize_distance_AP(evaluations["val_evaluation"])
    fig_test_dist_ap = visualize_distance_AP(evaluations["test_evaluation"])
    fig_val_iou_ap = visualize_iou_AP(evaluations["val_evaluation"])
    fig_test_iou_ap = visualize_iou_AP(evaluations["test_evaluation"])
    return {
        "validation_distance_AP": fig_val_dist_ap,
        "test_distance_AP": fig_test_dist_ap,
        "validation_IoU_AP": fig_val_iou_ap,
        "test_IoU_AP": fig_test_iou_ap
    }


def visualize_loss(values: list):
    result = flatten(values)
    x = np.arange(len(values))
    plot, figure = create_figure(x, result, "Losses")
    return figure


def visualize_losses(checkpoint: PTModel):
    by_epoch = checkpoint.losses
    losses = flatten(by_epoch)
    train_losses = visualize_loss(losses["train_losses"])
    val_losses = visualize_loss(losses["val_losses"])
    return {
        "training_losses": train_losses,
        "val_losses": val_losses
    }


def visualize_all(checkpoint: PTModel):
    result = visualize_evals(checkpoint)
    result.update(visualize_losses(checkpoint))
    return result


def init_parser():
    parser = ArgumentParser()
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    weights_path = Path(args.weights)
    output_path = Path(args.output)
    # load the model and save the graphs
    checkpoint = load_weights(weights_path, device=torch.device("cpu"))
    figures = visualize_all(checkpoint)
    for name, fig in figures.items():
        fig.savefig(output_path / name)