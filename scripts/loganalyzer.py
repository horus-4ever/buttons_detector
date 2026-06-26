import argparse
import matplotlib.pyplot as plt
import re
import numpy as np
from pathlib import Path


log_regex = re.compile(
    r"Epoch\s+\[(?P<epoch>\d+)/\d+\]\s+\|\s+"
    r"train loss:\s+(?P<train_loss>\d+(?:\.\d+)?)"
    r".*?\|\s+val loss:\s+(?P<val_loss>\d+(?:\.\d+)?)"
)

num = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"

log_regex_2 = re.compile(
    rf"Epoch\s+\[(?P<epoch>\d+)/(?P<total_epochs>\d+)\]\s*\|\s*"
    rf"train\s*\(\s*"
    rf"loss\s*=\s*(?P<train_loss>{num})\s*,\s*"
    rf"ce\s*=\s*(?P<train_ce>{num})\s*,\s*"
    rf"btn\s*=\s*(?P<train_btn>{num})\s*"
    rf"\)\s*\|\s*"
    rf"val\s*\(\s*"
    rf"loss\s*=\s*(?P<val_loss>{num})\s*,\s*"
    rf"ce\s*=\s*(?P<val_ce>{num})\s*,\s*"
    rf"btn\s*=\s*(?P<val_btn>{num})\s*"
    rf"\)"
)

REGEXES = [log_regex, log_regex_2]


def parse_log_file(file: Path):
    with open(file, "r") as document:
        data = document.read()
        return parse_log_data(data)


def parse_log_data(log_data):
    log_data = log_data.split("\n")
    losses = {
        "train_losses": {
            "loss": [],
            "class_loss": [],
            "coords_loss": []
        },
        "val_losses": {
            "loss": [],
            "class_loss": [],
            "coords_loss": []
        }
    }
    train_losses = losses["train_losses"]
    val_losses = losses["val_losses"]
    epochs = 0
    for line in log_data:
        results = [regex.search(line) for regex in REGEXES]
        result = None
        for match in results:
            if match:
                result = match
        if result is None:
            continue
        epochs += 1
        train_loss = float(result.group("train_loss"))
        train_class_loss = float(result.group("train_ce"))
        train_coords_loss = float(result.group("train_btn"))
        validation_loss = float(result.group("val_loss"))
        validation_class_loss = float(result.group("val_ce"))
        validation_coords_loss = float(result.group("val_btn"))
        train_losses["loss"].append(train_loss)
        train_losses["class_loss"].append(train_class_loss)
        train_losses["coords_loss"].append(train_coords_loss)
        val_losses["loss"].append(validation_loss)
        val_losses["class_loss"].append(validation_class_loss)
        val_losses["coords_loss"].append(validation_coords_loss)
    return losses, epochs

def plot_losses(losses, n_epochs):
    x_axis = np.arange(1, n_epochs + 1)
    train_losses = losses["train_losses"]["loss"]
    val_losses = losses["val_losses"]["loss"]
    plt.figure(figsize=(8, 5))
    plt.plot(x_axis, train_losses, marker="o", label="Train loss")
    plt.plot(x_axis, val_losses, marker="o", label="Validation loss")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def init_parser():
    parser = argparse.ArgumentParser("Visualize the training log.")
    parser.add_argument("--log-file", required=True, help="Path to the log file.")
    return parser

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    log_file = Path(args.log_file)
    # now get and visualize the data
    parsed_logs, n_epochs = parse_log_file(log_file)
    plot_losses(parsed_logs, n_epochs)
