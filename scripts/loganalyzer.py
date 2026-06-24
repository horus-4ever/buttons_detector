import argparse
import matplotlib.pyplot as plt
import re
import numpy as np
from pathlib import Path


log_regex = re.compile(
    r"Epoch\s*\[(?P<epoch>\d+)/\d+\]\s*\|\s*"
    r"train\s*\(\s*loss=\s*(?P<train_loss>\d+(?:\.\d+)?)"
    r".*?\|\s*"
    r"val\s*\(loss\s*=\s*(?P<val_loss>\d+(?:\.\d+)?)"
)


def parse_log_file(file: Path):
    with open(file, "r") as document:
        data = document.read()
        return parse_log_data(data)


def parse_log_data(log_data):
    log_data = log_data.split("\n")
    losses = []
    for line in log_data:
        result = log_regex.search(line)
        if not result:
            continue
        epoch = int(result.group("epoch"))
        train_loss = float(result.group("train_loss"))
        validation_loss = float(result.group("val_loss"))
        losses.append((train_loss, validation_loss))
    losses = np.stack(losses)
    return losses

def plot_losses(losses):
    n_epochs = len(losses)
    x_axis = np.arange(1, n_epochs + 1)
    train_losses = losses[:, 0]
    val_losses = losses[:, 1]
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
    parsed_logs = parse_log_file(log_file)
    plot_losses(parsed_logs)
