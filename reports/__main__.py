import json
import torch
from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from model.ptmodel import PTModel
from reports.reports import Report
import seaborn as sns


def load_weights(weights: Path, device):
    """
    Loads the model weigths.
    """
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    checkpoint = torch.load(weights, map_location=device, weights_only=False)
    checkpoint = PTModel.from_checkpoint(checkpoint)
    return checkpoint


def init_parser():
    parser = ArgumentParser()
    parser.add_argument("--reports", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    reports_path = Path(args.reports)
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="darkgrid")
    # load the reports and save the graphs
    reports = Report.from_json(json.load(open(reports_path)))
    reports.generate_reports(output_path)