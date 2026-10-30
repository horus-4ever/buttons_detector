import torch
from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from model.ptmodel import PTModel
from reports.reports import EvaluationVisualizer, LossVisualizer
from .requests import get_loss_reports, get_validation_reports


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
    parser.add_argument("--weights", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    weights_paths = [Path(path) for path in args.weights]
    output_path = Path(args.output)
    # load the model and save the graphs
    checkpoints = [load_weights(path, device=torch.device("cpu")) for path in weights_paths]
    for checkpoint in checkpoints:
        eval_visualizer = EvaluationVisualizer.from_checkpoint(checkpoint)
        loss_visualizer = LossVisualizer.from_checkpoint(checkpoint)
        figures = {}
        for key, request in get_loss_reports().items():
            fig_path = output_path / checkpoint.model_configuration.name / f"{key}.png"
            loss_visualizer.visualize_and_save(request, fig_path)
        for key, request in get_validation_reports().items():
            fig_path = output_path / checkpoint.model_configuration.name / f"{key}.png"
            eval_visualizer.visualize_and_save(request, fig_path)