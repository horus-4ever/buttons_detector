import torch
from pathlib import Path
from pprint import pprint
import matplotlib.pyplot as plt
from argparse import ArgumentParser

from model.ptmodel import PTModel
from reports.reports import ComparisonValidationFigureRequest, ComparisonValidationVisualizer, EvaluationVisualizer, LossVisualizer
from .requests import get_loss_reports, get_test_groups_reports, get_validation_reports


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

def create_invidual_reports(checkpoints: list[PTModel], output_path: Path):
    for checkpoint in checkpoints:
        eval_visualizer = EvaluationVisualizer.from_checkpoint(checkpoint)
        loss_visualizer = LossVisualizer.from_checkpoint(checkpoint)
        for key, request in get_loss_reports().items():
            fig_path = output_path / checkpoint.model_configuration.name / f"{key}.png"
            loss_visualizer.visualize_and_save(request, fig_path)
        for key, request in get_validation_reports().items():
            fig_path = output_path / checkpoint.model_configuration.name / f"{key}.png"
            eval_visualizer.visualize_and_save(request, fig_path)
        for key, request in get_test_groups_reports().items():
            fig_path = output_path / checkpoint.model_configuration.name / f"{key}.png"
            eval_visualizer.visualize_and_save(request, fig_path)


def create_comparison_reports(checkpoints: list[PTModel], output_path: Path):
    pass
    # comparator = ComparisonValidationVisualizer.from_checkpoints(checkpoints)
    # request = ComparisonValidationFigureRequest(
    #     curves=[
    #         "val_evaluation.distance.ALL.AP_10:AP",
    #         "val_evaluation.distance.INVISIBLE.AP_10:AP",
    #         "val_evaluation.distance.VISIBLE.AP_10:AP"
    #     ],
    #     curves_names=["ALL AP_5", "INVISIBLE AP_5", "VISIBLE AP_5"],
    #     models=["PRTR_pair", "PRTR_BEST2", "PRTR_BEST3"],
    #     fig_name="Figure"
    # )
    # comparator.visualize_and_save(request, output_path / "comparison.png")


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    weights_paths = [Path(path) for path in args.weights]
    output_path = Path(args.output)
    # load the model and save the graphs
    checkpoints = [load_weights(path, device=torch.device("cpu")) for path in weights_paths]
    # individual reports
    create_invidual_reports(checkpoints, output_path)
    # comparisons
    # create_comparison_reports(checkpoints, output_path)