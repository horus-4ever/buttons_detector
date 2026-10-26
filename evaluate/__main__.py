from .evaluate import Evaluator, PairIoUAllEvaluation, PairDistanceAllEvaluation, PairIoUEvaluation, PairDistanceEvaluation
from model.prtr import build_model_from
from model.utils import load_weights
from dataformat.dataset import DatasetConfig
from train.commons import collate_fn
from torch.utils.data import DataLoader
from train.transforms import ValidationTransform

import pprint
import argparse
from pathlib import Path
import torch


def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, required=False, default="cuda")
    parser.add_argument("--threshold", type=float, required=False, default=0.9)
    return parser


def get_evaluator(threshold: float = 0.9) -> Evaluator:
    """
    Returns an evaluator with the given threshold.
    """
    evaluator = Evaluator()
    evaluator.add_evaluation("distance", PairDistanceAllEvaluation())
    evaluator.add_evaluation("iou", PairIoUAllEvaluation())
    evaluator.add_evaluation("distance_threshold", PairDistanceAllEvaluation(confidence_threshold=threshold))
    evaluator.add_evaluation("iou_threshold", PairIoUAllEvaluation(confidence_threshold=threshold))
    return evaluator


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    # get the arguments
    model_path = Path(args.model)
    weights_path = Path(args.weights)
    dataset_path = Path(args.dataset)
    device_name = args.device
    threshold = args.threshold
    # get the device
    device = torch.device(device_name)
    print("Using device:", device)
    # load the model and its weights
    model = build_model_from(model_path)
    model.to(device=device)
    load_weights(model, weights=weights_path, device=device)
    # get the dataset
    dataset = DatasetConfig.open(config_path=dataset_path).load()
    train, val, test = dataset.to_torch_dataset()
    train.transform = ValidationTransform(512) # type: ignore
    val.transform = ValidationTransform(512) # type: ignore
    test.transform = ValidationTransform(512) # type: ignore
    # now evaluate a dataset based on the dataset cache it has
    evaluator = get_evaluator(threshold=threshold)
    train_loader = DataLoader(
        train,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda")
    )
    test_loader = DataLoader(
        test,
        batch_size=1,
        shuffle=False,
        num_workers=8,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda")
    )
    for loader in (train_loader, val_loader, test_loader):
        outputs = []
        for images, padding_mask, annotations, (W, H) in loader:
            annotations = annotations[0] # batch size is 1 for evaluation
            images = images.to(device, non_blocking=True)
            padding_mask = padding_mask.to(device, non_blocking=True)
            output = model(images, padding_mask)
            outputs.append((annotations, output))
        evaluations = evaluator.evaluate(outputs)
        print("#### EVALUATION ####")
        print("~~~~ Pair Distance AP ~~~~")
        pprint.pprint(evaluations["distance"])
        print("~~~~ Pair IoU AP ~~~~")
        pprint.pprint(evaluations["iou"])
        print(f"~~~~ Pair Distance AP (>={threshold}) ~~~~")
        pprint.pprint(evaluations["distance_threshold"])
        print(f"~~~~ Pair IoU AP (>={threshold}) ~~~~")
        pprint.pprint(evaluations["iou_threshold"])
        print()
