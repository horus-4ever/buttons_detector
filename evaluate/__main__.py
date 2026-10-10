from .evaluate import Evaluator
from model.prtr import build_model_from
from model.utils import load_weights
from dataformat.dataset import DatasetConfig

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
    # now evaluate a dataset based on the dataset cache it has
    evaluator = Evaluator(model, dataset, threshold, device)
    pair_distance_mAP, pair_iou_mAP, pair_distance_mAP_production, pair_iou_mAP_production = evaluator.evaluate()

    print("~~~~ Pair Distance AP ~~~~")
    pprint.pprint(pair_distance_mAP.results())
    print("~~~~ Pair IoU AP ~~~~")
    pprint.pprint(pair_iou_mAP.results())
    print(f"~~~~ Pair Distance AP (>={threshold}) ~~~~")
    pprint.pprint(pair_distance_mAP_production.results())
    print(f"~~~~ Pair IoU AP (>={threshold}) ~~~~")
    pprint.pprint(pair_iou_mAP_production.results())
