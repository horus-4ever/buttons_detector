import argparse
import json
import math
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from mpl_toolkits.axes_grid1 import ImageGrid
from dataformat.dataformat import *
from dataformat.dataset import DatasetConfig
from train.transforms import ValidationTransform
from train.commons import collate_fn

from model.prtr import build_model_from
from model.utils import load_weights, compute_iou, compute_pair_iou


class Prediction:
    def __init__(self, button_box, fastener_box, confidence, annotations: Annotation):
        self.button_box = button_box
        self.fastener_box = fastener_box
        self.confidence = confidence
        self.annotations = annotations

    @classmethod
    def from_outputs(cls, outputs: dict, annotations: Annotation) -> "PredictionList":
        pred_boxes = outputs["pred_boxes"].detach() # [B, Q, RpQ, 4]
        probs = outputs["pred_logits"].softmax(-1).detach() # [B, Q, num_classes+1]
        predicted_classes = probs.argmax(dim=-1)  # [B, Q]
        B, *_ = pred_boxes.size()
        # get the boxes with the right classes only
        scores = probs[..., 0] # [B, Q]
        object_boxes_indices = (predicted_classes == 0) # [B, Q]
        no_object_indices = (predicted_classes == 1)
        confidence = scores
        object_boxes = pred_boxes[object_boxes_indices] # [Q, RpQ, 4]
        # for each prediction of the batch
        Q, *_ = object_boxes.size()
        predictions = PredictionList()
        for b in range(B):
            for q in range(Q):
                button_box, fastener_box = object_boxes[q] # [RpQ, 4]
                prediction = cls(
                    button_box=button_box,
                    fastener_box=fastener_box,
                    confidence=confidence[b][q],
                    annotations=annotations
                )
                predictions.add(prediction)
        return predictions


class PredictionList:
    def __init__(self, predictions: list[Prediction] | None = None):
        self.predictions = predictions or []

    def add(self, *predictions):
        self.predictions.extend(predictions)

    def merge(self, predictions: "PredictionList"):
        self.predictions.extend(predictions.predictions)

    def sort_by_confidence(self):
        """
        Sort by confidence. A new list is returned (no in-place sort).
        """
        return PredictionList(sorted(self.predictions, key = lambda p: p.confidence))

    def __len__(self):
        return len(self.predictions)

    def __getitem__(self, key):
        return self.predictions[key]


@dataclass
class Metric:
    TP: int
    TN: int
    FP: int
    FN: int

    @property
    def precision(self) -> float:
        return float(self.TP) / (self.TP + self.FP)

    @property
    def recall(self) -> float:
        return float(self.TP) / (self.TP + self.FN)

    @property
    def F1(self) -> float:
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)


@dataclass
class MetricCollection:
    data: list[Metric] = field(default_factory=list)

    def add(self, *args):
        self.data.extend(*args)

    @property
    def average_precision(self) -> float:
        return sum(m.precision for m in self.data) / len(self.data)

    @property
    def average_recall(self) -> float:
        return sum(m.recall for m in self.data) / len(self.data)

    @property
    def average_F1(self) -> float:
        return sum(m.F1 for m in self.data) / len(self.data)


class Evaluator:
    def __init__(self, model, dataset: DatasetConfig, threshold: float, device):
        self.model = model
        self.dataset = dataset
        self.threshold = threshold
        self.device = device

    def run_one(self, annotation: Annotation):
        image_root = self.dataset.images_dir
        image = Image.open(image_root / annotation.image.url)
        transform = ValidationTransform(512)
        image, annotation = transform(image, annotation)
        images, masks, annotation, _ = collate_fn([(image, annotation)]) # type: ignore
        images = images.to(self.device, non_blocking=True)
        masks = masks.to(self.device, non_blocking=True)
        # now the inference
        self.model.eval()
        outputs = self.model(images, masks)
        return outputs

    def evaluate(self, threshold: float):
        val_dataset = dataset.validation_annotations
        mAP_buttons = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox", class_metrics=True)
        for i, annotation in enumerate(val_dataset):
            print(f"Image [{i} / {len(val_dataset)}]", end="\r")
            outputs = self.run_one(annotation)
            pred_boxes = outputs["pred_boxes"].detach().cpu()[0] # [Q, RpQ, 4]
            probs = outputs["pred_logits"].softmax(-1).detach().cpu()[0] # [Q, num_classes+1]
            predicted_classes = probs.argmax(dim=-1)  # [Q]
            Q, *_ = pred_boxes.size()
            mAP_buttons.update(
                [{
                    "boxes": pred_boxes[predicted_classes == 0][:, 0, :], # only the pair boxes
                    "scores": probs[predicted_classes == 0][:, 0],
                    "labels": predicted_classes[predicted_classes == 0]
                }],
                [{
                    "boxes": torch.stack([pair.button.bbox.to_tensor() for pair in annotation.cloth.pairs]),
                    "labels": torch.tensor([0] * len(annotation.cloth.pairs), dtype=torch.int64)
                }]
            )
        print("Inference finished.")
        total_ground_truths = sum(len(annotation.cloth.pairs) for annotation in val_dataset)
        return mAP_buttons.compute(), total_ground_truths

        

def init_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--device", type=str, required=False, default="cuda")
    parser.add_argument("--threshold", type=float, required=False, default=0.5)
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
    result = evaluator.evaluate(threshold=0.5)

    print("mAP:", result[0]) ; exit(0)
