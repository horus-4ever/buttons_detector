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
        pred_boxes = outputs["pred_boxes"] # [B, Q, RpQ, 4]
        probs = outputs["pred_logits"].softmax(-1) # [B, Q, num_classes+1]
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
                button_box, fastener_box = object_boxes[q]
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
    def __init__(self, model, dataset: DatasetConfig, threshold: float):
        self.model = model
        self.dataset = dataset
        self.threshold = threshold

    def run_one(self, annotation: Annotation):
        image_root = self.dataset.images_dir
        image = Image.open(image_root / annotation.image.url)
        transform = ValidationTransform(512)
        image, annotation = transform(image, annotation)
        image, masks, annotation, _ = collate_fn([(image, annotation)]) # type: ignore
        # now the inference
        self.model.eval()
        outputs = self.model(image, masks)
        return outputs

    def evaluate(self, threshold: float):
        val_dataset = dataset.validation_annotations
        predictions = PredictionList()
        for annotation in val_dataset:
            outputs = self.run_one(annotation)
            prediction = Prediction.from_outputs(outputs, annotations=annotation)
            predictions.merge(prediction)
        total_ground_truths = sum(len(annotation.cloth.pairs) for annotation in val_dataset)
        # sort by confidence
        predictions: PredictionList = predictions.sort_by_confidence()
        tp_flags = []
        fp_flags = []
        explored_ground_truth = {}
        for prediction in predictions:
            image_name = prediction.annotations.image.url
            pairs = prediction.annotations.cloth.pairs
            if image_name not in explored_ground_truth:
                explored_ground_truth[image_name] = [False] * len(prediction.annotations.cloth.pairs)
            # now we get the index of the best non explored ground truth
            max_index = -1
            last_max = float("-inf")
            for i, pair in enumerate(pairs):
                if explored_ground_truth[image_name][i]:
                    continue
                pred_button = pair.button
                pred_fastener = pair.fastener
                iou = compute_pair_iou(
                    prediction.button_box,
                    torch.tensor(pred_button.bbox.to_cxcywh()),
                    prediction.fastener_box,
                    torch.tensor(pred_fastener.bbox.to_cxcywh())
                )
                if iou > last_max:
                    last_max = iou
                    max_index = i
            if max_index >= 0 and last_max >= threshold: # then ok, this is a true positive
                tp_flags.append(1)
                fp_flags.append(0)
                explored_ground_truth[image_name][max_index] = True
            else: # else no it is a false positive
                tp_flags.append(0)
                fp_flags.append(1)
        cumulative_tp = np.cumsum(tp_flags)
        cumulative_fp = np.cumsum(fp_flags)
        precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1)
        recall = cumulative_tp / total_ground_truths

        return {
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "tp_flags": np.asarray(tp_flags),
            "fp_flags": np.asarray(fp_flags),
            "total_ground_truths": total_ground_truths,
        }

        

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
    evaluator = Evaluator(model, dataset, threshold)
    result = evaluator.evaluate(threshold=50)

    import matplotlib.pyplot as plt
    plt.plot(result["precision"], result["recall"])
    plt.show()
