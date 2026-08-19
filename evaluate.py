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
from torchvision.ops import box_convert, box_iou
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

    def fastener_recall(self, pred_boxes, pred_scores, gt_boxes, gt_visible, iou_threshold=0.5):
        """
        pred_scores: [Q]
        """
        keep = pred_scores >= self.threshold
        pred_boxes = pred_boxes[keep]
        pred_scores = pred_scores[keep]
        # get the number of ground truth
        n_gt = gt_boxes.shape[0]
        visible_total = int(gt_visible.sum().item())
        occluded_total = int((~gt_visible).sum().item())
        # compute the IoU
        pred_xyxy = box_convert(
            pred_boxes,
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )
        gt_xyxy = box_convert(
            gt_boxes,
            in_fmt="cxcywh",
            out_fmt="xyxy",
        )
        ious = box_iou(pred_xyxy, gt_xyxy) # [num_pred, num_gt]
        order = pred_scores.argsort(descending=True) # [num_pred]
        # define a tensor for the matched one, so that we don't reuse them
        gt_matched = torch.zeros(n_gt, dtype=torch.bool)
        # we loop for each predictions to match to a ground truth
        for i in order: # we match in order of confidence, from the highest confidence
            pred_ious = ious[i].clone() # [num_gt]
            pred_ious[gt_matched] = -1.0
            best_iou, best_gt_index = pred_ious.max(dim=0) # take the best iou and its index
            if best_iou >= iou_threshold:
                gt_matched[best_gt_index] = True
        # now we want the visibility
        visible_tp = int((gt_matched & gt_visible).sum().item())
        occluded_tp = int((gt_matched & (~gt_visible)).sum().item())
        return visible_tp, occluded_tp, visible_total, occluded_total

    def fastener_distance_error(self, pred_boxes, pred_scores, gt_boxes):
        """
        pred_boxes: [M, 4], we assume we pass only the fastener boxes
        pred_scores: [M]
        gt_boxes: [N, 4]
        """
        order = pred_scores.argsort(descending=True)
        gt_matched = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
        total_distance_error = 0.0
        for i in order:
            pred_box = pred_boxes[i]
            pred_center = pred_box[:2] # [cx, cy]
            # compute the distance to all ground truth boxes
            gt_centers = gt_boxes[:, :2] # [N, 2]
            distances = torch.norm(gt_centers - pred_center, dim=1) # [N]
            distances[gt_matched] = float("inf") # we don't want to match to already matched ground truths
            best_distance, best_gt_index = distances.min(dim=0)
            if best_distance < float("inf"):
                gt_matched[best_gt_index] = True
                total_distance_error += best_distance.item()
        nb_matched = int(gt_matched.sum().item())
        return total_distance_error, nb_matched

    def evaluate(self, threshold: float):
        val_dataset = dataset.validation_annotations
        mAP_buttons = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox", class_metrics=True)
        mAP_fasteners = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox", class_metrics=True)
        # and for the recall of visible and occluded fasteners
        visible_tp = 0
        visible_total = 0
        occluded_tp = 0
        occluded_total = 0
        total_distance_error = 0.0
        total_matched = 0
        for i, annotation in enumerate(val_dataset):
            print(f"Image [{i} / {len(val_dataset)}]", end="\r")
            outputs = self.run_one(annotation)
            pred_boxes = outputs["pred_boxes"].detach().cpu()[0] # [Q, RpQ, 4]
            probs = outputs["pred_logits"].softmax(-1).detach().cpu()[0] # [Q, num_classes+1]
            predicted_classes = probs.argmax(dim=-1)  # [Q]
            Q, *_ = pred_boxes.size()
            # some useful variables
            nb_pairs = len(annotation.cloth.pairs)
            nb_visible_fasteners = sum(1 for pair in annotation.cloth.pairs if pair.fastener.visible)
            nb_invisible_fasteners = nb_pairs - nb_visible_fasteners
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
            mAP_fasteners.update(
                [{
                    "boxes": pred_boxes[predicted_classes == 0][:, 1, :], # only the pair boxes
                    "scores": probs[predicted_classes == 0][:, 0],
                    "labels": predicted_classes[predicted_classes == 0]
                }],
                [{
                    "boxes": torch.stack([pair.fastener.bbox.to_tensor() for pair in annotation.cloth.pairs]),
                    "labels": torch.tensor([0] * len(annotation.cloth.pairs), dtype=torch.int64)
                }]
            )
            # now we compute the recall for visible and occluded fasteners
            new_visible_tp, new_occluded_tp, new_visible_total, new_occluded_total = self.fastener_recall(
                pred_boxes=pred_boxes[predicted_classes == 0][:, 1, :], # only the pair boxes
                pred_scores=probs[predicted_classes == 0][:, 0],
                gt_boxes=torch.stack([pair.fastener.bbox.to_tensor() for pair in annotation.cloth.pairs]),
                gt_visible=torch.tensor([pair.fastener.visible for pair in annotation.cloth.pairs], dtype=torch.bool),
                iou_threshold=0.5
            )
            visible_tp += new_visible_tp
            occluded_tp += new_occluded_tp
            visible_total += new_visible_total
            occluded_total += new_occluded_total
            # now we compute the distance error for the fasteners
            distance_error, nb_matched = self.fastener_distance_error(
                pred_boxes=pred_boxes[predicted_classes == 0][:, 1, :], # only the pair boxes
                pred_scores=probs[predicted_classes == 0][:, 0],
                gt_boxes=torch.stack([pair.fastener.bbox.to_tensor() for pair in annotation.cloth.pairs])
            )
            image_size = math.sqrt(annotation.image.width ** 2 + annotation.image.height ** 2)
            total_distance_error += distance_error / image_size # normalize by the diagonal of the image
            total_matched += nb_matched
        print("Inference finished.")
        total_ground_truths = sum(len(annotation.cloth.pairs) for annotation in val_dataset)
        visibility_recall = visible_tp / visible_total if visible_total > 0 else 0.0
        occlusion_recall = occluded_tp / occluded_total if occluded_total > 0 else 0.0
        average_distance_error = total_distance_error / total_matched if total_matched > 0 else 0.0
        return mAP_buttons.compute(), mAP_fasteners.compute(), visibility_recall, occlusion_recall, total_ground_truths, average_distance_error

        

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
    mAP_buttons, mAP_fasteners, visibility_recall, occlusion_recall, total_ground_truths, average_distance_error = evaluator.evaluate(threshold=0.5)

    print("~ mAP_buttons:", mAP_buttons)
    print("~ mAP_fasteners: ", mAP_fasteners)
    print("~ Visibility Recall: ", visibility_recall)
    print("~ Occlusion Recall: ", occlusion_recall)
    print("~ Average Distance Error: ", average_distance_error)