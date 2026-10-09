import abc
import argparse
import json
import math
import numpy as np
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, override

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from torchvision.ops import box_convert, box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from mpl_toolkits.axes_grid1 import ImageGrid
from dataformat.dataformat import *
from dataformat.dataformat import Annotation
from dataformat.dataset import DatasetConfig
from train.transforms import ValidationTransform
from train.commons import collate_fn

from model.prtr import build_model_from
from model.utils import load_weights, compute_iou, compute_pair_iou


class PairWrapper:
    """
    A wrapper around the Pair class that adds an image ID.
    For the evaluation, we need to know which pair belongs to which image.
    """
    def __init__(self, pair: Pair, image_id: str):
        self.pair = pair
        self.image_id = image_id

    @property
    def button(self) -> Button:
        return self.pair.button

    @property
    def counterpart(self) -> Counterpart:
        return self.pair.counterpart

    @property
    def fastened(self) -> bool:
        return self.pair.fastened


class Prediction:
    BOX_FORMAT = "cxcywh"

    def __init__(self, button_box, counterpart_box, confidence, image_id):
        """
        Represents a single prediction of the model.
          button_box: cxcywh
          counterpart_box: cxcywh
        """
        self.button_box = button_box # cxcywh
        self.counterpart_box = counterpart_box # cxcywh
        self.confidence = confidence
        self.image_id = image_id

    @classmethod
    def from_outputs(cls, outputs: dict, image_id) -> list["Prediction"]:
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
        predictions = []
        for b in range(B):
            for q in range(Q):
                button_box, counterpart_box = object_boxes[q] # [RpQ, 4]
                prediction = cls(
                    button_box=button_box,
                    counterpart_box=counterpart_box,
                    confidence=confidence[b][q],
                    image_id=image_id
                )
                predictions.append(prediction)
        return predictions



@dataclass
class Metric:
    TP: int
    TN: int
    FP: int
    FN: int

    @classmethod
    def empty(cls) -> "Metric":
        return cls(0, 0, 0, 0)

    @property
    def precision(self) -> float:
        return float(self.TP) / (self.TP + self.FP)

    @property
    def recall(self) -> float:
        return float(self.TP) / (self.TP + self.FN)

    @property
    def F1(self) -> float:
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    def __add__(self, other: "Metric") -> "Metric":
        return Metric(
            self.TP + other.TP,
            self.TN + other.TN,
            self.FP + other.FP,
            self.FN + other.FN
        )


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



class Evaluation(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, predictions: list[Prediction], annotations: list[PairWrapper]) -> Any:
        pass

    def filter_confidence(self, predictions: list[Prediction], threshold: float):
        """
        Filters the predictions whose confidences are above the given threshold.
        """
        result = []
        for predicton in predictions:
            if predicton.confidence >= threshold:
                continue
            result.append(predicton)
        return result

    def sort_by_confidence(self, predictions: list[Prediction], ascending: bool = True):
        """
        Sort the given predictions by their confidence, in the given order.
        """
        return sorted(predictions, key=lambda p: p.confidence, reverse=not ascending)


class PairIoUEvaluation(Evaluation):
    def __init__(self, iou_threshold: float):
        self.iou_threshold = iou_threshold

    def evaluate(self, predictions: list[Prediction], annotations: list[PairWrapper]) -> Metric:
        # keep only the predictions whose confidence is above the threshold
        predictions = self.filter_confidence(predictions, 0.5)
        # sort by highest confidence order
        predictions = self.sort_by_confidence(predictions, ascending=False)
        matched = torch.zeros(len(annotations), dtype=torch.bool)
        for prediction in predictions:
            # get the button and counterpart bboxes (single tensor)
            pred_button_bbox = prediction.button_box # [4]
            pred_counterpart_bbox = prediction.counterpart_box # [4]
            # get the ground truth button and counterpart IoU (list of tensors)
            gt_button_bboxes = torch.stack([pair.button.bbox.to_tensor() for pair in annotations]) # [num_gt, 4]
            gt_counterpart_bboxes = torch.stack([pair.counterpart.bbox.to_tensor() for pair in annotations]) # [num_gt, 4]
            # compute the IoU for each ground truth with the predicted button and counterpart
            ious_button = box_iou(pred_button_bbox[None, :], gt_button_bboxes)[0] # [num_gt]
            ious_counterpart = box_iou(pred_counterpart_bbox[None, :], gt_counterpart_bboxes)[0] # [num_gt]
            ious_button[matched] = -1.0
            ious_counterpart[matched] = -1.0
            keep = (ious_button >= self.iou_threshold) & (ious_counterpart >= self.iou_threshold)
            if not keep.any():
                continue
            kept_together = torch.min(ious_button, ious_counterpart)
            kept_together[~keep] = -1.0
            _, best_gt_index = kept_together.max(dim=0)
            matched[best_gt_index] = True
        nb_matched = int(matched.sum().item())
        nb_predictions = len(predictions)
        nb_ground_truths = len(annotations)
        return Metric(
            TP=nb_matched,
            TN=0,
            FP=nb_predictions - nb_matched,
            FN=nb_ground_truths - nb_matched
        )


class PairDistanceEvaluation(Evaluation):
    """
    Computes the non-interpolated AP, based on Euclidean distance metric, at a given distance threshold.
    Evaluation based on the distance between predicted and ground truth bounding boxes.
    The evaluation is based on greedy matching.
    Given a set of predictions and a set of ground truth annotations:
    - sort predictions by confidence (highest to lowest)
    - for each prediction:
        - compute the distance to all unmatched ground truth annotations
        - filter the predictions whose distances are below the threshold
        - match the prediction to the closest ground truth annotation (max then argmin)
    """

    def __init__(self, distance_threshold: float):
        self.distance_threshold = distance_threshold
        self.AP = 0.0
        self.overall = Metric.empty()

    @override
    def evaluate(self, predictions: list[Prediction], annotations: list[PairWrapper]):
        # sort by highest confidence order
        predictions = self.sort_by_confidence(predictions, ascending=False)
        matched = torch.zeros(len(annotations), dtype=torch.bool)
        nb_ground_truths = len(annotations)
        # compute the AP based on greedy matching
        AP = 0.0
        previous_recall = 0.0
        for n, prediction in enumerate(predictions):
            # get the annotations corresponding to the image of the prediction
            image_ann_indices = torch.tensor([index for index, pair in enumerate(annotations) if pair.image_id == prediction.image_id])
            pred_annotations = [pair for pair in annotations if pair.image_id == prediction.image_id]
            if not pred_annotations: # no ground truth for this image, we skip it
                continue
            # get the button and counterpart centers (single tensor)
            pred_button_center = prediction.button_box[:2] # [2]
            pred_counterpart_center = prediction.counterpart_box[:2] # [2]
            # get the ground truth button and counterpart centers (list of tensors)
            gt_button_centers = torch.stack([pair.button.bbox.to_tensor()[:2] for pair in pred_annotations]) # [num_gt, 2]
            gt_counterpart_centers = torch.stack([pair.counterpart.bbox.to_tensor()[:2] for pair in pred_annotations]) # [num_gt, 2]
            # compute for each ground truth the distance to the predicted button and counterpart
            distance_button = torch.linalg.vector_norm(
                pred_button_center.unsqueeze(0) - gt_button_centers,
                dim=1
            ) # [num_gt]
            distance_fastener = torch.linalg.vector_norm(
                pred_counterpart_center.unsqueeze(0) - gt_counterpart_centers,
                dim=1
            ) # [num_gt]
            # already matched ground truths should be ignored (set to infinity)
            distance_button[matched[image_ann_indices]] = math.inf
            distance_fastener[matched[image_ann_indices]] = math.inf
            # match only if both distances for the button and the counterpart are below the threshold
            keep = (distance_button <= self.distance_threshold) & (distance_fastener <= self.distance_threshold)
            if not keep.any():
                continue
            kept_together = torch.max(distance_button, distance_fastener)
            kept_together[~keep] = math.inf
            _, best_gt_index = kept_together.min(dim=0)
            matched[image_ann_indices[best_gt_index]] = True
            # now we compute the running AP
            nb_matched = int(matched.sum().item())
            nb_predictions = n + 1
            recall = nb_matched / nb_ground_truths if nb_ground_truths > 0 else 0.0
            precision = nb_matched / nb_predictions if nb_predictions > 0 else 0.0
            AP += (recall - previous_recall) * precision
            previous_recall = recall
        nb_matched = int(matched.sum().item())
        nb_predictions = len(predictions)
        self.overall = Metric(
            TP=nb_matched,
            TN=0,
            FP=nb_predictions - nb_matched,
            FN=nb_ground_truths - nb_matched
        )
        self.AP = AP


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

    def evaluate(self, threshold: float):
        val_dataset = dataset.validation_annotations
        mAP_buttons = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox", class_metrics=True)
        mAP_counterparts = MeanAveragePrecision(box_format="cxcywh", iou_type="bbox", class_metrics=True)
        pair_distance_evaluator = PairDistanceEvaluation(distance_threshold=5.0)
        pair_iou_evaluator = PairIoUEvaluation(iou_threshold=0.5)
        # collect all pairs in a list
        all_pairs = []
        all_predictions = []
        for i, annotation in enumerate(val_dataset):
            print(f"Image [{i} / {len(val_dataset)}]", end="\r")
            outputs = self.run_one(annotation)
            all_pairs.extend([PairWrapper(pair, image_id=annotation.image.url) for pair in annotation.cloth.pairs])
            predictions = Prediction.from_outputs(outputs, image_id=annotation.image.url)
            all_predictions.extend(predictions)
        pair_distance_mAP = pair_distance_evaluator.evaluate(all_predictions, all_pairs)
        pair_iou_mAP = pair_iou_evaluator.evaluate(all_predictions, all_pairs)
        mAP_buttons_result = mAP_buttons.compute()
        mAP_counterparts_result = mAP_counterparts.compute()
        return (
            mAP_buttons_result,
            mAP_counterparts_result,
            pair_distance_mAP,
            pair_iou_mAP
        )

        

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
    mAP_buttons, mAP_fasteners, visibility_recall, occlusion_recall, total_ground_truths, average_distance_error, average_normalized_distance_error, pair_precision, pair_recall, pair_distance_precision, pair_distance_recall = evaluator.evaluate(threshold=0.5)

    print("~ mAP_buttons:", mAP_buttons)
    print("~ mAP_fasteners: ", mAP_fasteners)
    print("~ Visibility Recall: ", visibility_recall)
    print("~ Occlusion Recall: ", occlusion_recall)
    print("~ Average Distance Error: ", average_distance_error)
    print("~ Average Normalized Distance Error: ", average_normalized_distance_error)
    print("~ Pair Precision@50: ", pair_precision)
    print("~ Pair Recall@50: ", pair_recall)
    print("~ Pair Distance Precision@5: ", pair_distance_precision)
    print("~ Pair Distance Recall@5: ", pair_distance_recall)