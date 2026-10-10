import abc
import math
from dataclasses import dataclass, field
from typing import Any, override

import torch
from PIL import Image
from torchvision.ops import box_convert, box_iou
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from dataformat.dataformat import *
from dataformat.dataformat import Annotation
from dataformat.dataset import DatasetConfig
from train.transforms import ValidationTransform
from train.commons import collate_fn


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
        pred_boxes = outputs["pred_boxes"].detach().cpu() # [B, Q, RpQ, 4]
        probs = outputs["pred_logits"].softmax(-1).detach().cpu() # [B, Q, num_classes+1]
        assert pred_boxes.shape[0] == 1, "Only batch size 1 is supported"
        # get the boxes with the right classes only
        confidence = probs[0, ..., 0] # [Q]
        object_boxes = pred_boxes[0] # [Q, RpQ, 4]
        # for each prediction of the batch
        predictions = []
        for conf, pred in zip(confidence, object_boxes):
            button_box, counterpart_box = pred # [RpQ, 4]
            prediction = cls(
                button_box=button_box,
                counterpart_box=counterpart_box,
                confidence=conf.item(),
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
        self.AP = 0.0
        self.overall = Metric.empty()

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
            # get the button and counterpart bboxes (single tensor)
            pred_button_bbox = prediction.button_box # [4]
            pred_counterpart_bbox = prediction.counterpart_box # [4]
            # get the ground truth button and counterpart IoU (list of tensors)
            gt_button_bboxes = torch.stack([pair.button.bbox.to_tensor() for pair in pred_annotations]) # [num_gt, 4]
            gt_counterpart_bboxes = torch.stack([pair.counterpart.bbox.to_tensor() for pair in pred_annotations]) # [num_gt, 4]
            # compute the IoU for each ground truth with the predicted button and counterpart
            ious_button = box_iou(pred_button_bbox[None, :], gt_button_bboxes, fmt=prediction.BOX_FORMAT)[0] # [num_gt]
            ious_counterpart = box_iou(pred_counterpart_bbox[None, :], gt_counterpart_bboxes, fmt=prediction.BOX_FORMAT)[0] # [num_gt]
            ious_button[matched[image_ann_indices]] = -1.0
            ious_counterpart[matched[image_ann_indices]] = -1.0
            keep = (ious_button >= self.iou_threshold) & (ious_counterpart >= self.iou_threshold)
            if not keep.any():
                continue
            kept_together = torch.min(ious_button, ious_counterpart)
            kept_together[~keep] = -1.0
            _, best_gt_index = kept_together.max(dim=0)
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
        return self


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

    def __init__(self, distance_threshold: float, image_size: int = 512):
        self.distance_threshold = distance_threshold
        self.image_size = image_size
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
            distance_counterpart = torch.linalg.vector_norm(
                pred_counterpart_center.unsqueeze(0) - gt_counterpart_centers,
                dim=1
            ) # [num_gt]
            # already matched ground truths should be ignored (set to infinity)
            distance_button[matched[image_ann_indices]] = math.inf
            distance_counterpart[matched[image_ann_indices]] = math.inf
            # match only if both distances for the button and the counterpart are below the threshold
            keep = (distance_button <= self.distance_threshold / self.image_size) & (distance_counterpart <= self.distance_threshold / self.image_size)
            if not keep.any():
                continue
            kept_together = torch.max(distance_button, distance_counterpart)
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
        return self


class PairIoUAllEvaluation(Evaluation):
    def __init__(self):
        self.AP_50 = 0.0
        self.AP_75 = 0.0
        self.AP_90 = 0.0
        self.AP = 0.0

    def evaluate(self, predictions: list[Prediction], annotations: list[PairWrapper], confidence_threshold: float = 0.0):
        predictions = self.filter_confidence(predictions, confidence_threshold)
        ap_50_evaluator = PairIoUEvaluation(iou_threshold=0.5)
        ap_75_evaluator = PairIoUEvaluation(iou_threshold=0.75)
        ap_90_evaluator = PairIoUEvaluation(iou_threshold=0.9)
        self.AP_50 = ap_50_evaluator.evaluate(predictions, annotations).AP
        self.AP_75 = ap_75_evaluator.evaluate(predictions, annotations).AP
        self.AP_90 = ap_90_evaluator.evaluate(predictions, annotations).AP
        self.AP = (self.AP_50 + self.AP_75 + self.AP_90) / 3
        return self

    def results(self):
        return {
            "AP_50": self.AP_50,
            "AP_75": self.AP_75,
            "AP_90": self.AP_90,
            "AP": self.AP
        }



class PairDistanceAllEvaluation(Evaluation):
    def __init__(self):
        self.AP_1 = 0.0
        self.AP_3 = 0.0
        self.AP_5 = 0.0
        self.AP_10 = 0.0

    def evaluate(self, predictions: list[Prediction], annotations: list[PairWrapper], confidence_threshold: float = 0.0):
        predictions = self.filter_confidence(predictions, confidence_threshold)
        ap_1_evaluator = PairDistanceEvaluation(distance_threshold=1.0)
        ap_3_evaluator = PairDistanceEvaluation(distance_threshold=3.0)
        ap_5_evaluator = PairDistanceEvaluation(distance_threshold=5.0)
        ap_10_evaluator = PairDistanceEvaluation(distance_threshold=10.0)
        self.AP_1 = ap_1_evaluator.evaluate(predictions, annotations).AP
        self.AP_3 = ap_3_evaluator.evaluate(predictions, annotations).AP
        self.AP_5 = ap_5_evaluator.evaluate(predictions, annotations).AP
        self.AP_10 = ap_10_evaluator.evaluate(predictions, annotations).AP
        self.AP = (self.AP_1 + self.AP_3 + self.AP_5 + self.AP_10) / 4
        return self

    def results(self):
        return {
            "AP": self.AP,
            "AP_1": self.AP_1,
            "AP_3": self.AP_3,
            "AP_5": self.AP_5,
            "AP_10": self.AP_10
        }


class Evaluator:
    def __init__(self, model, dataset: DatasetConfig, threshold: float, device):
        """
        Evaluator class that evaluates a model on a given dataset.
        The threshold defined is the confidence threshold for the predictions to be considered in the evaluation.
        This threshold is meant to reflect actual production usage.
        """
        self.model = model
        self.dataset = dataset
        self.threshold = threshold
        self.device = device

    def run_one(self, annotation: Annotation):
        image_root = self.dataset.dataset_root / "images"
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

    def evaluate(self):
        val_dataset = self.dataset.validation_annotations
        # collect all pairs in a list
        all_pairs = []
        all_predictions = []
        for i, annotation in enumerate(val_dataset):
            print(f"Image [{i} / {len(val_dataset)}]", end="\r")
            outputs = self.run_one(annotation)
            all_pairs.extend([PairWrapper(pair, image_id=annotation.image.url) for pair in annotation.cloth.pairs])
            predictions = Prediction.from_outputs(outputs, image_id=annotation.image.url)
            all_predictions.extend(predictions)
        pair_distance_mAP = PairDistanceAllEvaluation().evaluate(all_predictions, all_pairs)
        pair_iou_mAP = PairIoUAllEvaluation().evaluate(all_predictions, all_pairs)
        pair_distance_mAP_production = PairDistanceAllEvaluation().evaluate(all_predictions, all_pairs, confidence_threshold=self.threshold)
        pair_iou_mAP_production = PairIoUAllEvaluation().evaluate(all_predictions, all_pairs, confidence_threshold=self.threshold)
        return (
            pair_distance_mAP,
            pair_iou_mAP,
            pair_distance_mAP_production,
            pair_iou_mAP_production
        )
