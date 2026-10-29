import abc
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, override

import torch
from torchvision.ops import box_iou
from dataformat.dataformat import *
from dataformat.dataformat import Annotation


class EvaluationGroup:
    """
    Define an evaluation group.
    An evaluation group is defined by a boolean function that determine if the
    given annotation is part of the group.
    """
    def __init__(self, name: str, func: Callable):
        self.name = name
        self.func = func

    def is_applicable(self, pair: Pair):
        return self.func(pair)

    @classmethod
    def ALL(cls) -> "EvaluationGroup":
        return cls("ALL", lambda _: True)

    @classmethod
    def VISIBLE_GROUP(cls) -> "EvaluationGroup":
        return cls("VISIBLE", lambda pair: pair.counterpart.visible and not pair.fastened)

    @classmethod
    def INVISIBLE_GROUP(cls) -> "EvaluationGroup":
        return cls("INVISIBLE", lambda pair: not pair.counterpart.visible and not pair.fastened)


class RunningEvaluation:
    def __init__(self, group, nb_ground_truths):
        self.group = group
        self.nb_ground_truths = nb_ground_truths
        self.nb_matched = 0
        self.nb_predictions = 0
        self.previous_recall = 0.0
        self.AP = 0.0

    @property
    def recall(self):
        return self.nb_matched / self.nb_ground_truths if self.nb_ground_truths > 0 else 0.0

    @property
    def precision(self):
        return self.nb_matched / self.nb_predictions if self.nb_predictions > 0 else 0.0

    def update_AP(self):
        self.AP += (self.recall - self.previous_recall) * self.precision
        self.previous_recall = self.recall


@dataclass
class EvaluationResult:
    nb_gt: int = 0
    nb_matched: int = 0
    precision: float = 0.0
    recall: float = 0.0
    AP: float = 0.0

    @property
    def F1(self) -> float:
        return 2 * (self.precision * self.recall) / (self.precision + self.recall)

    @classmethod
    def from_running(cls, other: RunningEvaluation):
        return cls(
            nb_gt=other.nb_ground_truths,
            nb_matched=other.nb_matched,
            precision=other.precision,
            recall=other.recall,
            AP=other.AP
        )

    def __add__(self, other: "EvaluationResult"):
        return EvaluationResult(
            other.nb_gt,
            other.nb_matched,
            self.precision + other.precision,
            self.recall + other.recall,
            self.AP + other.AP
        )

    def __truediv__(self, n: int | float):
        return EvaluationResult(
            self.nb_gt,
            self.nb_matched,
            self.precision / n,
            self.recall / n,
            self.AP / n
        )


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

    @staticmethod
    def DEFAULT_EVALUATION_GROUP() -> list[EvaluationGroup]:
        return [EvaluationGroup.ALL()]

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

    def _get_groups(self, groups: list[EvaluationGroup] | None):
        result = self.DEFAULT_EVALUATION_GROUP()
        result.extend(group for group in (groups or []) if group.name != "ALL")
        return result

    def _group_mean(self, metrics: list[dict], group_names: list[str]):
        """
        Compute the mean value over n metrics for each group.
        """
        n_metrics = len(metrics)
        results = {group_name: EvaluationResult() for group_name in group_names}
        for metric in metrics:
            for group_name, value in metric.items():
                results[group_name] += value / n_metrics
        return results

    def _group_increment_predictions(self, groups: dict[str, RunningEvaluation]):
        for running in groups.values():
            running.nb_predictions += 1


class PairIoUEvaluation(Evaluation):
    """
    Computes the non-interpolated AP, based on IoU metric, at a given IoU threshold.
    Evaluation based on the IoU between predicted and ground truth bounding boxes.
    The evaluation is based on greedy matching.
    Given a set of predictions and a set of ground truth annotations:
    - sort predictions by confidence (highest to lowest)
    - for each prediction:
        - compute the IoU to all unmatched ground truth annotations
        - filter the predictions whose IoUs are above the threshold
        - match the prediction to the closest ground truth annotation (max then argmax)
    """
    
    def __init__(self, iou_threshold: float, groups: list[EvaluationGroup] | None = None):
        self.iou_threshold = iou_threshold
        self.groups = self._get_groups(groups)

    def evaluate(self, predictions: list[Prediction], annotations: list[PairWrapper]):
        # define the groups
        groups = {}
        for group in self.groups:
            group_annotations = [annotation for annotation in annotations if group.is_applicable(annotation.pair)]
            groups[group.name] = RunningEvaluation(group, len(group_annotations))
        # sort by highest confidence order
        predictions = self.sort_by_confidence(predictions, ascending=False)
        matched = torch.zeros(len(annotations), dtype=torch.bool)
        # compute the AP based on greedy matching
        for prediction in predictions:
            # get the annotations corresponding to the image of the prediction
            image_ann_indices = torch.tensor([index for index, pair in enumerate(annotations) if pair.image_id == prediction.image_id])
            pred_annotations = [pair for pair in annotations if pair.image_id == prediction.image_id]
            if not pred_annotations: # no ground truth for this image, we skip it
                self._group_increment_predictions(groups)
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
                self._group_increment_predictions(groups)
                continue
            kept_together = torch.min(ious_button, ious_counterpart)
            kept_together[~keep] = -1.0
            _, best_gt_index = kept_together.max(dim=0)
            matched[image_ann_indices[best_gt_index]] = True
            # group update
            annotation = pred_annotations[best_gt_index]
            for running in groups.values():
                if not running.group.is_applicable(annotation.pair):
                    continue
                running.nb_predictions += 1
                running.nb_matched += 1
                running.update_AP()
        return {name: EvaluationResult.from_running(running) for name, running in groups.items()}


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

    def __init__(self, distance_threshold: float, image_size: int = 512, groups: list[EvaluationGroup] | None = None):
        self.distance_threshold = distance_threshold
        self.image_size = image_size
        self.groups = self._get_groups(groups)

    @override
    def evaluate(self, predictions: list[Prediction], annotations: list[PairWrapper]):
        # define the groups
        groups = {}
        for group in self.groups:
            group_annotations = [annotation for annotation in annotations if group.is_applicable(annotation.pair)]
            groups[group.name] = RunningEvaluation(group, len(group_annotations))
        # sort by highest confidence order
        predictions = self.sort_by_confidence(predictions, ascending=False)
        matched = torch.zeros(len(annotations), dtype=torch.bool)
        # compute the AP based on greedy matching
        for prediction in predictions:
            # get the annotations corresponding to the image of the prediction
            image_ann_indices = torch.tensor([index for index, pair in enumerate(annotations) if pair.image_id == prediction.image_id])
            pred_annotations = [pair for pair in annotations if pair.image_id == prediction.image_id]
            if not pred_annotations: # no ground truth for this image, we skip it
                self._group_increment_predictions(groups)
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
                self._group_increment_predictions(groups)
                continue
            kept_together = torch.max(distance_button, distance_counterpart)
            kept_together[~keep] = math.inf
            _, best_gt_index = kept_together.min(dim=0)
            matched[image_ann_indices[best_gt_index]] = True
            # now we compute the running AP for the groups
            annotation = pred_annotations[best_gt_index]
            for running in groups.values():
                if not running.group.is_applicable(annotation.pair):
                    continue
                running.nb_predictions += 1
                running.nb_matched += 1
                running.update_AP()
        return {name: EvaluationResult.from_running(running) for name, running in groups.items()}


class PairIoUAllEvaluation(Evaluation):
    def __init__(self, confidence_threshold: float = 0.0, groups: list[EvaluationGroup] | None = None):
        self.confidence_threshold = confidence_threshold
        self.groups = self._get_groups(groups)

    def evaluate(self, predictions: list[Prediction], annotations: list[PairWrapper]):
        predictions = self.filter_confidence(predictions, self.confidence_threshold)
        ap_50_evaluator = PairIoUEvaluation(iou_threshold=0.5, groups=self.groups)
        ap_75_evaluator = PairIoUEvaluation(iou_threshold=0.75, groups=self.groups)
        ap_90_evaluator = PairIoUEvaluation(iou_threshold=0.9, groups=self.groups)
        AP_50 = ap_50_evaluator.evaluate(predictions, annotations)
        AP_75 = ap_75_evaluator.evaluate(predictions, annotations)
        AP_90 = ap_90_evaluator.evaluate(predictions, annotations)
        group_names = list(group.name for group in self.groups)
        AP = self._group_mean([AP_50, AP_75, AP_90], group_names)
        metrics = {
            "AP_50": AP_50,
            "AP_75": AP_75,
            "AP_90": AP_90,
            "AP": AP
        }
        return metrics



class PairDistanceAllEvaluation(Evaluation):
    def __init__(self, confidence_threshold: float = 0.0, groups: list[EvaluationGroup] | None = None):
        self.confidence_threshold = confidence_threshold
        self.groups = self._get_groups(groups)

    def evaluate(self, predictions: list[Prediction], annotations: list[PairWrapper]):
        predictions = self.filter_confidence(predictions, self.confidence_threshold)
        ap_1_evaluator = PairDistanceEvaluation(distance_threshold=1.0, groups=self.groups)
        ap_3_evaluator = PairDistanceEvaluation(distance_threshold=3.0, groups=self.groups)
        ap_5_evaluator = PairDistanceEvaluation(distance_threshold=5.0, groups=self.groups)
        ap_10_evaluator = PairDistanceEvaluation(distance_threshold=10.0, groups=self.groups)
        AP_1 = ap_1_evaluator.evaluate(predictions, annotations)
        AP_3 = ap_3_evaluator.evaluate(predictions, annotations)
        AP_5 = ap_5_evaluator.evaluate(predictions, annotations)
        AP_10 = ap_10_evaluator.evaluate(predictions, annotations)
        group_names = list(group.name for group in self.groups)
        AP = self._group_mean([AP_1, AP_3, AP_5, AP_10], group_names)
        metrics = {
            "AP_1": AP_1,
            "AP_3": AP_3,
            "AP_5": AP_5,
            "AP_10": AP_10,
            "AP": AP
        }
        return metrics


class Evaluator:
    def __init__(self):
        """
        Evaluator class that evaluates a model on a given dataset.
        """
        self.evaluations = {}

    def add_evaluation(self, name: str, evaluation: Evaluation):
        self.evaluations[name] = evaluation

    def evaluate(self, outputs: list[tuple[Annotation, dict]]):
        # collect all pairs in a list
        all_pairs = []
        all_predictions = []
        results = {}
        # first, we need to put the predictions in the right format
        for i, (annotation, output) in enumerate(outputs):
            print(f"Image [{i + 1} / {len(outputs)}]", end="\r")
            all_pairs.extend([PairWrapper(pair, image_id=annotation.image.url) for pair in annotation.cloth.pairs])
            predictions = Prediction.from_outputs(output, image_id=annotation.image.url)
            all_predictions.extend(predictions)
        # then, we run the evaluations
        for name, evaluation in self.evaluations.items():
            results[name] = evaluation.evaluate(all_predictions, all_pairs)
        print()
        return results
