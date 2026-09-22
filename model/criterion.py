import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Dict, Mapping
from scipy.optimize import linear_sum_assignment
from .utils import compute_giou


class HungarianMatcher(nn.Module):
    """
    Matches predicted queries to ground-truth pairs.
    """
    def __init__(self, cost_class: float = 1.0, cost_coord: float = 5.0, cost_giou: float = 1.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_coord = cost_coord
        self.cost_giou = cost_giou

        if cost_class == 0. and cost_coord == 0. and cost_giou == 0.:
            raise ValueError("All costs cannot be 0")

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        """
        outputs:
            pred_logits: [B, Q, C+1]
            pred_boxes: [B, Q, RpQ, 4]

        targets:
            list of dicts with:
                labels: [num_gt]
                buttons: [num_gt, 4]

        Returns:
            list of size B, each element is (pred_indices, target_indices)
        """
        pred_logits = outputs["pred_logits"]      # [B, Q, C+1]
        # WARNING: now, the predictions are of shape [B, Q, RqP, 4]
        pred_positions = outputs["pred_boxes"]    # [B, Q, RqP, 4]
        # we split into buttons and keypoints
        pred_buttons = pred_positions[:, :, 0, :] # [B, Q, 4]
        pred_counterparts = pred_positions[:, :, 1, :] # [B, Q, 4]

        bs, num_queries = pred_logits.shape[:2] # predictions, get the batch size

        # Convert logits to probabilities
        out_prob = pred_logits.softmax(-1)  # [B, Q, C+1]
        out_coord = pred_buttons            # [B, Q, 4]
        pred_holes = pred_counterparts

        indices = []

        for b in range(bs):
            tgt_labels = targets[b]["labels"]     # [num_gt] number of ground-truth buttons (2, 3, 4, 5, 6, 7)
            tgt_buttons = targets[b]["buttons"]    # [num_gt, 4]
            tgt_holes = targets[b]["counterparts"]   # [num_gt, 4]

            if tgt_buttons.numel() == 0:
                indices.append((
                    torch.empty(0, dtype=torch.int64),
                    torch.empty(0, dtype=torch.int64)
                ))
                continue

            # Classification cost:
            # want high probability for the target class (class 0 here)
            # cost shape [Q, num_gt]
            cost_class = -out_prob[b][:, tgt_labels]

            # Coordinate cost
            # out_coord[b]: [Q, 4], tgt_buttons: [num_gt, 4]
            cost_coord = torch.cdist(out_coord[b], tgt_buttons, p=1)
            cost_hole = torch.cdist(pred_holes[b], tgt_holes, p=1)
            # compute the GIoU cost for each pair
            giou_buttons = compute_giou(
                pred_buttons[b][:, None, :], # [Q, 1, 4]
                tgt_buttons[None, :, :], # [1, N, 4]
            ) # [Q, N]
            giou_counterparts = compute_giou(
                pred_holes[b][:, None, :], # [Q, 1, 4]
                tgt_holes[None, :, :], # [1, N, 4]
            ) # [Q, N]
            cost_giou = -(giou_buttons + giou_counterparts)
            # total cost
            C = self.cost_class * cost_class + self.cost_coord * (cost_coord + cost_hole) + self.cost_giou * cost_giou
            C = C.cpu()

            pred_ind, tgt_ind = linear_sum_assignment(C)
            indices.append((
                torch.as_tensor(pred_ind, dtype=torch.int64),
                torch.as_tensor(tgt_ind, dtype=torch.int64)
            ))
        return indices


class SetCriterion(nn.Module):
    """
    DETR-style criterion for:
      - class prediction
      - button coordinate prediction
    """

    def __init__(self, num_classes: int, matcher: HungarianMatcher, weight_dict: Dict[str, float], eos_coef: float = 0.1):
        """
        eos_coef: relative classification weight applied to the no-object category
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

        # Weight for classification:
        # class 0 = button
        # class 1 = no-object
        empty_weight = torch.ones(num_classes + 1) # [C+1]
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def class_loss(self, outputs, targets, indices):
        """
        Compute the cross-entropy loss of the predicted pair classes.
        """
        src_logits = outputs["pred_logits"]  # [B, Q, C+1]
        bs, num_queries, num_classes = src_logits.shape
        # default target class for all queries: no-object
        target_classes = torch.full(
            (bs, num_queries),
            fill_value=self.num_classes,  # index of no-object
            dtype=torch.int64,
            device=src_logits.device,
        ) # [B, Q]

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[b, src_idx] = targets[b]["labels"][tgt_idx].to(src_logits.device)

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),  # [B, C+1, Q]
            target_classes, # [B, Q]
            weight=self.empty_weight, # type: ignore
        )
        return {"loss_ce": loss_ce}

    def pair_loss(self, outputs, targets, indices):
        """
        Compute the L1 and GIoU losses from the predicted pairs.
        """
        src_coords = outputs["pred_boxes"]  # [B, Q, RqP, 4]
        # split into buttons and counterparts
        src_button_coords = src_coords[:, :, 0, :] # [B, Q, 4]
        src_counterparts_coords = src_coords[:, :, 1, :] # [B, Q, 4]

        matched_button_coords = []
        matched_counterparts_coords = []
        matched_button_target = []
        matched_counterparts_target = []
        # get the matched button and counterpart predictions from the Hungarian Matcher indices
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) == 0:
                continue
            matched_button_coords.append(src_button_coords[b, src_idx])
            matched_counterparts_coords.append(src_counterparts_coords[b, src_idx])
            matched_button_target.append(targets[b]["buttons"][tgt_idx].to(src_coords.device))
            matched_counterparts_target.append(targets[b]["counterparts"][tgt_idx].to(src_coords.device))
        # compute the losses
        pair_losses = {}
        if len(matched_button_coords) == 0: # there may be no targets or no predictions
            L1_losses = self._null_L1_loss(src_coords.device)
            GIoU_losses = self._null_GIoU_loss(src_coords.device)
        else:
            # put the tensors in the right size
            matched_button_coords = torch.cat(matched_button_coords, dim=0) # [B * Q, 4]
            matched_counterparts_coords = torch.cat(matched_counterparts_coords, dim=0) # [B * Q, 4]
            matched_button_target = torch.cat(matched_button_target, dim=0)
            matched_counterparts_target = torch.cat(matched_counterparts_target, dim=0)
            # now we compute both the L1 loss and the GIoU loss
            L1_losses = self.L1_loss(matched_button_coords, matched_button_target, matched_counterparts_coords, matched_counterparts_target)
            GIoU_losses = self.GIoU_loss(matched_button_coords, matched_button_target, matched_counterparts_coords, matched_counterparts_target)
        pair_losses.update(L1_losses)
        pair_losses.update(GIoU_losses)
        return pair_losses

    def L1_loss(self, pred_button_coords, target_button_coords, pred_counterpart_coords, target_counterpart_coords):
        """
        Compute the L1 loss.
        """
        loss_buttons = F.l1_loss(pred_button_coords, target_button_coords, reduction="mean")
        loss_counterparts = F.l1_loss(pred_counterpart_coords, target_counterpart_coords, reduction="mean")
        loss_pair = loss_buttons + loss_counterparts
        return {
            "L1_loss_pair": loss_pair,
            "L1_loss_buttons": loss_buttons,
            "L1_loss_counterparts": loss_counterparts
        }

    
    def GIoU_loss(self, pred_button_coords, target_button_coords, pred_counterpart_coords, target_counterpart_coords):
        """
        Compute the GIoU loss.
        """
        giou_buttons = compute_giou(pred_button_coords, target_button_coords)
        giou_buttons_loss = (1 - giou_buttons).mean()
        giou_counterparts = compute_giou(pred_counterpart_coords, target_counterpart_coords)
        giou_counterparts_loss = (1 - giou_counterparts).mean()
        # the pair GIoU loss is the sum of both components losses
        giou_loss = giou_buttons_loss + giou_counterparts_loss
        return {
            "GIoU_loss_pair": giou_loss,
            "GIoU_loss_buttons": giou_buttons_loss,
            "GIoU_loss_counterparts": giou_counterparts_loss
        }

    def _null_L1_loss(self, device):
        """
        Get a null L1 loss.
        """
        return {
            "L1_loss_pair": torch.tensor(0.0, device=device),
            "L1_loss_buttons": torch.tensor(0.0, device=device),
            "L1_loss_counterparts": torch.tensor(0.0, device=device)
        }

    def _null_GIoU_loss(self, device):
        """
        Get a null GIoU loss.
        """
        return {
            "GIoU_loss_pair": torch.tensor(0.0, device=device),
            "GIoU_loss_buttons": torch.tensor(0.0, device=device),
            "GIoU_loss_counterparts": torch.tensor(0.0, device=device)
        }


    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)
        # compute the losses
        losses = {}
        losses.update(self.class_loss(outputs, targets, indices))
        losses.update(self.pair_loss(outputs, targets, indices))
        # the total loss is the weighted sum of the losses
        total_loss = (
            losses["class_loss"] * self.weight_dict["class_loss"] +
            losses["L1_loss_pair"] * self.weight_dict["L1_loss"] +
            losses["GIoU_loss_pair"] * self.weight_dict["GIoU_loss"]
        )
        losses["loss"] = total_loss
        return losses