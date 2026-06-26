import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, List, Dict, Mapping
from scipy.optimize import linear_sum_assignment


class HungarianMatcher(nn.Module):
    """
    Matches predicted queries to GT buttons.

    Cost = classification cost + coordinate L1 cost
    """

    def __init__(self, cost_class: float = 1.0, cost_coord: float = 5.0):
        super().__init__()
        self.cost_class = cost_class
        self.cost_coord = cost_coord

        if cost_class == 0 and cost_coord == 0:
            raise ValueError("All costs cannot be 0")

    @torch.no_grad()
    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        """
        outputs:
            pred_logits: [B, Q, C+1]
            pred_buttons: [B, Q, 2]

        targets:
            list of dicts with:
                labels: [num_gt]
                buttons: [num_gt, 2]

        Returns:
            list of size B, each element is (pred_indices, target_indices)
        """
        pred_logits = outputs["pred_logits"]      # [B, Q, C+1]
        pred_buttons = outputs["pred_buttons"]    # [B, Q, 2]

        bs, num_queries = pred_logits.shape[:2] # predictions, get the batch size

        # Convert logits to probabilities
        out_prob = pred_logits.softmax(-1)  # [B, Q, C+1]
        out_coord = pred_buttons            # [B, Q, 2]

        indices = []

        for b in range(bs):
            tgt_labels = targets[b]["labels"]     # [num_gt] number of ground-truth buttons (2, 3, 4, 5, 6, 7)
            # print(tgt_labels)
            tgt_coords = targets[b]["buttons"]    # [num_gt, 2]
            # print(tgt_coords)

            if tgt_coords.numel() == 0:
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
            # out_coord[b]: [Q, 2], tgt_coords: [num_gt, 2]
            cost_coord = torch.cdist(out_coord[b], tgt_coords, p=2)
            # Total cost
            C = self.cost_class * cost_class + self.cost_coord * cost_coord
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

    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_dict: Dict[str, float],
        eos_coef: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict

        # Weight for classification:
        # class 0 = button
        # class 1 = no-object
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[-1] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, outputs, targets, indices):
        src_logits = outputs["pred_logits"]  # [B, Q, C+1]
        bs, num_queries, num_classes_plus_bg = src_logits.shape

        # default target class for all queries = no-object
        target_classes = torch.full(
            (bs, num_queries),
            fill_value=self.num_classes,  # index of no-object
            dtype=torch.int64,
            device=src_logits.device,
        )

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                target_classes[b, src_idx] = targets[b]["labels"][tgt_idx].to(src_logits.device)

        loss_ce = F.cross_entropy(
            src_logits.transpose(1, 2),  # [B, C+1, Q]
            target_classes,
            weight=self.empty_weight, # type: ignore
        )
        return {"loss_ce": loss_ce}

    def loss_buttons(self, outputs, targets, indices):
        src_coords = outputs["pred_buttons"]  # [B, Q, 2]
        # NEW: we now define intermediate losses
        intermediate_refpoints = outputs["intermediate_reference_points"] # num_layers * [B, Q, 2]
        # -> [num_layers, B, Q, 2]
        intermediate_refpoints = torch.stack(intermediate_refpoints, dim=0) # [num_layers, B, Q, 2]
        # split into buttons and keypoints
        src_button_coords = intermediate_refpoints[:, :, :, :]

        matched_button_coords = []
        matched_button_target = []
        # get the matched button and keypoint predictions
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                matched_button_coords.append(src_button_coords[:, b, src_idx])
                matched_button_target.append(targets[b]["buttons"][tgt_idx].to(src_coords.device))
        # if there is no predictions, then the loss is null
        if len(matched_button_coords) == 0:
            loss_button = src_coords.sum() * 0.0
        else:
            # B * [num_layers, n_pred, 2] -> [num_layers, B * n_pred, 2]
            matched_button_coords = torch.cat(matched_button_coords, dim=1)
            n_layers, total_pred, _ = matched_button_coords.size()

            # -> [B * n_pred, 2]
            matched_button_target = torch.cat(matched_button_target, dim=0)
            # [B * n_pred, 2] -> [num_layers, B * n_pred, 2]
            matched_button_target = matched_button_target[None, :, :].expand(n_layers, total_pred, 2)
            # now we define the loss
            # we first compute two independent losses for buttons and keypoints
            loss_buttons = (torch.linalg
                .vector_norm(matched_button_coords - matched_button_target, ord=1, dim=2)
                .mean(dim=1)) # take the mean over batches and predictions
            # TODO: Apply here some parameter to control the contribution of each level
            #       We can think of it as a learned parameter inside the network, or hard-coded
            loss_button = loss_buttons.mean()
        return {"loss_button": loss_button}


    def forward(self, outputs, targets):
        indices = self.matcher(outputs, targets)

        losses = {}
        losses.update(self.loss_labels(outputs, targets, indices))
        losses.update(self.loss_buttons(outputs, targets, indices))

        total_loss = 0.0
        for k, v in losses.items():
            total_loss = total_loss + self.weight_dict[k] * v

        losses["loss"] = total_loss
        return losses