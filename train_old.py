import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from math import exp

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

# Import your model here
from prtr import PRTR


# ============================================================
# Dataset
# ============================================================

class ButtonDataset(Dataset):
    """
    Dataset structure:
        dataset/
            images/
                cloth_xxx.png
            annotations/
                cloth_xxx.json
    """

    def __init__(self, root: str, image_size: int = 1024):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.ann_dir = self.root / "annotations"
        self.image_size = image_size

        ann_paths = sorted(self.ann_dir.glob("*.json"))
        to_keep = []
        for ann_path in ann_paths:
            with open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
                params = ann["parameters"]
                good = True
                for param in params:
                    name = param["name"]
                    value = param["value"]
                    if (name == "brightness" and value < 1000) or (name == "button_scale" and value < 0.9):
                        good = False
                if True:
                    to_keep.append(ann_path)
        self.ann_paths = to_keep
        if len(self.ann_paths) == 0:
            raise RuntimeError(f"No JSON files found in {self.ann_dir}")

        self.to_tensor = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.ann_paths)

    def __getitem__(self, idx: int):
        ann_path = self.ann_paths[idx]

        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

        name = ann["name"]
        width = ann["width"]
        height = ann["height"]

        img_path = self.images_dir / f"{name}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image for annotation: {img_path}")

        image = Image.open(img_path).convert("RGB")
        image = self.to_tensor(image)  # [3, H, W], values in [0,1]

        buttons = ann.get("buttons", [])
        coords = []

        for b in buttons:
            # Prefer normalized coordinates if present
            if "x_ndc" in b and "y_ndc" in b:
                x = float(b["x_ndc"])
                y = float(b["y_ndc"])
            if True:
                x = float(b["x_px"]) / float(width)
                y = float(b["y_px"]) / float(height)

            coords.append([x, y])

        if len(coords) == 0:
            target_buttons = torch.zeros((0, 2), dtype=torch.float32)
        else:
            target_buttons = torch.tensor(coords, dtype=torch.float32)

        target = {
            "labels": torch.zeros((len(target_buttons),), dtype=torch.int64),  # only one class: button => class 0
            "buttons": target_buttons,  # [num_buttons, 2], normalized
            "image_id": name,
            "size": torch.tensor([height, width], dtype=torch.int64),
        }

        return image, target


def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, dim=0)
    return images, list(targets)


# ============================================================
# Hungarian matching
# ============================================================

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

        bs, num_queries = pred_logits.shape[:2]

        # Convert logits to probabilities
        out_prob = pred_logits.softmax(-1)  # [B, Q, C+1]
        out_coord = pred_buttons            # [B, Q, 2]

        indices = []

        for b in range(bs):
            tgt_labels = targets[b]["labels"]     # [num_gt]
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
            # print(cost_class)

            # Coordinate L1 cost
            # out_coord[b]: [Q, 2], tgt_coords: [num_gt, 2]
            # print(out_coord[b], tgt_coords, sep="\n=============\n")
            cost_coord = torch.cdist(out_coord[b], tgt_coords, p=2)
            # print("++++ COSTS ++++")
            # print(cost_coord)
            # print()

            # Total cost
            C = self.cost_class * cost_class + self.cost_coord * cost_coord
            C = C.cpu()

            pred_ind, tgt_ind = linear_sum_assignment(C)
            indices.append((
                torch.as_tensor(pred_ind, dtype=torch.int64),
                torch.as_tensor(tgt_ind, dtype=torch.int64)
            ))

        return indices


# ============================================================
# Loss
# ============================================================

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
            weight=self.empty_weight,
        )
        return {"loss_ce": loss_ce}

    def loss_buttons(self, outputs, targets, indices):
        src_coords = outputs["pred_buttons"]  # [B, Q, 2]
        bs, q, _ = src_coords.shape

        matched_pred = []
        matched_tgt = []

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if len(src_idx) > 0:
                matched_pred.append(src_coords[b, src_idx])
                matched_tgt.append(targets[b]["buttons"][tgt_idx].to(src_coords.device))

        if len(matched_pred) == 0:
            loss_button = torch.tensor(0.0, device=src_coords.device)
        else:
            matched_pred = torch.cat(matched_pred, dim=0)
            matched_tgt = torch.cat(matched_tgt, dim=0)
            loss_button = F.l1_loss(matched_pred, matched_tgt, reduction="sum")
            loss_button /= bs
            loss_button = torch.norm(matched_pred - matched_tgt, dim=-1).mean()
            # loss_button = loss_button + (exp(15 * loss_button) - 1)

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


# ============================================================
# Train / eval loops
# ============================================================

def train_one_epoch(model, criterion, dataloader, optimizer, device):
    model.train()
    criterion.train()

    running = {
        "loss": 0.0,
        "loss_ce": 0.0,
        "loss_button": 0.0,
    }

    for images, targets in dataloader:
        images = images.to(device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        # print(targets)

        outputs = model(images)
        losses = criterion(outputs, targets)
        # print("LOSSES")
        # print(losses["loss_button"])

        optimizer.zero_grad()
        losses["loss"].backward()
        optimizer.step()

        for k in running:
            running[k] += losses[k].item()

    n = len(dataloader)
    return {k: v / n for k, v in running.items()}


@torch.no_grad()
def evaluate(model, criterion, dataloader, device):
    model.eval()
    criterion.eval()

    running = {
        "loss": 0.0,
        "loss_ce": 0.0,
        "loss_button": 0.0,
    }

    for images, targets in dataloader:
        images = images.to(device)
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        outputs = model(images)
        losses = criterion(outputs, targets)

        for k in running:
            running[k] += losses[k].item()

    n = len(dataloader)
    return {k: v / n for k, v in running.items()}


# ============================================================
# Main
# ============================================================

def main(resume_from_best=False):
    # ----------------------------
    # Config
    # ----------------------------
    dataset_root = "dataset"
    batch_size = 2
    num_epochs = 50
    lr = 1e-4
    weight_decay = 1e-4
    train_split = 0.9
    num_workers = 4
    seed = 42

    # DETR-style task config
    num_classes = 1   # only "button"
    num_queries = 10  # should be >= max number of buttons in one image

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    random.seed(seed)
    torch.manual_seed(seed)

    # ----------------------------
    # Dataset
    # ----------------------------
    full_dataset = ButtonDataset(dataset_root)
    n_total = len(full_dataset)
    n_train = int(train_split * n_total)
    n_val = n_total - n_train

    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # ----------------------------
    # Model
    # ----------------------------
    model = PRTR(num_classes=num_classes)
    model = model.to(device)

    # ----------------------------
    # Loss
    # ----------------------------
    matcher = HungarianMatcher(cost_class=1.0, cost_coord=5.0)

    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict={
            "loss_ce": 1.0,
            "loss_button": 5.0,
        },
        eos_coef=0.1,
    ).to(device)

    # ----------------------------
    # Optimizer
    # ----------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # Optional LR scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    start_epoch = 0
    best_val_loss = float("inf")

    best_ckpt_path = Path("checkpoints") / "best.pt"

    if resume_from_best and best_ckpt_path.exists():
        print(f"Loading checkpoint from {best_ckpt_path}")

        checkpoint = torch.load(best_ckpt_path, map_location=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        start_epoch = checkpoint["epoch"]
        best_val_loss = checkpoint["val_loss"]

        # IMPORTANT: move optimizer tensors to correct device
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)
        
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        print(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ----------------------------
    # Training loop
    # ----------------------------
    best_val_loss = float("inf")
    save_dir = Path("checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_stats = train_one_epoch(model, criterion, train_loader, optimizer, device)
        val_stats = evaluate(model, criterion, val_loader, device)

        scheduler.step()

        print(
            f"Epoch [{epoch+1}/{num_epochs}] | "
            f"train loss: {train_stats['loss']:.4f} "
            f"(ce={train_stats['loss_ce']:.4f}, btn={train_stats['loss_button']:.4f}) | "
            f"val loss: {val_stats['loss']:.4f} "
            f"(ce={val_stats['loss_ce']:.4f}, btn={val_stats['loss_button']:.4f})"
        )

        # Save latest
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_stats["loss"],
            },
            save_dir / "last.pt"
        )

        # Save best
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_stats["loss"],
                },
                save_dir / "best.pt"
            )
            print(f"  Saved new best checkpoint: val_loss={best_val_loss:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", choices=["none", "best", "last"], default="none")
    args = parser.parse_args()
    resume = args.resume == "best"
    main(resume)