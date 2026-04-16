import json
import random
from pathlib import Path
from typing import List, Dict, Tuple
from math import exp
from xml.parsers.expat import model
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from prtr import PRTR
from criterion import HungarianMatcher, SetCriterion
from transforms import *


class ButtonDataset(Dataset):
    """
    Images will be resized by the `image_size` parameter.
    """

    def __init__(self, root: str):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.ann_dir = self.root / "annotations"

        self.ann_paths = sorted(self.ann_dir.glob("*.json"))
        # check if there are annotations
        if len(self.ann_paths) == 0:
            raise RuntimeError(f"No JSON files found in {self.ann_dir}")

    def transform_image(self, image, labels):
        trainer = get_trainer() # type: ignore
        to_tensor = trainer.transforms # type: ignore
        return to_tensor(image, labels) # type: ignore

    def __len__(self):
        return len(self.ann_paths)

    def __getitem__(self, idx: int):
        """
        Get one item of the dataset.
        """
        # load the json
        ann_path = self.ann_paths[idx]
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)
        # take the relevant data
        name = ann["name"]
        width = ann["width"]
        height = ann["height"]
        img_path = self.images_dir / f"{name}.png"
        if not img_path.exists():
            raise FileNotFoundError(f"Missing image for annotation: {img_path}")
        # get the annotations
        buttons = ann.get("buttons", [])
        coords = []
        for b in buttons:
            # WARNING: Coordinates are in Blender representation (top -> bottom).
            # We need to convert them in bottom -> top
            if "x_ndc" in b and "y_ndc" in b:
                x = float(b["x_ndc"])
                y = 1 - float(b["y_ndc"]) # convert in bottom -> top representation
            else: # if normalized coordinates are not present, compute them
                x = float(b["x_px"]) / float(width)
                y = float(b["y_px"]) / float(height)
            coords.append([x, y])
        # open the corresponding image (RGB mode)
        image = Image.open(img_path).convert("RGB")
        image, coords = self.transform_image(image, coords)  # [3, H, W], values in [0,1]
        # this should not happen
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


class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device, dataloader, val_dataloader):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.val_dataloader = val_dataloader
        self.dataloader = dataloader
        self.epoch = 0
        self.best_val_loss = float("inf")
        self.last_was_best = False
        self._transforms = None

    def train_one_epoch(self):
        self.model.train()
        self.criterion.train()
        running = {
            "loss": 0.0,
            "loss_ce": 0.0,
            "loss_button": 0.0,
        }
        for images, targets in self.dataloader:
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            # forward into the model and comoute the loss
            outputs = self.model(images)
            losses = self.criterion(outputs, targets)
            # backpropagate
            self.optimizer.zero_grad()
            losses["loss"].backward()
            self.optimizer.step()
            # compute the total loss of the epoch
            for k in running:
                running[k] += losses[k].item()
        # return the mean of the loss for the epoch
        n = len(self.dataloader)
        return {k: v / n for k, v in running.items()}
    
    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        self.criterion.eval()
        # define the losses
        running = {"loss": 0.0, "loss_ce": 0.0, "loss_button": 0.0}
        for images, targets in self.val_dataloader:
            images = images.to(self.device)
            targets = [{k: v.to(self.device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
            outputs = self.model(images)
            losses = self.criterion(outputs, targets)
            for k in running:
                running[k] += losses[k].item()
        # compute the mean of the loss
        n = len(self.val_dataloader)
        return {k: v / n for k, v in running.items()}
    
    def step(self):
        self._transforms = self._get_epoch_transforms()
        train_stats = self.train_one_epoch()
        val_stats = self.evaluate()
        self.scheduler.step()
        # update the metrics
        if val_stats["loss"] < self.best_val_loss:
            self.best_val_loss = val_stats["loss"]
            self.last_was_best = True
        else:
            self.last_was_best = False
        self.epoch += 1
        return train_stats, val_stats

    def resume(self, checkpoint_path: Path):
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        saved_epoch = checkpoint.get("epoch", None)
        if saved_epoch is None:
            raise RuntimeError("The checkpoint does not contain an 'epoch' field.")
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.epoch = saved_epoch
        self.best_val_loss = checkpoint.get("val_loss", float("inf"))
    
    def save_checkpoint(self, save_path: Path):
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "val_loss": self.best_val_loss,
            },
            save_path
        )

    @property
    def transforms(self):
        return self._transforms

    def _get_epoch_transforms(self) -> ComposeWithLabels:
        image_sizes = [512, 640, 768, 896]
        image_size = random.choice(image_sizes)
        return ComposeWithLabels([
            ComposeWrapper(transforms.Resize((image_size, image_size))),
            RandomSafeErasing(p=0.6),
            # RandomButtonErasing(p=0.2),
            RandomHorizontalFlip(),
            RandomHorizontalTranslation(p=0.5, min=-0.3, max=0.3),
            RandomVerticalTranslation(p=0.5, min=-0.3, max=0.3),
            RandomRotation(p=0.5, min_angle=-45, max_angle=45),
            ComposeWrapper(transforms.RandomGrayscale(p=0.1)),
            RandomProgressiveFoveatedBlur(p=0.5, current_epoch=self.epoch),
            ComposeWrapper(transforms.ToTensor()),
            ComposeWrapper(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        ])


_trainer = None

def get_trainer():
    global _trainer
    if _trainer is not None:
        return _trainer

def init_trainer(model_config):
    global _trainer
    # configuration
    dataset_root = "dataset"
    training_parameters = model_config["training_parameters"]
    model_parameters = model_config["parameters"]
    model_name = model_config["model_name"]
    batch_size = training_parameters["batch_size"]
    num_epochs = training_parameters["num_epochs"]
    lr = training_parameters["lr"]
    weight_decay = training_parameters["weight_decay"]
    train_split = training_parameters["train_split"]
    num_workers = training_parameters["num_workers"]
    seed = training_parameters["seed"]
    cost_class = training_parameters["cost_class"]
    cost_coord = training_parameters["cost_coord"]
    # use the GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # sets the seed
    random.seed(seed)
    torch.manual_seed(seed)

    # load the full dataset
    full_dataset = ButtonDataset(dataset_root)
    n_total = len(full_dataset)
    n_train = int(train_split * n_total)
    n_val = n_total - n_train
    # then split the dataset into training and evaluation
    train_dataset, val_dataset = random_split(
        full_dataset,
        [n_train, n_val],
        generator=torch.Generator().manual_seed(seed)
    )
    # load the datasets
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

    # create the model
    model = PRTR(model_name, **model_parameters)
    model = model.to(device)
    # display the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_params_backbone = sum(p.numel() for p in model.backbone.parameters())
    print("# Number of parameters:", total_params)
    print("# Number of parameters of the backbone:", total_params_backbone)
    print("# Real parameters count:", total_params - total_params_backbone)
    # create the loss calculator
    matcher = HungarianMatcher(cost_class=cost_class, cost_coord=cost_coord)
    criterion = SetCriterion(
        num_classes=model_parameters["num_classes"],
        matcher=matcher,
        weight_dict={
            "loss_ce": cost_class,
            "loss_button": cost_coord,
        },
        eos_coef=0.1,
    ).to(device)
    # optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    # scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

    # create the trainer
    _trainer = Trainer(model, criterion, optimizer, scheduler, device, train_loader, val_loader)
    return _trainer


def main(model_config, resume_path=None):
    save_dir = Path("checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = init_trainer(model_config)
    if resume_path is not None:
        trainer.resume(Path(resume_path))

    num_epochs = model_config["training_parameters"]["num_epochs"]
    start_epoch = trainer.epoch
    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_stats, val_stats = trainer.step()
        # log the epoch results
        print(
            f"Epoch [{epoch+1}/{start_epoch + num_epochs}] | "
            f"train loss: {train_stats['loss']:.4f} "
            f"(ce={train_stats['loss_ce']:.4f}, btn={train_stats['loss_button']:.4f}) | "
            f"val loss: {val_stats['loss']:.4f} "
            f"(ce={val_stats['loss_ce']:.4f}, btn={val_stats['loss_button']:.4f})"
        )
        # save latest
        trainer.save_checkpoint(save_dir / "last.pt")
        if trainer.last_was_best:
            trainer.save_checkpoint(save_dir / "best.pt")
            print(f"    New best model saved with val loss: {val_stats['loss']:.4f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    # configuration
    config_path = Path("model.json")
    with open(config_path, "r") as file:
        model_config = json.load(file)
    main(model_config, resume_path=args.resume)
