import json
import random
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T

from prtr import PRTR
from criterion import HungarianMatcher, SetCriterion

from transforms import (
    ComposeWithLabels,
    ComposeWrapper,
    RandomSafeErasing,
    RandomHorizontalFlip,
    RandomHorizontalTranslation,
    RandomVerticalTranslation,
)


class ButtonDataset(Dataset):
    """
    Images are transformed by the provided transform callable.

    Expected target coordinate convention after transform:
        buttons: [N, 2], normalized [x, y]
        x = 0 left,   x = 1 right
        y = 0 top,    y = 1 bottom
    """

    def __init__(self, root: str, transform: Optional[Callable] = None):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.ann_dir = self.root / "annotations"
        self.transform = transform

        self.ann_paths = sorted(self.ann_dir.glob("*.json"))
        if len(self.ann_paths) == 0:
            raise RuntimeError(f"No JSON files found in {self.ann_dir}")

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

        buttons = ann.get("buttons", [])
        coords = []

        for b in buttons:
            center = b["center"]
            if "x_ndc" in center and "y_ndc" in center:
                x = float(center["x_ndc"])
                y = 1.0 - float(center["y_ndc"])
            else:
                x = float(center["x_px"]) / float(width)
                y = float(center["y_px"]) / float(height)

            coords.append([x, y])

        image = Image.open(img_path).convert("RGB")

        # apply transforms to the image
        if self.transform is not None:
            image, coords = self.transform(image, coords)

        if torch.is_tensor(coords):
            target_buttons = coords.to(dtype=torch.float32)
        elif len(coords) == 0:
            target_buttons = torch.zeros((0, 2), dtype=torch.float32)
        else:
            target_buttons = torch.tensor(coords, dtype=torch.float32)

        if torch.is_tensor(image):
            out_h, out_w = image.shape[-2:]
        else:
            out_w, out_h = image.size

        target = {
            "labels": torch.zeros((len(target_buttons),), dtype=torch.int64),
            "buttons": target_buttons,
            "image_id": name,
            "size": torch.tensor([out_h, out_w], dtype=torch.int64),
        }

        return image, target


class RandomTrainTransform:
    """
    Preserves your original behavior:
    every sample randomly chooses one resize size from [512, 640, 768, 896].
    """

    def __init__(self, sizes=(512,)):
        self.sizes = sizes

    def __call__(self, image, labels):
        size = random.choice(self.sizes)

        transform = ComposeWithLabels([
            ComposeWrapper(T.Resize((size, size))),
            RandomSafeErasing(p=0.6),
            # RandomButtonErasing(p=0.2),
            RandomHorizontalFlip(),
            RandomHorizontalTranslation(p=0.5, min=-0.3, max=0.3),
            RandomVerticalTranslation(p=0.5, min=-0.3, max=0.3),
            # RandomRotation(p=0.5, min_angle=-45, max_angle=45),
            ComposeWrapper(T.RandomGrayscale(p=0.1)),
            ComposeWrapper(T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            )),
            ComposeWrapper(T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))),
            # RandomProgressiveFoveatedBlur(p=0.5, current_epoch=self.epoch),
            ComposeWrapper(T.ToTensor()),
            ComposeWrapper(T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )),
        ])

        return transform(image, labels)


def make_val_transform(size: int = 640):
    """
    Deterministic validation transform.
    No erasing, no jitter, no random translation, no random blur.
    """
    return ComposeWithLabels([
        ComposeWrapper(T.Resize((size, size))),
        ComposeWrapper(T.ToTensor()),
        ComposeWrapper(T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )),
    ])


def collate_fn(batch):
    """
    Padds and computes each image mask for the given batch.
    """
    images, targets = zip(*batch)
    # get the maximum image size of the batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)

    batch_size = len(images)
    channels = images[0].shape[0]
    dtype = images[0].dtype
    # initialize the tensors that will contain the batch
    # the mask is: 0 means image data ; 1 means padded data (to be ignored)
    # [B, C, max H, max W]
    padded_images = torch.zeros((batch_size, channels, max_h, max_w), dtype=dtype)
    padding_mask = torch.ones((batch_size, max_h, max_w), dtype=torch.bool)

    new_targets = []
    for i, (img, tgt) in enumerate(zip(images, targets)):
        _, h, w = img.shape
        padded_images[i, :, :h, :w] = img
        padding_mask[i, :h, :w] = False

        tgt = dict(tgt)
        tgt["size"] = torch.tensor([h, w], dtype=torch.int64)
        new_targets.append(tgt)

    return padded_images, padding_mask, new_targets


def seed_worker(worker_id: int):
    """
    Seeds Python's random module inside each DataLoader worker.
    PyTorch seeds workers automatically, but your custom transforms use Python random.
    """
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


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

    def _move_targets_to_device(self, targets):
        return [
            {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in t.items()}
            for t in targets
        ]

    def _accumulate_losses(self, running: Dict[str, float], losses: Dict[str, torch.Tensor]):
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v.item())

    def _mean_losses(self, running: Dict[str, float], n: int):
        if n == 0:
            raise RuntimeError("Dataloader produced zero batches.")
        return {k: v / n for k, v in running.items()}

    def train_one_epoch(self):
        self.model.train()
        self.criterion.train()

        running: Dict[str, float] = {}

        for images, padding_mask, targets in self.dataloader:
            images = images.to(self.device, non_blocking=True)
            padding_mask = padding_mask.to(self.device, non_blocking=True)
            targets = self._move_targets_to_device(targets)

            outputs = self.model(images, padding_mask)
            losses = self.criterion(outputs, targets)

            self.optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            self.optimizer.step()

            self._accumulate_losses(running, losses)

        return self._mean_losses(running, len(self.dataloader))

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        self.criterion.eval()

        running: Dict[str, float] = {}

        for images, padding_mask, targets in self.val_dataloader:
            images = images.to(self.device, non_blocking=True)
            padding_mask = padding_mask.to(self.device, non_blocking=True)
            targets = self._move_targets_to_device(targets)

            outputs = self.model(images, padding_mask)
            losses = self.criterion(outputs, targets)

            self._accumulate_losses(running, losses)

        return self._mean_losses(running, len(self.val_dataloader))

    def step(self):
        train_stats = self.train_one_epoch()
        val_stats = self.evaluate()

        self.scheduler.step()

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
            save_path,
        )


def format_stats(stats: Dict[str, float]):
    parts = [
        f"loss={stats.get('loss', 0.0):.4f}",
        f"ce={stats.get('loss_ce', 0.0):.4f}",
        f"btn={stats.get('loss_button', 0.0):.4f}",
    ]

    if "loss_attn" in stats:
        parts.append(f"attn={stats['loss_attn']:.4f}")

    return "(" + ", ".join(parts) + ")"


def init_trainer(model_config):
    dataset_root = model_config["dataset"]
    training_parameters = model_config["training_parameters"]
    model_parameters = model_config["parameters"]
    model_name = model_config["model_name"]

    batch_size = training_parameters["batch_size"]
    lr = training_parameters["lr"]
    weight_decay = training_parameters["weight_decay"]
    train_split = training_parameters["train_split"]
    num_workers = training_parameters["num_workers"]
    seed = training_parameters["seed"]

    cost_class = training_parameters["cost_class"]
    cost_coord = training_parameters["cost_coord"]
    cost_attn_map = training_parameters["cost_attn_map"]

    val_size = training_parameters.get("val_size", 640)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    random.seed(seed)
    torch.manual_seed(seed)

    # Build a temporary dataset only to know length and annotation order.
    base_dataset = ButtonDataset(dataset_root, transform=None)
    n_total = len(base_dataset)

    n_train = int(train_split * n_total)
    n_val = n_total - n_train

    if n_train <= 0 or n_val <= 0:
        raise ValueError(
            f"Invalid split: n_total={n_total}, n_train={n_train}, n_val={n_val}. "
            f"Check training_parameters['train_split']."
        )

    split_generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n_total, generator=split_generator).tolist()

    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = Subset(
        ButtonDataset(dataset_root, transform=RandomTrainTransform()),
        train_indices,
    )

    val_dataset = Subset(
        ButtonDataset(dataset_root, transform=make_val_transform(size=val_size)),
        val_indices,
    )

    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
    )

    model = PRTR(model_name, **model_parameters)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    total_params_backbone = sum(p.numel() for p in model.backbone.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("# Number of parameters:", total_params)
    print("# Number of parameters of the backbone:", total_params_backbone)
    print("# Trainable parameters:", trainable_params)

    matcher = HungarianMatcher(
        cost_class=cost_class,
        cost_coord=cost_coord,
    )

    criterion = SetCriterion(
        num_classes=model_parameters["num_classes"],
        matcher=matcher,
        weight_dict={
            "loss_ce": cost_class,
            "loss_button": cost_coord,
            "loss_attn": cost_attn_map,
        },
        eos_coef=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=20,
        gamma=0.1,
    )

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        dataloader=train_loader,
        val_dataloader=val_loader,
    )

    return trainer


def main(model_config, resume_path=None, save_weights_folder="checkpoints"):
    save_dir = Path(save_weights_folder)
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = init_trainer(model_config)

    if resume_path is not None:
        trainer.resume(Path(resume_path))

    num_epochs = model_config["training_parameters"]["num_epochs"]
    start_epoch = trainer.epoch

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_stats, val_stats = trainer.step()

        print(
            f"Epoch [{epoch + 1}/{start_epoch + num_epochs}] | "
            f"train {format_stats(train_stats)} | "
            f"val {format_stats(val_stats)}"
        )

        trainer.save_checkpoint(save_dir / "last.pt")

        if trainer.last_was_best:
            trainer.save_checkpoint(save_dir / "best.pt")
            print(f"    New best model saved with val loss: {val_stats['loss']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--model", type=str, default="model.json")
    parser.add_argument("--save-weights", type=str, default="checkpoints")

    args = parser.parse_args()

    config_path = Path(args.model)

    with open(config_path, "r") as file:
        model_config = json.load(file)

    main(
        model_config,
        resume_path=args.resume,
        save_weights_folder=args.save_weights,
    )
