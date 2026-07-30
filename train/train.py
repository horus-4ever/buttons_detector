import json
import random
from pathlib import Path
from typing import Optional, Callable, Dict, Any

import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms as T

from model.prtr import PRTR, build_model
from model.criterion import HungarianMatcher, SetCriterion
from model.config import ModelConfig
from dataformat.dataset import DatasetConfig, Annotation
from train.transforms import TrainingTransform, ValidationTransform


def collate_fn(batch):
    """
    Padds and computes each image mask for the given batch.
    """
    images, targets = zip(*batch)
    # get the maximum image size of the batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    # targets: list of [Annotation]

    batch_size = len(images)
    channels = images[0].shape[0]
    dtype = images[0].dtype
    # initialize the tensors that will contain the batch
    # the mask is: 0 means image data ; 1 means padded data (to be ignored)
    # [B, C, max H, max W]
    padded_images = torch.zeros((batch_size, channels, max_h, max_w), dtype=dtype)
    padding_mask = torch.ones((batch_size, max_h, max_w), dtype=torch.bool)

    new_targets = []
    for i, (image, target) in enumerate(zip(images, targets)):
        _, h, w = image.shape
        padded_images[i, :, :h, :w] = image
        padding_mask[i, :h, :w] = False
        new_targets.append(target)
    # take the new common size
    common_size = (max_w, max_h)
    return padded_images, padding_mask, new_targets, common_size


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

    def _accumulate_losses(self, running: Dict[str, float], losses: Dict[str, torch.Tensor]):
        for k, v in losses.items():
            running[k] = running.get(k, 0.0) + float(v.item())

    def _mean_losses(self, running: Dict[str, float], n: int):
        if n == 0:
            raise RuntimeError("Dataloader produced zero batches.")
        return {k: v / n for k, v in running.items()}
    
    def _annotations_to_tensor(self, annotations: list[Annotation], device):
        targets = []
        for annotation in annotations:
            coord_buttons, coord_fasteners = annotation.to_tensor()
            coord_buttons = coord_buttons.to(device=device)
            coord_fasteners = coord_fasteners.to(device=device)
            labels = torch.zeros(coord_buttons.size()[0], dtype=torch.long, device=device)
            targets.append({
                "labels": labels,
                "buttons": coord_buttons,
                "keypoints": coord_fasteners
            })
        return targets

    def train_one_epoch(self):
        self.model.train()
        self.criterion.train()
        self.model.backbone.body.eval() # freeze the backbone

        running: Dict[str, float] = {}

        for images, padding_mask, annotations, (W, H) in self.dataloader:
            # annotations: b * Annotation
            images = images.to(self.device, non_blocking=True)
            padding_mask = padding_mask.to(self.device, non_blocking=True)
            # now transform the targets into tensors
            targets = self._annotations_to_tensor(annotations, device=self.device)

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

        for images, padding_mask, annotations, (W, H) in self.val_dataloader:
            images = images.to(self.device, non_blocking=True)
            padding_mask = padding_mask.to(self.device, non_blocking=True)
            # now transform the targets into tensors
            targets = self._annotations_to_tensor(annotations, device=self.device)

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


def load_weights(model, weights: Path, device):
    if not weights.exists():
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    checkpoint = torch.load(weights, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])


def freeze_backbone(model):
    for p in model.backbone.body.parameters():
        p.requires_grad = False
    model.backbone.eval()

def init_trainer(model_config: ModelConfig, finetune: bool):
    parameters = model_config.training_parameters if not finetune else model_config.finetune_parameters
    model_params = model_config.model_parameters
    # get the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    # set the random seed
    random.seed(parameters.seed)
    torch.manual_seed(parameters.seed)
    # get the dataset
    dataset_config = DatasetConfig.open(parameters.dataset)
    dataset_config.load() # builds cache
    train_dataset, val_dataset, test_dataset = dataset_config.to_torch_dataset()
    train_dataset.transform = TrainingTransform()
    val_dataset.transform = ValidationTransform(512)

    # create the data loaders
    loader_generator = torch.Generator()
    loader_generator.manual_seed(parameters.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=parameters.batch_size,
        shuffle=True,
        num_workers=parameters.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
        generator=loader_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=parameters.batch_size,
        shuffle=False,
        num_workers=parameters.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
    )
    # build the model
    model = build_model(model_config)
    model = model.to(device)
    print(f"# model built and loaded to '{device}'")
    # if finetuning, then load the weights
    if finetune:
        load_weights(model, parameters.weights, device)
        freeze_backbone(model)

    total_params = sum(p.numel() for p in model.parameters())
    total_params_backbone = sum(p.numel() for p in model.backbone.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("# Number of parameters:", total_params)
    print("# Number of parameters of the backbone:", total_params_backbone)
    print("# Trainable parameters:", trainable_params)

    matcher = HungarianMatcher(
        cost_class=parameters.cost_class,
        cost_coord=parameters.cost_coord,
    )

    criterion = SetCriterion(
        num_classes=1,
        matcher=matcher,
        weight_dict={
            "loss_ce": parameters.cost_class,
            "loss_button": parameters.cost_coord,
            "loss_giou": parameters.cost_giou
        },
        eos_coef=0.1,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=parameters.lr,
        weight_decay=parameters.weight_decay,
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


def train(model_config: ModelConfig, finetune=False, resume_path=None, save_weights_folder="checkpoints"):
    save_dir = Path(save_weights_folder)
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = init_trainer(model_config, finetune=finetune)

    if resume_path is not None:
        trainer.resume(Path(resume_path))

    num_epochs = model_config.training_parameters.num_epochs
    start_epoch = trainer.epoch
    # training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # step one epoch of the trainer
        train_stats, val_stats = trainer.step()
        # display the epoch results
        print(
            f"Epoch [{epoch + 1}/{start_epoch + num_epochs}] | "
            f"train {format_stats(train_stats)} | "
            f"val {format_stats(val_stats)}"
        )
        # save the last model, and the best one
        trainer.save_checkpoint(save_dir / "last.pt")
        if trainer.last_was_best:
            trainer.save_checkpoint(save_dir / "best.pt")
            print(f"    New best model saved with val loss: {val_stats['loss']:.4f}")
