import json
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

import train as train_utils
from train import ButtonDataset, collate_fn, Trainer
from prtr import PRTR
from criterion import HungarianMatcher, SetCriterion
from transforms import *


class FineTuneTrainer(Trainer):
    def __init__(
        self,
        model,
        criterion,
        optimizer,
        scheduler,
        device,
        dataloader,
        val_dataloader,
        freeze_backbone_epochs: int = 0,
    ):
        super().__init__(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            dataloader=dataloader,
            val_dataloader=val_dataloader,
        )
        self.freeze_backbone_epochs = freeze_backbone_epochs
        self.backbone_frozen = False

    def set_backbone_trainable(self, trainable: bool):
        for p in self.model.backbone.parameters():
            p.requires_grad = trainable
        self.backbone_frozen = not trainable

    def on_epoch_start(self):
        should_freeze = self.epoch < self.freeze_backbone_epochs

        if should_freeze and not self.backbone_frozen:
            self.set_backbone_trainable(False)
            print(f"Epoch {self.epoch + 1}: backbone frozen")

        elif not should_freeze and self.backbone_frozen:
            self.set_backbone_trainable(True)
            print(f"Epoch {self.epoch + 1}: backbone unfrozen")

    def train_one_epoch(self):
        self.on_epoch_start()

        self.model.train()
        self.criterion.train()

        # If frozen, keep backbone in eval mode too
        if self.backbone_frozen:
            self.model.backbone.eval()

        running = {
            "loss": 0.0,
            "loss_ce": 0.0,
            "loss_button": 0.0,
        }

        for images, padding_mask, targets in self.dataloader:
            images = images.to(self.device)
            padding_mask = padding_mask.to(self.device)  # kept for consistency
            targets = [
                {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in t.items()}
                for t in targets
            ]

            outputs = self.model(images)
            losses = self.criterion(outputs, targets)

            self.optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            self.optimizer.step()

            for k in running:
                running[k] += losses[k].item()

        n = len(self.dataloader)
        return {k: v / n for k, v in running.items()}

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        self.criterion.eval()

        running = {
            "loss": 0.0,
            "loss_ce": 0.0,
            "loss_button": 0.0,
        }

        for images, padding_mask, targets in self.val_dataloader:
            images = images.to(self.device)
            padding_mask = padding_mask.to(self.device)  # kept for consistency
            targets = [
                {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in t.items()}
                for t in targets
            ]

            outputs = self.model(images)
            losses = self.criterion(outputs, targets)

            for k in running:
                running[k] += losses[k].item()

        n = len(self.val_dataloader)
        return {k: v / n for k, v in running.items()}

    def get_transforms(self) -> ComposeWithLabels:
        """
        Slightly lighter augmentation than pretraining.
        Usually better for fine-tuning on real images.
        """
        size = random.choice([640, 704, 768, 832, 896])

        return ComposeWithLabels([
            ComposeWrapper(transforms.Resize((size, size))),
            RandomHorizontalFlip(),
            RandomHorizontalTranslation(p=0.3, min=-0.15, max=0.15),
            RandomVerticalTranslation(p=0.3, min=-0.15, max=0.15),
            RandomRotation(p=0.3, min_angle=-15, max_angle=15),
            ComposeWrapper(transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02
            )),
            ComposeWrapper(transforms.ToTensor()),
            ComposeWrapper(transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )),
        ])


def build_optimizer(model, lr_backbone: float, lr_heads: float, weight_decay: float):
    backbone_params = []
    head_params = []

    for name, param in model.named_parameters():
        if name.startswith("backbone."):
            backbone_params.append(param)
        else:
            head_params.append(param)

    param_groups = []
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": lr_backbone})
    if head_params:
        param_groups.append({"params": head_params, "lr": lr_heads})

    optimizer = torch.optim.AdamW(
        param_groups,
        weight_decay=weight_decay,
    )
    return optimizer


def load_pretrained_weights(model, checkpoint_path: Path, device, strict: bool = True):
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    incompatible = model.load_state_dict(state_dict, strict=strict)

    if not strict:
        print("Loaded with strict=False")
        print("Missing keys:", incompatible.missing_keys)
        print("Unexpected keys:", incompatible.unexpected_keys)
    else:
        print(f"Loaded pretrained weights from: {checkpoint_path}")


_finetune_trainer = None


def init_finetune_trainer(model_config, dataset_root_override=None):
    global _finetune_trainer

    fine_cfg = model_config["fine_tuning_parameters"]
    model_parameters = model_config["parameters"]
    model_name = model_config["model_name"]

    dataset_root = dataset_root_override or fine_cfg["dataset_root"]
    batch_size = fine_cfg["batch_size"]
    lr_backbone = fine_cfg["lr_backbone"]
    lr_heads = fine_cfg["lr_heads"]
    weight_decay = fine_cfg["weight_decay"]
    train_split = fine_cfg["train_split"]
    num_workers = fine_cfg["num_workers"]
    seed = fine_cfg["seed"]
    cost_class = fine_cfg["cost_class"]
    cost_coord = fine_cfg["cost_coord"]
    freeze_backbone_epochs = fine_cfg.get("freeze_backbone_epochs", 0)
    scheduler_step_size = fine_cfg.get("scheduler_step_size", 10)
    scheduler_gamma = fine_cfg.get("scheduler_gamma", 0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    random.seed(seed)
    torch.manual_seed(seed)

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

    model = PRTR(model_name, **model_parameters).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    total_params_backbone = sum(p.numel() for p in model.backbone.parameters())
    print("# Number of parameters:", total_params)
    print("# Number of parameters of the backbone:", total_params_backbone)
    print("# Real parameters count:", total_params - total_params_backbone)

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

    optimizer = build_optimizer(
        model=model,
        lr_backbone=lr_backbone,
        lr_heads=lr_heads,
        weight_decay=weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size,
        gamma=scheduler_gamma,
    )

    _finetune_trainer = FineTuneTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        dataloader=train_loader,
        val_dataloader=val_loader,
        freeze_backbone_epochs=freeze_backbone_epochs,
    )

    # Important: ButtonDataset fetches transforms through trainer.get_trainer()
    # so we register this trainer there too.
    train_utils._trainer = _finetune_trainer

    return _finetune_trainer


def main(model_config, pretrained_path=None, resume_path=None, dataset_root_override=None):
    fine_cfg = model_config["fine_tuning_parameters"]
    num_epochs = fine_cfg["num_epochs"]
    save_dir = Path(fine_cfg.get("save_dir", "checkpoints_finetune"))
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = init_finetune_trainer(
        model_config=model_config,
        dataset_root_override=dataset_root_override,
    )

    if resume_path is not None:
        trainer.resume(Path(resume_path))
        print(f"Resumed fine-tuning from: {resume_path}")
    else:
        cfg_pretrained = fine_cfg.get("pretrained_checkpoint", None)
        final_pretrained = pretrained_path or cfg_pretrained
        if final_pretrained is None:
            raise RuntimeError(
                "No pretrained checkpoint provided. "
                "Set fine_tuning_parameters.pretrained_checkpoint in model.json "
                "or pass --pretrained."
            )

        strict_loading = fine_cfg.get("strict_model_loading", True)
        load_pretrained_weights(
            trainer.model,
            Path(final_pretrained),
            trainer.device,
            strict=strict_loading,
        )

    start_epoch = trainer.epoch

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_stats, val_stats = trainer.step()

        lrs = [group["lr"] for group in trainer.optimizer.param_groups]
        lr_backbone = lrs[0] if len(lrs) > 0 else 0.0
        lr_heads = lrs[1] if len(lrs) > 1 else lr_backbone

        print(
            f"Epoch [{epoch + 1}/{start_epoch + num_epochs}] | "
            f"train loss: {train_stats['loss']:.4f} "
            f"(ce={train_stats['loss_ce']:.4f}, btn={train_stats['loss_button']:.4f}) | "
            f"val loss: {val_stats['loss']:.4f} "
            f"(ce={val_stats['loss_ce']:.4f}, btn={val_stats['loss_button']:.4f}) | "
            f"lr_backbone={lr_backbone:.2e} lr_heads={lr_heads:.2e}"
        )

        trainer.save_checkpoint(save_dir / "last.pt")
        if trainer.last_was_best:
            trainer.save_checkpoint(save_dir / "best.pt")
            print(f"    New best fine-tuned model saved with val loss: {val_stats['loss']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="model.json")
    parser.add_argument("--pretrained", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    args = parser.parse_args()

    config_path = Path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    main(
        model_config=model_config,
        pretrained_path=args.pretrained,
        resume_path=args.resume,
        dataset_root_override=args.dataset,
    )