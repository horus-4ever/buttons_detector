import json
import random
from pathlib import Path
from typing import Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset

from model.prtr import PRTR
from model.criterion import HungarianMatcher, SetCriterion

# Assumes your existing script is saved as train.py
from train.train import (
    ButtonDataset,
    RandomTrainTransform,
    make_val_transform,
    collate_fn,
    seed_worker,
    Trainer,
    format_stats,
)

class FinetuneDataset(Dataset):
    def __init__(self, dataset, transform = None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        image, target = self.dataset[idx]
        if self.transform is None:
            return image, target
        coords = target["buttons"].tolist()
        image_id = target["image_id"]
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
            "image_id": image_id,
            "size": torch.tensor([out_h, out_w], dtype=torch.int64),
        }

        return image, target


def extract_model_state_dict(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """
    Supports checkpoints saved as:
      - {"model_state_dict": ...}
      - {"state_dict": ...}
      - {"model": ...}
      - raw model.state_dict()
    """
    if isinstance(checkpoint, dict):
        for key in ("model_state_dict", "state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                return checkpoint[key]

        if all(torch.is_tensor(v) for v in checkpoint.values()):
            return checkpoint

    raise RuntimeError(
        "Could not find model weights in checkpoint. Expected one of: "
        "'model_state_dict', 'state_dict', 'model', or a raw state_dict."
    )


def strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Handles weights saved from DataParallel / DistributedDataParallel,
    where keys may start with 'module.'.
    """
    cleaned = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module."):]
        cleaned[key] = value

    return cleaned


def load_pretrained_weights(
    model: torch.nn.Module,
    pretrained_path: Path,
    device: torch.device,
    strict: bool = True,
):
    if not pretrained_path.exists():
        raise FileNotFoundError(f"Pretrained weights not found: {pretrained_path}")

    checkpoint = torch.load(pretrained_path, map_location=device)
    pretrained_state = strip_module_prefix(extract_model_state_dict(checkpoint))

    if strict:
        model.load_state_dict(pretrained_state, strict=True)
        print(f"Loaded pretrained weights strictly from: {pretrained_path}")
        return

    model_state = model.state_dict()

    compatible_state = {}
    skipped = []

    for key, value in pretrained_state.items():
        if key not in model_state:
            skipped.append((key, "missing in current model"))
            continue

        if model_state[key].shape != value.shape:
            skipped.append(
                (
                    key,
                    f"shape mismatch: checkpoint {tuple(value.shape)} "
                    f"vs model {tuple(model_state[key].shape)}",
                )
            )
            continue

        compatible_state[key] = value

    missing, unexpected = model.load_state_dict(compatible_state, strict=False)

    print(f"Loaded pretrained weights non-strictly from: {pretrained_path}")
    print(f"Compatible tensors loaded: {len(compatible_state)}")
    print(f"Skipped tensors: {len(skipped)}")

    if skipped:
        print("Skipped checkpoint tensors:")
        for key, reason in skipped[:30]:
            print(f"  - {key}: {reason}")
        if len(skipped) > 30:
            print(f"  ... and {len(skipped) - 30} more")

    if missing:
        print(f"Missing model tensors not loaded: {len(missing)}")

    if unexpected:
        print(f"Unexpected checkpoint tensors: {len(unexpected)}")


def freeze_backbone(model):
    for param in model.backbone.parameters():
        param.requires_grad = False
    model.backbone.body.eval()


def build_finetune_parameters(model_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uses training_parameters as defaults, then overlays optional
    finetuning_parameters.

    This lets your config stay backward-compatible.
    """
    if "training_parameters" not in model_config:
        raise KeyError("Config must contain 'training_parameters'.")

    training_parameters = dict(model_config["training_parameters"])
    finetuning_parameters = model_config.get("finetuning_parameters", {})

    training_parameters.update(finetuning_parameters)

    return training_parameters


def init_finetuner(
    model_config: Dict[str, Any],
    pretrained_path: Path,
    strict_pretrained: bool = True,
):
    """
    The finetuner takes the parameters from a json file.
    When finetuning, there is a possibility to choose to mix data with another dataset.
    In this case, we indicate the percentage of mix into the dataset.
    WARNING: there must be enough data in the other dataset.
    """
    training_parameters = model_config["finetuning_parameters"]
    model_parameters = model_config["parameters"]
    model_name = model_config["model_name"]

    # get the training parameters
    dataset_root = training_parameters["dataset"]
    mix_dataset_root = training_parameters["mix_dataset"]
    mix_ratio = training_parameters["mix_ratio"]
    batch_size = training_parameters["batch_size"]
    lr = training_parameters["lr"]
    weight_decay = training_parameters["weight_decay"]
    train_split = training_parameters["train_split"]
    num_workers = training_parameters["num_workers"]
    seed = training_parameters["seed"]
    # get the costs of each class
    cost_class = training_parameters["cost_class"]
    cost_coord = training_parameters["cost_coord"]
    # indicate the size to which the images are resized
    validation_image_size = training_parameters.get("val_size", 512)
    training_image_sizes = training_parameters.get("train_sizes", [512])

    scheduler_step_size = training_parameters.get("scheduler_step_size", 20)
    scheduler_gamma = training_parameters.get("scheduler_gamma", 0.1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    random.seed(seed)
    torch.manual_seed(seed)

    base_dataset = ButtonDataset(dataset_root, transform=None)
    mix_dataset = ButtonDataset(mix_dataset_root, transform=None)
    # we first check if there is enough data
    # first, we get the number of data for training and validation from the base dataset
    mix_ratio = float(mix_ratio) / 100
    n_total = int(len(base_dataset) / (1 - mix_ratio))
    n_mix_needed = int(mix_ratio * n_total)

    if n_mix_needed > len(mix_dataset):
        raise ValueError(f"The mix dataset must be large enough (need {n_mix_needed}, but there is only {len(base_dataset)}).")
    # now, let's generate the dataset
    split_generator = torch.Generator().manual_seed(seed)
    mix_dataset_permutations = torch.randperm(len(mix_dataset), generator=split_generator).tolist()
    # sample `n_mix_needed` from the mix dataset
    mix_dataset_indices = mix_dataset_permutations[:n_mix_needed]
    mix_subdataset = Subset(mix_dataset, mix_dataset_indices)

    # now combine the two datasets together
    final_dataset = ConcatDataset([base_dataset, mix_subdataset])
    # now, split this dataset into training and validation
    assert n_total == len(final_dataset)
    n_total = len(final_dataset)
    n_train = int(n_total * float(train_split))
    n_valid = n_total - n_train
    # get the indices, then split
    indices = torch.randperm(n_total, generator=split_generator).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    # generate the two datasets
    train_dataset = FinetuneDataset(
        Subset(final_dataset, train_indices),
        transform=RandomTrainTransform(training_image_sizes)
    )
    val_dataset = FinetuneDataset(
        Subset(final_dataset, val_indices),
        transform=make_val_transform(size=validation_image_size)
    )

    print(f"~> finetune on {len(final_dataset)} (mix: {mix_ratio})")

    # now create the data loaders
    # first define the random generator for the dataloader
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    # then define the two data loaders
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
    # construct the model and put it to the GPU
    model = PRTR(model_name, **model_parameters)
    model = model.to(device)

    load_pretrained_weights(
        model=model,
        pretrained_path=pretrained_path,
        device=device,
        strict=strict_pretrained,
    )

    freeze_backbone(model)

    total_params = sum(p.numel() for p in model.parameters())
    total_params_backbone = sum(p.numel() for p in model.backbone.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("# Number of parameters:", total_params)
    print("# Number of parameters of the backbone:", total_params_backbone)
    print("# Trainable parameters:", trainable_params)

    if trainable_params == 0:
        raise RuntimeError("No trainable parameters found. Check freeze settings.")

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
            "loss_attn": 0,
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
        step_size=scheduler_step_size,
        gamma=scheduler_gamma,
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


def main(
    model_config: Dict[str, Any],
    pretrained_path: str,
    resume_path: str = None,
    save_weights_folder: str = "finetune_checkpoints",
    strict_pretrained: bool = True,
):
    save_dir = Path(save_weights_folder)
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = init_finetuner(
        model_config=model_config,
        pretrained_path=Path(pretrained_path),
        strict_pretrained=strict_pretrained,
    )

    # Optional: resume a fine-tuning run, including optimizer/scheduler state.
    # This is different from loading pretrained weights.
    if resume_path is not None:
        trainer.resume(Path(resume_path))

    training_parameters = build_finetune_parameters(model_config)
    num_epochs = training_parameters["num_epochs"]

    start_epoch = trainer.epoch

    for epoch in range(start_epoch, start_epoch + num_epochs):
        train_stats, val_stats = trainer.step()

        print(
            f"Finetune Epoch [{epoch + 1}/{start_epoch + num_epochs}] | "
            f"train {format_stats(train_stats)} | "
            f"val {format_stats(val_stats)}"
        )

        trainer.save_checkpoint(save_dir / "last.pt")

        if trainer.last_was_best:
            trainer.save_checkpoint(save_dir / "best.pt")
            print(f"    New best fine-tuned model saved with val loss: {val_stats['loss']:.4f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to JSON configuration file.",
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        required=True,
        help="Path to pretrained weights/checkpoint, for example checkpoints/best.pt.",
    )

    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Optional path to a fine-tuning checkpoint to resume from.",
    )

    parser.add_argument(
        "--save-weights",
        type=str,
        default="finetune_checkpoints",
        help="Folder where fine-tuning checkpoints will be saved.",
    )

    parser.add_argument(
        "--non-strict",
        action="store_true",
        help=(
            "Load only compatible pretrained tensors. Useful if the fine-tuning "
            "model has a changed head, changed number of classes, or changed query count."
        ),
    )

    args = parser.parse_args()

    config_path = Path(args.model)

    with open(config_path, "r", encoding="utf-8") as file:
        model_config = json.load(file)

    main(
        model_config=model_config,
        pretrained_path=args.pretrained,
        resume_path=args.resume,
        save_weights_folder=args.save_weights,
        strict_pretrained=not args.non_strict,
    )