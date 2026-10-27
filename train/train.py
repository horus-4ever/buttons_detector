import random
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from model.prtr import build_model
from model.criterion import HungarianMatcher, SetCriterion
from model.config import ModelConfig
from model.utils import load_weights, freeze_backbone
from dataformat.dataset import DatasetConfig
from .transforms import TrainingTransform, ValidationTransform
from .commons import collate_fn
from .evaluator import get_evaluator
from .trainer import Trainer


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def format_stats(stats: Dict[str, float]):
    """
    Format the loss statistics to display to the user.
    """
    parts = [
        f"loss={stats.get('loss', 0.0):.4f}",
        f"ce={stats.get('ce_loss', 0.0):.4f}",
        f"pair_L1={stats.get('L1_loss_pair', 0.0):.4f}",
        f"pair_IoU={stats.get('GIoU_loss_pair', 0.0):.4f}"
    ]
    return "(" + ", ".join(parts) + ")"


def init_trainer(model_config: ModelConfig):
    parameters = model_config.training_parameters
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
    image_size = parameters.img_size
    train_dataset.transform = TrainingTransform(model_config.augmentations, image_size)
    val_dataset.transform = ValidationTransform(image_size)
    test_dataset.transform = ValidationTransform(image_size)

    # create the data loaders
    # all the loaders are initialized with the same random seed
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
        batch_size=1,
        shuffle=False,
        num_workers=parameters.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
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
    if parameters.finetune is not None:
        load_weights(model, parameters.finetune, device)
        freeze_backbone(model)

    # display statistics over the number of parameters
    total_params = sum(p.numel() for p in model.parameters())
    total_params_backbone = sum(p.numel() for p in model.backbone.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("# Number of parameters:", total_params)
    print("# Number of parameters of the backbone:", total_params_backbone)
    print("# Trainable parameters:", trainable_params)
    # display statistics over the number of images in the dataset
    train_dataset_length = len(train_dataset)
    validation_dataset_length = len(val_dataset)
    test_dataset_length = len(test_dataset)
    total_dataset_length = train_dataset_length + validation_dataset_length + test_dataset_length
    train_ratio = train_dataset_length / total_dataset_length
    val_ratio = validation_dataset_length / total_dataset_length
    test_ratio = test_dataset_length / total_dataset_length
    print(f"$ train={train_dataset_length}, validation={validation_dataset_length}, test={test_dataset_length}")
    print(f"$   --> {train_ratio:.2f}/{val_ratio:.2f}/{test_ratio:.2f}")

    matcher = HungarianMatcher(
        cost_class=parameters.cost_class,
        cost_coord=parameters.cost_coord,
        cost_giou=parameters.cost_giou
    )

    criterion = SetCriterion(
        num_classes=1,
        matcher=matcher,
        weight_dict={
            "cost_ce": parameters.cost_class,
            "cost_l1": parameters.cost_coord,
            "cost_giou": parameters.cost_giou
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

    evaluator = get_evaluator(threshold=0.9)

    trainer = Trainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        dataloader=train_loader,
        val_dataloader=val_loader,
        test_dataloader=test_loader,
        model_config=model_config,
        dataset_config=dataset_config,
        evaluator=evaluator
    )
    return trainer


def train(model_config: ModelConfig, resume_path=None, save_weights_folder="checkpoints"):
    save_dir = Path(save_weights_folder)
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer = init_trainer(model_config)

    if resume_path is not None:
        trainer.resume(Path(resume_path))

    num_epochs = model_config.training_parameters.num_epochs
    start_epoch = trainer.epoch
    # training loop
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # step one epoch of the trainer
        train_stats, val_stats, val_evaluation, test_evaluation = trainer.step()
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
