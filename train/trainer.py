from pathlib import Path
import time
from typing import Dict

import torch

from model.config import ModelConfig
from model.mode import Mode
from model.utils import load_weights, freeze_backbone
from dataformat.dataset import DatasetConfig, Annotation
from evaluate import Evaluator
from train import runner


class Elapsed:
    def __init__(self):
        self.start_time = 0
        self.elapsed = 0

    @property
    def current(self):
        return time.time_ns() - self.start_time

    def __enter__(self, *args):
        self.start_time = time.time_ns()
        return self

    def __exit__(self, *args):
        self.elapsed = time.time_ns() - self.start_time


class Trainer:
    """
    Trainer class is responsible for the training of the models.
    Models are evaluated at each step on:
    - the validation dataset --> provides the target signal for best model selection
    - the test dataset --> provides data for subsequent analysis of generalisation capabilities
    The following informations are saved in each .pt file:
    - model weights, optimizer and scheduler states
    - losses over all training epochs
    - evaluations over all training epochs
    The evaluations are stored in the following format:
    [{ 
        "validation": {
            "distance": {
                "<group-name>": {
                    "<metric-name>": <value>
                }
            },
            "iou": {
                ...
            }
        }
        "test": {
            ...
        }
    }, ...]
    """
    def __init__(
            self,
            model,
            criterion,
            optimizer,
            scheduler,
            device,
            dataloader,
            val_dataloader,
            test_dataloader,
            model_config: ModelConfig,
            dataset_config: DatasetConfig,
            evaluator: Evaluator
        ):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader
        self.dataloader = dataloader
        self.model_config = model_config
        self.dataset_config = dataset_config
        self.epoch = 0
        self.losses = [] # save losses over epochs
        self.evaluations = [] # save evaluations over epochs
        self.best_val_loss = float("inf")
        self.last_was_best = False
        self.evaluator = evaluator

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
        freeze_backbone(self.model) # freeze the backbone
        # store the accumulated losses
        running: Dict[str, float] = {}
        # epoch main loop
        n_images = len(self.dataloader)
        start_time = time.time_ns()
        for i, (images, padding_mask, annotations, (W, H)) in enumerate(self.dataloader):
            print(f"\r\033[2K  - TRAIN: [{i + 1:<4} / {n_images}]", end="", flush=True)
            outputs, targets = runner.run_one(self.model, images, padding_mask, annotations, self.device)
            losses = self.criterion(outputs, targets)
            self.optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward() # backward propagation on the global loss
            self.optimizer.step()
            # accumulate the losses stored in the `losses` dict
            self._accumulate_losses(running, losses)
        elapsed = (time.time_ns() - start_time) / 1e9
        print(f" {elapsed:.4f}")
        # compute and return the mean of the losses over all batches
        nb_batches = len(self.dataloader)
        mean_losses = self._mean_losses(running, nb_batches)
        return mean_losses

    @torch.no_grad()
    def evaluate(self):
        """
        Computes the current checkpoint's loss on the validation dataset.
        """
        self.model.eval()
        self.criterion.eval()
        running: Dict[str, float] = {}
        # evaluation main loop
        for images, padding_mask, annotations, (W, H) in self.val_dataloader:
            outputs, targets = runner.run_one(self.model, images, padding_mask, annotations, self.device)
            losses = self.criterion(outputs, targets)
            self._accumulate_losses(running, losses)
        nb_batches = len(self.val_dataloader)
        mean_losses = self._mean_losses(running, nb_batches)
        return mean_losses

    @torch.inference_mode()
    def _run_evaluation(self, dataloader, what=""):
        self.model.eval()
        outputs = []
        n_images = len(dataloader)
        start_time = time.time_ns()
        for i, (images, padding_mask, annotations, (W, H)) in enumerate(dataloader):
            print(f"\r\033[2K  - EVALUATE ({what}): [{i + 1:<4} / {n_images}]", end="", flush=True)
            annotations = annotations[0] # batch size is 1 for evaluation
            images = images.to(self.device, non_blocking=True)
            padding_mask = padding_mask.to(self.device, non_blocking=True)
            output = self.model(images, padding_mask, mode=Mode.EVALUATION)
            outputs.append((annotations, output))
        elapsed = (time.time_ns() - start_time) / 1e9
        print(f" ({elapsed:.4f}s)")
        return self.evaluator.evaluate(outputs)

    @torch.inference_mode()
    def run_evaluations(self):
        """
        Run evaluations on the validation and test datasets.
        """
        validation_results = self._run_evaluation(self.val_dataloader, what="validation")
        test_results = self._run_evaluation(self.test_dataloader, what="test")
        return validation_results, test_results

    def step(self):
        """
        Perform one training epoch and its evaluation pass.
        """
        train_losses = self.train_one_epoch()
        val_losses = self.evaluate()
        val_evaluation, test_evaluation = self.run_evaluations()
        # save the losses
        self.losses.append({
            "train": train_losses,
            "validation": val_losses
        })
        self.evaluations.append({
            "validation": val_evaluation,
            "test": test_evaluation
        })
        self.scheduler.step()
        # save the best loss each time
        if val_losses["loss"] < self.best_val_loss:
            self.best_val_loss = val_losses["loss"]
            self.last_was_best = True
        else:
            self.last_was_best = False
        self.epoch += 1
        return train_losses, val_losses, val_evaluation, test_evaluation

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
        self.evaluations = checkpoint.get("evaluations", [])

    def save_checkpoint(self, save_path: Path):
        torch.save(
            {
                "epoch": self.epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "losses": self.losses,
                "evaluations": self.evaluations,
                "model_configuration": self.model_config.to_json(),
                "dataset_configuration": self.dataset_config.to_json()
            },
            save_path,
        )
