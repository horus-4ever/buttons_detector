from pathlib import Path
from typing import Dict

import torch

from model.config import ModelConfig
from model.utils import load_weights, freeze_backbone
from dataformat.dataset import DatasetConfig, Annotation
from evaluate import Evaluator
from train import runner


class Trainer:
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
        for images, padding_mask, annotations, (W, H) in self.dataloader:
            outputs, targets = runner.run_one(self.model, images, padding_mask, annotations, self.device)
            losses = self.criterion(outputs, targets)
            self.optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward() # backward propagation on the global loss
            self.optimizer.step()
            # accumulate the losses stored in the `losses` dict
            self._accumulate_losses(running, losses)
        # compute and return the mean of the losses over all batches
        nb_batches = len(self.dataloader)
        mean_losses = self._mean_losses(running, nb_batches)
        return mean_losses

    @torch.no_grad()
    def evaluate(self):
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

    @torch.no_grad()
    def _run_evaluation(self, dataloader):
        self.model.eval()
        outputs = []
        for images, padding_mask, annotations, (W, H) in dataloader:
            annotations = annotations[0] # batch size is 1 for evaluation
            images = images.to(self.device, non_blocking=True)
            padding_mask = padding_mask.to(self.device, non_blocking=True)
            output = self.model(images, padding_mask)
            outputs.append((annotations, output))
        return self.evaluator.evaluate(outputs)

    @torch.no_grad()
    def run_evaluations(self):
        validation_results = self._run_evaluation(self.val_dataloader)
        test_results = self._run_evaluation(self.test_dataloader)
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
            "train_losses": train_losses,
            "val_losses": val_losses
        })
        self.evaluations.append({
            "val_evaluation": val_evaluation,
            "test_evaluation": test_evaluation
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
