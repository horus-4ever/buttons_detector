from dataclasses import dataclass
import torch

from dataformat.dataset import DatasetConfig
from model.config import ModelConfig


@dataclass
class PTModel:
    epoch: int
    model_state_dict: dict
    optimizer_state_dict: dict
    scheduler_state_dict: dict
    losses: list
    evaluations: list
    model_configuration: ModelConfig
    dataset_configuration: DatasetConfig

    @classmethod
    def from_checkpoint(cls, checkpoint):
        return cls(
            epoch=checkpoint["epoch"],
            model_state_dict=checkpoint["model_state_dict"],
            optimizer_state_dict=checkpoint["optimizer_state_dict"],
            scheduler_state_dict=checkpoint["scheduler_state_dict"],
            losses=checkpoint["losses"],
            evaluations=checkpoint["evaluations"],
            model_configuration=checkpoint["model_configuration"],
            dataset_configuration=checkpoint["dataset_configuration"]
        )

    def to_json(self):
        return {
            "epoch": self.epoch,
            "model_state_dict": self.model_state_dict,
            "optimizer_state_dict": self.optimizer_state_dict,
            "scheduler_state_dict": self.scheduler_state_dict,
            "losses": self.losses,
            "evaluations": self.evaluations,
            "model_configuration": self.model_configuration.to_json(),
            "dataset_configuration": self.dataset_configuration.to_json()
        }

    