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
    branch_infos: dict[str, str]

    @property
    def parameters_count(self):
        n_params = sum(tensor.numel() for tensor in self.model_state_dict.values())
        return n_params

    @classmethod
    def from_checkpoint(cls, checkpoint):
        return cls(
            epoch=checkpoint["epoch"],
            model_state_dict=checkpoint["model_state_dict"],
            optimizer_state_dict=checkpoint["optimizer_state_dict"],
            scheduler_state_dict=checkpoint["scheduler_state_dict"],
            losses=checkpoint["losses"],
            evaluations=checkpoint["evaluations"],
            model_configuration=ModelConfig.from_json(checkpoint["model_configuration"]),
            dataset_configuration=DatasetConfig.from_json(checkpoint["dataset_configuration"]),
            branch_infos=checkpoint["branch_infos"]
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
            "dataset_configuration": self.dataset_configuration.to_json(),
            "branch_infos": self.branch_infos
        }
