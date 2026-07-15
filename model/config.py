import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelParameters:
    num_queries: int
    d_model: int
    n_heads: int
    n_encoder_layers: int
    n_decoder_layers: int
    n_points: int
    dim_ffn: int
    dropout: float
    activation: str
    mlp_hidden_dim: int
    mlp_num_layers: int

    @classmethod
    def from_json(cls, data):
        return cls(
            num_queries=data["num_queries"],
            d_model=data["d_model"],
            n_heads=data["n_heads"],
            n_encoder_layers=data["n_encoder_layers"],
            n_decoder_layers=data["n_decoder_layers"],
            n_points=data["n_points"],
            dim_ffn=data["dim_ffn"],
            dropout=data["dropout"],
            activation=data["activation"],
            mlp_hidden_dim=data["mlp_hidden_dim"],
            mlp_num_layers=data["mlp_num_layers"]
        )


@dataclass
class TrainingParameters:
    dataset: Path # path to the json file
    batch_size: int
    num_epochs: int
    lr: float
    weight_decay: float
    num_workers: int
    seed: int
    cost_class: float
    cost_coord: float
    cost_giou: float

    @classmethod
    def from_json(cls, data):
        return cls(
            dataset=Path(data["dataset"]),
            batch_size=data["batch_size"],
            num_epochs=data["num_epochs"],
            lr=data["lr"],
            weight_decay=data["weight_decay"],
            num_workers=data["num_workers"],
            seed=data["seed"],
            cost_class=data["cost_class"],
            cost_coord=data["cost_coord"],
            cost_giou=data["cost_giou"]
        )


@dataclass
class FinetuneParameters(TrainingParameters):
    weights: Path

    @classmethod
    def from_json(cls, data):
        object = TrainingParameters.from_json(data)
        object.weights = Path(data["weights"])
        return object


@dataclass
class ModelConfig:
    name: str
    model_parameters: ModelParameters
    training_parameters: TrainingParameters
    finetune_parameters: FinetuneParameters

    @classmethod
    def from_json(cls, data):
        return cls(
            name=data["name"],
            model_parameters=ModelParameters.from_json(data["model_parameters"]),
            training_parameters=TrainingParameters.from_json(data["training_parameters"]),
            finetune_parameters=FinetuneParameters.from_json(data["finetune_parameters"])
        )

    @classmethod
    def open(cls, path: Path):
        with open(path, "r") as file:
            json_data = json.load(file)
            return cls.from_json(json_data)