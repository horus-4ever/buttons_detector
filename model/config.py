import json
from dataclasses import dataclass
from pathlib import Path
from augmentations import get_transforms, serialize_transforms, Transform


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

    def to_json(self):
        return {
            "num_queries": self.num_queries,
            "d_model": self.d_model,
            "n_heads": self.n_heads,
            "n_encoder_layers": self.n_encoder_layers,
            "n_decoder_layers": self.n_decoder_layers,
            "n_points": self.n_points,
            "dim_ffn": self.dim_ffn,
            "dropout": self.dropout,
            "activation": self.activation,
            "mlp_hidden_dim": self.mlp_hidden_dim,
            "mlp_num_layers": self.mlp_num_layers
        }

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
    finetune: Path | None # path to the base model or None
    batch_size: int
    num_epochs: int
    lr: float
    weight_decay: float
    num_workers: int
    seed: int
    cost_class: float
    cost_coord: float
    cost_giou: float
    img_size: int

    def to_json(self):
        return {
            "dataset": self.dataset,
            "finetune": self.finetune,
            "batch_size": self.batch_size,
            "num_epochs": self.num_epochs,
            "lr": self.lr,
            "weight_decay": self.weight_decay,
            "num_workers": self.num_workers,
            "seed": self.seed,
            "cost_class": self.cost_class,
            "cost_l1": self.cost_coord,
            "cost_giou": self.cost_giou,
            "img_size": self.img_size
        }

    @classmethod
    def from_json(cls, data):
        return cls(
            dataset=Path(data["dataset"]),
            finetune=Path(data["finetune"]) if data["finetune"] is not None else None,
            batch_size=data["batch_size"],
            num_epochs=data["num_epochs"],
            lr=data["lr"],
            weight_decay=data["weight_decay"],
            num_workers=data["num_workers"],
            seed=data["seed"],
            cost_class=data["cost_class"],
            cost_coord=data["cost_l1"],
            cost_giou=data["cost_giou"],
            img_size=data["img_size"]
        )



@dataclass
class ModelConfig:
    name: str
    model_parameters: ModelParameters
    training_parameters: TrainingParameters
    augmentations: list[Transform]

    def to_json(self):
        return {
            "name": self.name,
            "model_parameters": self.model_parameters.to_json(),
            "training": self.training_parameters.to_json(),
            "augmentations": serialize_transforms(self.augmentations)
        }

    @classmethod
    def from_json(cls, data):
        return cls(
            name=data["name"],
            model_parameters=ModelParameters.from_json(data["model_parameters"]),
            training_parameters=TrainingParameters.from_json(data["training"]),
            augmentations=get_transforms(data["augmentations"])
        )

    @classmethod
    def open(cls, path: Path):
        with open(path, "r") as file:
            json_data = json.load(file)
            return cls.from_json(json_data)
