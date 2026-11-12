from dataclasses import dataclass
from pathlib import Path
import git
from model.config import ModelConfig
from dataformat.dataset import DatasetConfig
import json
from argparse import ArgumentParser


REPO = git.Repo(".")


def try_load_model_config(path: Path):
    with open(path, "r") as file:
        json_data = json.load(file)
        try:
            model_config = ModelConfig.from_json(json_data)
            return None, model_config
        except Exception as e:
            return e, None


def try_load_dataset_config(path: Path):
    try:
        config = DatasetConfig.open(path)
        return None, config
    except Exception as e:
        return e, None


@dataclass
class Experiment:
    name: str
    branch: str

    @classmethod
    def from_json(cls, json_data: dict):
        return cls(
            name=json_data["name"],
            branch=json_data["branch"]
        )

    def check(self, models: Path, datasets: Path, weights: Path):
        # check if the model configuration exists
        model_config_path = (models / f"{self.name}.config.json")
        if not model_config_path.exists():
            print(f"File '{model_config_path}' doesn't exist.") ; exit(-1)
        # check if the model's configuration is valid
        err, model_config = try_load_model_config(model_config_path)
        if model_config is None:
            print(f"Invalid file format: {err}") ; exit(-1)
        # make some assumptions over the model configuration
        if model_config.training_parameters.img_size != 512:
            print(f"Image size must be of 512x512.") ; exit(-1)
        # make the assumption that the model name must match the experiment name
        if model_config.name != self.name:
            print(f"Model configuration's name ({model_config.name}) doesn't match experiment name ({self.name}).") ; exit(0)
        # check if the dataset configuration exists
        dataset_config_path = datasets / model_config.training_parameters.dataset
        if not dataset_config_path.exists():
            print(f"Dataset configuration '{dataset_config_path}' doesn't exist.") ; exit(-1)
        # now, try to load the dataset configuration
        err, dataset_config = try_load_dataset_config(dataset_config_path)
        if dataset_config is None:
            print(f"Error while loading the dataset configuration: {err}") ; exit(0)
        # else, everything was fine
        print("All is good, ready for training.")


@dataclass
class Everything:
    models: Path
    datasets: Path
    weights: Path
    experiments: list[Experiment]
    reports: Path

    @classmethod
    def from_json(cls, json_data: dict):
        return cls(
            models=Path(json_data["models"]),
            datasets=Path(json_data["datasets"]),
            weights=Path(json_data["weights"]),
            experiments=[Experiment.from_json(experiment) for experiment in json_data["runs"]],
            reports=Path(json_data["reports"])
        )

    def check(self):
        for experiment in self.experiments:
            experiment.check(self.models, self.datasets, self.weights)


def init_parser():
    parser = ArgumentParser()
    parser.add_argument("--json", type=Path, required=True)
    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    json_path = args.json

    with open(json_path, "r") as file:
        json_data = json.load(file)
    everything = Everything.from_json(json_data)
    everything.check()