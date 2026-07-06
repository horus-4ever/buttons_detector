from dataclasses import dataclass, field
from enum import StrEnum, Enum
from pathlib import Path
import json


class SamplingPolicy(StrEnum):
    RANDOM = "RANDOM"
    ALL_COMBINATIONS = "ALL_COMBINATIONS"

    @classmethod
    def from_str(cls, value: str):
        return cls._member_map_[value.upper()]


@dataclass
class ParameterValue:
    value: object
    parameter: "Parameter" = field(init=False)

    @property
    def name(self):
        return self.parameter.name
    
    def to_json(self):
        return {
            "name": self.name,
            "value": self.value
        }
    
    def __repr__(self):
        return f"ParameterValue(name={repr(self.name)}, value={repr(self.value)})"


@dataclass
class Parameter:
    name: str
    training_values: list[ParameterValue]
    testing_values: list[ParameterValue]
    sampling_policy: Enum

    @classmethod
    def from_json(cls, json_data) -> "Parameter":
        name = json_data["name"]
        training_values = [ParameterValue(obj) for obj in json_data["training_values"]]
        testing_values = [ParameterValue(obj) for obj in json_data["testing_values"]]
        sampling_policy = SamplingPolicy.from_str(json_data["sampling_policy"])
        return cls(name, training_values, testing_values, sampling_policy)
    
    def __post_init__(self):
        for elem in self.training_values:
            elem.parameter = self
        for elem in self.testing_values:
            elem.parameter = self

@dataclass
class Parameters:
    parameters: list[Parameter]

    @property
    def permutation_parameters(self):
        result = []
        for parameter in self.parameters:
            if parameter.sampling_policy == SamplingPolicy.ALL_COMBINATIONS:
                result.append(parameter)
        return result
    
    @property
    def random_parameters(self):
        result = []
        for parameter in self.parameters:
            if parameter.sampling_policy == SamplingPolicy.RANDOM:
                result.append(parameter)
        return result

    def get_parameters(self) -> tuple:
        return self.permutation_parameters, self.random_parameters


@dataclass
class SceneConfiguration:
    name: str
    no_distractions: float
    max_generate: int

    @classmethod
    def from_json(cls, name: str, json_data) -> "SceneConfiguration":
        no_distractions = json_data["no_distractions"]
        max_generate = json_data["max_generate"]
        return cls(name, no_distractions, max_generate)


@dataclass
class Configuration:
    seed: int
    scenes: list[SceneConfiguration]
    parameters: Parameters

    @classmethod
    def from_json(cls, json_data) -> "Configuration":
        seed = json_data["seed"]
        scenes = [SceneConfiguration.from_json(name, config) for name, config in json_data["scenes"].items()]
        parameters = Parameters([Parameter.from_json(param) for param in json_data["parameters"]])
        return cls(seed=seed, scenes=scenes, parameters=parameters)


def read_configuration(path: Path):
    with open(path) as file:
        json_data = json.load(file)
        return Configuration.from_json(json_data)
