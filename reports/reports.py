import abc
from dataclasses import dataclass

from matplotlib import pyplot as plt
import numpy as np
from evaluate.evaluate import EvaluationResult
from model.ptmodel import PTModel
from .utils import flatten_data, swap_layers


@dataclass
class EvaluationFigureRequest:
    curves: list[str]
    curves_names: list[str]
    fig_name: str


@dataclass
class LossFigureRequest:
    curves: list[str]
    curves_names: list[str]
    fig_name: str


class Visualizer(abc.ABC):
    def get(self, data, name: str):
        name, _, what = name.partition(":")
        paths = name.split(".")
        result = data
        for path in paths:
            result = result[path]
        result = [elem.to_dict() if isinstance(elem, EvaluationResult) else {"": elem} for elem in result]
        data = flatten_data(result, 1)
        return data[what]


class EvaluationVisualizer(Visualizer):
    def __init__(self, data: dict, length: int):
        self.data = data
        self.length = length

    @classmethod
    def from_checkpoint(cls, checkpoint: PTModel):
        length = len(checkpoint.evaluations)
        flattened_data = flatten_data(checkpoint.evaluations, 4)
        flattened_data = swap_layers(flattened_data, 2)
        return cls(
            data=flattened_data,
            length=length
        )

    @property
    def xs(self):
        return np.arange(1, self.length + 1)

    def visualize(self, request: EvaluationFigureRequest):
        figure = plt.figure()
        plot = figure.add_subplot(111)
        x = self.xs
        for curve_path, curve_name in zip(request.curves, request.curves_names):
            datas = self.get(self.data, curve_path)
            plot.plot(x, datas, label=curve_name)
        plot.legend()
        plot.set_title(request.fig_name)
        plot.set_xlabel("Epochs")
        plot.set_xlim(1, self.length)
        plot.set_ylim(0, 1.1)
        return plot, figure


class LossVisualizer(Visualizer):
    def __init__(self, data: dict, length: int):
        self.data = data
        self.length = length

    @classmethod
    def from_checkpoint(cls, checkpoint: PTModel):
        length = len(checkpoint.losses)
        flattened_data = flatten_data(checkpoint.losses, 2)
        return cls(
            data=flattened_data,
            length=length
        )

    def visualize(self, request: LossFigureRequest):
        figure = plt.figure()
        plot = figure.add_subplot(111)
        x = np.arange(1, self.length + 1)
        for curve_path, curve_name in zip(request.curves, request.curves_names):
            datas = self.get(self.data, curve_path)
            plot.plot(x, datas, label=curve_name, marker="o")
        plot.legend()
        plot.set_title(request.fig_name)
        plot.set_xlabel("Epochs")
        plot.set_ylabel("Loss")
        plot.set_xlim(1, self.length)
        plot.set_ylim(0)
        return plot, figure