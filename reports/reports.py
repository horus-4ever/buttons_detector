import abc
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import torch
from evaluate.evaluate import EvaluationResult
from model.ptmodel import PTModel
from .utils import flatten_data, swap_layers, load_weights
from matplotlib.ticker import MaxNLocator


@dataclass
class PTModelData:
    """
    Stores the data contained in a PTModel.
    """
    name: str
    data: dict

    @classmethod
    def from_checkpoint(cls, checkpoint: PTModel):
        flattened_losses = flatten_data(checkpoint.losses, 2)
        flattened_evaluations = flatten_data(checkpoint.evaluations, 5)
        return cls(
            name=checkpoint.model_configuration.name,
            data={
                "losses": flattened_losses,
                "evaluations": flattened_evaluations
            }
        )

    def get(self, path: str):
        """
        Data is stored as nested dictionnaries.
        We implement a simple lookup algorithm to retrieve the desired values from the given path.
        The return value is a numpy array of the values.
        """
        paths = path.split(".")
        result = self.data
        for path in paths:
            result = result[path]
        return np.array(result, dtype=float)


class ReportType(StrEnum):
    CURVE = "curve"
    BAR = "bar"


@dataclass
class ModelGroup:
    """
    Defines a model group.
    A model group refers to a group of same models but trained on different seeds / splits.
    In other words, a model group is a group of models trained to study variance.
    """
    name: str
    models: list[PTModelData]

    def get(self, path: str, last_only: bool = False):
        """
        Return the result as a numpy array of shape (<n_models>, <data_size>).
        The 'last_only' parameters is used to retrieve the last value of the data.
        This may be used for retrieving only last epoch evaluation.
        """
        results = []
        for model in self.models:
            result = model.get(path)
            if last_only:
                result = result[-1]
            results.append(result)
        return np.array(results, dtype=float)


@dataclass
class ReportRequest:
    """
    Defines a report request.
    """
    title: str
    filename: str
    type: ReportType
    models: list[ModelGroup]
    data: dict

    @classmethod
    def from_json(cls, json_data: dict, all_models: dict[str, PTModelData]):
        models = []
        for group in json_data["models"]:
            model_group = ModelGroup(
                name=group["name"],
                models=[all_models[model_name] for model_name in group["models"]]
            )
            models.append(model_group)
        return cls(
            title=json_data["title"],
            type=ReportType(json_data["type"]),
            filename=json_data["filename"],
            models=models,
            data=json_data["data"]
        )


@dataclass
class Report:
    """
    Define a series of reports to generate.
    """
    root: Path
    models: dict[str, PTModelData]
    reports: list[ReportRequest]

    @classmethod
    def from_json(cls, json_data: dict):
        root = Path(json_data["root"])
        models = {}
        for model_name in json_data["models"]:
            path = root / f"{model_name}.pt"
            model = load_weights(path, device=torch.device("cpu"))
            models[model_name] = PTModelData.from_checkpoint(model)
        reports = [ReportRequest.from_json(report_data, models) for report_data in json_data["reports"]]
        return cls(
            root=root,
            models=models,
            reports=reports
        )

    def _collect_data(self, report: ReportRequest, last_only: bool = False):
        groups = {}
        for group in report.models:
            group_data = {}
            for curve_name, curve_path in report.data.items():
                group_data[curve_name] = group.get(curve_path, last_only=last_only)
            groups[group.name] = group_data
        return groups

    def _generate_curve_report(self, report: ReportRequest):
        data = self._collect_data(report, last_only=False)
        figure, ax = plt.subplots(layout="constrained")
        for group_name, group_data in data.items():
            for line_name, line_data in group_data.items():
                x = np.arange(1, line_data.shape[1] + 1)
                mean = np.mean(line_data, axis=0)
                min = np.min(line_data, axis=0)
                max = np.max(line_data, axis=0)
                sns.lineplot(x=x, y=mean, ax=ax, label=f"{group_name} - {line_name}")
                ax.fill_between(x, min, max, alpha=0.2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(report.title)
        ax.set_xlabel("Epochs")
        ax.set_ylabel("Value")
        ax.legend()
        return figure
        
    def _generate_bar_report(self, report: ReportRequest):
        data = self._collect_data(report, last_only=True)
        figure, ax = plt.subplots(layout="constrained")
        n_groups = len(data)
        n_bars = len(next(iter(data.values())))
        bar_width = 0.8 / n_groups
        x = np.arange(n_bars)
        for i, (group_name, group_data) in enumerate(data.items()):
            offset = (i - (n_groups - 1) / 2) * bar_width
            means = [np.mean(group_data[line_name]) for line_name in group_data]
            yerr = [np.std(group_data[line_name]) for line_name in group_data]
            ax.bar(x + offset, means, width=bar_width, label=group_name, yerr=yerr)
        ax.set_xticks(x)
        ax.set_xticklabels(next(iter(data.values())).keys())
        ax.set_title(report.title)
        ax.set_ylabel("Value")
        ax.legend()
        return figure

    def _generate_report(self, report: ReportRequest):
        match report.type:
            case ReportType.CURVE:
                return self._generate_curve_report(report)
            case ReportType.BAR:
                return self._generate_bar_report(report)

    def generate_reports(self, output_path: Path):
        for report in self.reports:
            figure = self._generate_report(report)
            figure.savefig(output_path / report.filename)
            plt.close(figure)
