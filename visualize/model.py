from typing import Iterable, Iterator
from .utils import Event
from pathlib import Path
import torch
import random
from model.prtr import build_model
from model.config import ModelConfig



def dummy_data():
    all_losses = []
    for _ in range(40):
        losses = {
            "GIoU_loss_pair": torch.tensor(random.random()),
            "GIoU_loss_buttons": torch.tensor(random.random()),
            "GIoU_loss_counterparts": torch.tensor(random.random()),
            "L1_loss_pair": torch.tensor(random.random()),
            "L1_loss_buttons": torch.tensor(random.random()),
            "L1_loss_counterparts": torch.tensor(random.random()),
            "class_loss": torch.tensor(random.random())
        }
        losses["loss"] = losses["GIoU_loss_pair"] + losses["L1_loss_pair"] + losses["class_loss"]
        all_losses.append(losses)
    return all_losses



class Application:
    def __init__(self, device = None):
        self.loss_graph_frame = GraphFrame()
        self.eval_graph_frame = GraphFrame()
        self.frame_manager = FrameManager(
            frames={
                "loss_graph": self.loss_graph_frame,
                "eval_graph": self.eval_graph_frame
            },
            active_frame="loss_graph"
        )
        self.device = device or torch.device("cpu")
        # self.open(Path("good_runs/good_run_8.pt"), device=device)

    def open(self, weights_path: Path):
        self.checkpoint = torch.load(weights_path, map_location=self.device, weights_only=False)
        # self._load_model(self.checkpoint)
        self._create_graph(self.checkpoint)

    def _load_model(self, checkpoint):
        model_configuration = ModelConfig.from_json(self.checkpoint["model_configuration"])
        self.model = build_model(model_configuration)
        self.model.load_state_dict(self.checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def _create_graph(self, checkpoint):
        losses = dummy_data() # checkpoint["losses"]
        losses = self._format_data(losses)
        graph = Graph(losses)
        self.frame_manager["loss_graph"].set_graph(graph)

    def _format_data(self, losses: list[dict]):
        result = {}
        for data_point in losses:
            for k, v in data_point.items():
                if k not in result:
                    result[k] = []
                result[k].append(v)
        return result


class FrameManager:
    def __init__(self, frames: dict | None = None, active_frame: str | None = None):
        self.frames = frames or {}
        self.active_frame = active_frame
        # define an event
        self.changed = Event("changed")

    def add_frame(self, name, frame):
        self.frames[name] = frame
        self.active_frame = name

    def set_active_frame(self, name):
        self.active_frame = name
        self.changed.fire(self)

    def get_active_frame(self):
        if self.active_frame is None:
            raise ValueError("No active frame defined.")
        return self.frames[self.active_frame]

    def __getitem__(self, key):
        return self.frames[key]

    def __len__(self):
        return len(self.frames)

    def __iter__(self):
        return iter(self.frames.items())


class CurveToogle:
    def __init__(self, curve_name: str, initial_value: bool = True):
        self.curve_name = curve_name
        self._visible = initial_value
        # define the event
        self.changed = Event("changed")

    @property
    def visible(self):
        return self._visible

    def toggle(self):
        self._visible = not self._visible
        self.changed.fire(self)

    def enable(self):
        self._visible = True
        self.changed.fire(self)

    def disable(self):
        self._visible = False
        self.changed.fire(self)


class CurveToogleSet:
    def __init__(self, curve_names: list[str]):
        self.curves_names = curve_names
        self._create_toggles()

    def _create_toggles(self):
        self.toggles = {}
        for curve_name in self.curves_names:
            toggle = CurveToogle(curve_name)
            self.toggles[curve_name] = toggle

    def enable(self):
        for toggle in self.toggles.values():
            toggle.enable()

    def disable(self):
        for toggle in self.toggles.values():
            toggle.disable()

    def __iter__(self) -> Iterator[CurveToogle]:
        return iter(self.toggles.values())

    def __getitem__(self, key):
        return self.toggles[key]


class GraphFrame:
    def __init__(self, graph = None):
        self.graph = graph or Graph()
        # event
        self.changed = Event("changed")

    def set_graph(self, graph: "Graph"):
        self.graph = graph
        self.changed.fire(self)


class Graph:
    def __init__(self, data: dict | None = None):
        self.data = data or {}
        self.toggles = CurveToogleSet(list(self.data.keys()))
