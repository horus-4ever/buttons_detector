from dataclasses import dataclass, field
from dataformat.dataformat import Garment
from pathlib import Path
import json
import tkinter as tk
from .utils import AnyEvent, Event, ListVar


class Application:
    """
    Represents the application.
    """
    def __init__(self):
        self.garment_split = GarmentSplit()
        self.menu = Menu(application=self)

    def _load_garment(self, garment_path: Path) -> Garment:
        if not garment_path.exists():
            return Garment.empty()
        with open(garment_path, "r") as f:
            garment_data = json.load(f)
        return Garment.from_json(garment_data)

    def open_directory(self, path: Path):
        directories = [d for d in path.iterdir() if d.is_dir()]
        directories.sort(key=lambda d: d.name)
        self.garment_split.reset()
        for directory in directories:
            garment_path = directory / "garment.json"
            garment = self._load_garment(garment_path)
            garment_entry = GarmentEntry(garment=garment, path=garment_path, name=directory.name, use=True)
            self.garment_split.train.append(garment_entry)
        self.garment_split.root_path = path

class Menu:
    """
    Represents the menu of the application.
    """
    def __init__(self, application: Application):
        self.application = application

    def open_directory(self, path: Path):
        self.application.open_directory(path)


@dataclass
class GarmentEntry:
    garment: Garment
    name: str
    use: bool
    path: Path = field(default_factory=Path)
    position: str = field(default="train")



class GarmentInfo:
    def __init__(self, garment: Garment):
        self._garment = garment
        self.type = tk.StringVar(value=garment.type)
        self.fastener = tk.StringVar(value=garment.fastener)
        self.assistive = tk.BooleanVar(value=garment.assistive)
        self.n_pairs = tk.IntVar(value=garment.n_pairs)
        self.description = tk.StringVar(value=garment.description)

    def backpopulate(self):
        self._garment.type = self.type.get()
        self._garment.fastener = self.fastener.get()
        self._garment.assistive = self.assistive.get()
        self._garment.n_pairs = self.n_pairs.get()
        self._garment.description = self.description.get()

    def save(self, path: Path):
        with open(path, "w") as f:
            json.dump(self._garment.to_json(), f, indent=4)


class GarmentSplit:
    def __init__(self, root_path: Path | None = None, train: list | None = None, val: list | None = None, test: list | None = None):
        self.root_path = root_path or Path("")
        self.train = ListVar(train or [])
        self.val = ListVar(val or [])
        self.test = ListVar(test or [])
        # event to notify when the garment split changes
        self.changed_event = AnyEvent("changed_event", [self.train.on_changed, self.val.on_changed, self.test.on_changed])

    def reset(self):
        self.train.clear()
        self.val.clear()
        self.test.clear()
        self.changed_event.fire(self)

    def _get_split(self, split_name: str) -> ListVar:
        match split_name:
            case "train":
                return self.train
            case "val":
                return self.val
            case "test":
                return self.test
            case _:
                raise ValueError(f"Invalid split name: {split_name}")

    def move_to(self, garment: GarmentEntry, source: str, destination: str):
        """
        Moves a garment from one split to another.
        """
        source_split = self._get_split(source)
        dest_split = self._get_split(destination)
        source_split.remove(garment)
        dest_split.append(garment)
        garment.position = destination
        self.changed_event.fire()

    def to_json(self):
        return {
            "root": str(self.root_path),
            "train": [garment.name for garment in self.train],
            "validation": [garment.name for garment in self.val],
            "test": [garment.name for garment in self.test]
        }
