from dataclasses import dataclass, field
import torch


@dataclass
class BoundingBox:
    """
    Represents a bounding box. `cx` and `cy` are the center coordinates.
    `w` and `h` are respectively the width and the height of the bounding box.
    """
    cx: float
    cy: float
    w: float
    h: float

    def to_tensor(self, device = None):
        """
        Return the bounding box as a torch tensor.
        """
        return torch.tensor([self.cx, self.cy, self.w, self.h], dtype=torch.float, device=device)

    def to_json(self):
        return {
            "cx": self.cx,
            "cy": self.cy,
            "w": self.w,
            "h": self.h
        }

    @classmethod
    def from_json(cls, json_data: dict) -> "BoundingBox":
        return cls(
            cx=json_data["cx"],
            cy=json_data["cy"],
            w=json_data["w"],
            h=json_data["h"]
        )
    
    @classmethod
    def from_x1y1x2y2(cls, x1: float, y1: float, x2: float, y2: float) -> "BoundingBox":
        """
        Creates a normalized bounding box from corner coordinates.
        """
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        return cls(cx, cy, width, height)
    
    def to_x1y1x2y2(self) -> tuple[float, float, float, float]:
        """
        Converts the bounding box from center coordinates to corner coordinates.
        Returns a tuple of (x1, y1, x2, y2).
        """
        x1 = self.cx - self.w / 2
        y1 = self.cy - self.h / 2
        x2 = self.cx + self.w / 2
        y2 = self.cy + self.h / 2
        return (x1, y1, x2, y2)


@dataclass
class Button:
    """
    Represents a button.
    """
    bbox: BoundingBox
    visible: bool

    def to_tensor(self, device = None):
        """
        Return the bbox as a torch tensor.
        """
        return self.bbox.to_tensor(device=device)

    def to_json(self):
        return {
            "bbox": self.bbox.to_json(),
            "visible": self.visible
        }

    @classmethod
    def from_json(cls, json_data: dict) -> "Button":
        return cls(
            bbox=BoundingBox.from_json(json_data["bbox"]),
            visible=json_data["visible"]
        )


@dataclass
class Counterpart:
    """
    Represents a counterpart.
    A counterpart will be generally a velcro, a buttonhole or a snap button.
    """
    bbox: BoundingBox
    visible: bool
    type: str

    def to_tensor(self, device = None):
        """
        Return the bbox as a torch tensor.
        """
        return self.bbox.to_tensor(device=device)

    def to_json(self):
        return {
            "bbox": self.bbox.to_json(),
            "visible": self.visible,
            "type": self.type
        }

    @classmethod
    def from_json(cls, json_data: dict) -> "Counterpart":
        return cls(
            bbox=BoundingBox.from_json(json_data["bbox"]),
            visible=json_data["visible"],
            type=json_data["type"]
        )


@dataclass
class Pair:
    """
    Represents a pair between a button and a counterpart.
    """
    button: Button
    counterpart: Counterpart
    fastened: bool

    def to_json(self):
        return {
            "button": self.button.to_json(),
            "counterpart": self.counterpart.to_json(),
            "fastened": self.fastened
        }

    @classmethod
    def from_json(cls, json_data: dict) -> "Pair":
        return cls(
            button=Button.from_json(json_data["button"]),
            counterpart=Counterpart.from_json(json_data["counterpart"]),
            fastened=json_data["fastened"]
        )


@dataclass
class Cloth:
    """
    Represent a clothing item, with its pairs of <button, counterpart>.
    `segmentation` represents the path to the segmentation mask of the clothing item.
    """
    type: str
    segmentation: str
    pairs: list[Pair]

    def to_json(self):
        return {
            "type": self.type,
            "segmentation": self.segmentation,
            "pairs": list(pair.to_json() for pair in self.pairs)
        }

    @classmethod
    def from_json(cls, json_data: dict) -> "Cloth":
        return cls(
            type=json_data["type"],
            segmentation=json_data["segmentation"],
            pairs=[Pair.from_json(pair) for pair in json_data["pairs"]]
        )


@dataclass
class ImageInfo:
    """
    Represents an image in the dataset.
    """
    url: str
    width: int
    height: int

    def to_json(self):
        return {
            "url": self.url,
            "width": self.width,
            "height": self.height
        }

    @classmethod
    def from_json(cls, json_data: dict) -> "ImageInfo":
        return cls(
            url=json_data["url"],
            width=json_data["width"],
            height=json_data["height"]
        )


@dataclass
class Annotation:
    """
    Represents an annotation of a clothing item in an image.
    """
    image: ImageInfo
    cloth: Cloth

    def to_tensor(self, device = None):
        """
        Returns the annotations for the image a a tensor.
        The image is not returned.
        - returns: [n_buttons, 4]
        """
        pairs = self.cloth.pairs
        if pairs:
            labels_button = torch.stack([label.button.to_tensor(device=device) for label in pairs])
            labels_counterpart = torch.stack([label.counterpart.to_tensor(device=device) for label in pairs])
        else:
            labels_button = torch.tensor([], device=device)
            labels_counterpart = torch.tensor([], device=device)
        visibility_counterpart = torch.tensor([label.counterpart.visible for label in pairs], dtype=torch.float, device=device)
        visibility_button = torch.tensor([label.button.visible for label in pairs], dtype=torch.float, device=device)
        classes = torch.zeros(labels_button.size()[0], dtype=torch.long, device=device)
        return classes, labels_button, labels_counterpart, visibility_button, visibility_counterpart

    def to_json(self):
        return {
            "image": self.image.to_json(),
            "cloth": self.cloth.to_json() 
        }

    @classmethod
    def from_json(cls, json_data: dict) -> "Annotation":
        return cls(
            image=ImageInfo.from_json(json_data["image"]),
            cloth=Cloth.from_json(json_data["cloth"])
        )


@dataclass
class Garment:
    """
    Represents a garment
    """
    type: str
    fastener: str
    assistive: bool
    n_pairs: int
    description: str

    def to_json(self):
        return {
            "type": self.type,
            "fastener": self.fastener,
            "assistive": self.assistive,
            "n_pairs": self.n_pairs,
            "description": self.description
        }

    @classmethod
    def from_json(cls, json_data: dict) -> "Garment":
        return cls(
            type=json_data["type"],
            fastener=json_data["fastener"],
            assistive=json_data["assistive"],
            n_pairs=json_data["n_pairs"],
            description=json_data["description"]
        )

    @classmethod
    def empty(cls) -> "Garment":
        return cls(
            type="",
            fastener="",
            assistive=False,
            n_pairs=0,
            description=""
        )
