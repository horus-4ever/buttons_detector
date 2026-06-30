from PIL import Image
from PIL import ImageDraw
from pathlib import Path
from dataclasses import dataclass
import json


@dataclass
class Item:
    image: Image.Image
    image_path: Path
    annotation_path: Path
    segmentations: list[Image.Image]


class Loader:
    def __init__(self, seg_path: Path, ann_path: Path, img_path: Path):
        self.seg_path = seg_path
        self.ann_path = ann_path
        self.img_path = img_path
        self.seg_paths = None
        self.ann_paths = None
        self.img_paths = None
        # load the annotations
        self.load()

    def load(self) -> "Loader":
        self.seg_paths = list(self.seg_path.glob("*.json"))
        self.ann_paths = list(self.ann_path.glob("*.json"))
        self.img_paths = list(self.img_path.glob("*.jpg"))
        # now sort them in the same order
        # we assume that all annotations and images have the same stem
        self.seg_paths.sort(key=lambda path: path.stem)
        self.ann_paths.sort(key=lambda path: path.stem)
        self.img_paths.sort(key=lambda path: path.stem)
        return self
    
    def __len__(self):
        assert self.seg_paths is not None
        return len(self.seg_paths)
    
    def _segmentations_to_image(self, image: Image.Image, segmentations):
        """
        `segmentation` is in the format [[x1,y1,...xn,yn],[ ]], which define a polygon.
        """
        W, H = image.size
        result = Image.new(mode="RGB", size=(W, H), color=(0, 0, 0))
        draw = ImageDraw.Draw(result)
        # iterate over segmentations
        for segmentation in segmentations:
            if len(segmentation) < 6: # the segmentation must be a polygon so at least 3 points
                continue
            # get and draw the points for the given sub-segmentation
            points = [
                (segmentation[i], segmentation[i + 1])
                for i in range(0, len(segmentation), 2)
            ]
            draw.polygon(points, fill=(255, 255, 255))
        return result


    def _get_segmentations_data(self, path: Path):
        segmentations = []
        with open(path, "r") as file:
            json_data = json.load(file)
        for key, data in json_data.items():
            if not key.startswith("item"):
                continue
            # WARNING: segmentation data is a list because the object may be in multiple parts
            segmentation_data = data["segmentation"] # list
            segmentations.append(segmentation_data)
        return segmentations
    
    def __getitem__(self, key):
        if self.seg_paths is None or self.ann_paths is None or self.img_paths is None:
            raise ValueError("Loader is not initialized.")
        # load the image
        image_path = self.img_paths[key]
        image = Image.open(image_path)
        # now load the segmentations data
        # sometimes the segmentation path is not direct so find it from the image path
        seg_path = None
        for path in self.seg_paths:
            if path.stem == image_path.stem:
                seg_path = path
        if seg_path is None:
            raise ValueError("This should not happen.")
        segmentations = self._get_segmentations_data(seg_path)
        # now for all segmentations turn it into a binary image
        segmentation_images = []
        for object in segmentations:
            seg_image = self._segmentations_to_image(image, object)
            segmentation_images.append(seg_image)
        # now construct the item object
        annotation_path = self.ann_paths[key]
        result = Item(
            image=image,
            image_path=image_path,
            annotation_path=annotation_path,
            segmentations=segmentation_images
        )
        return result


class Application:
    def __init__(self):
        self.loader = None
        self.annotations_path = None
        self.segmentations_path = None
        self.images_path = None
        self.current_index = 0

    @property
    def initialized(self):
        return self.annotations_path and self.segmentations_path and self.images_path

    def _try_initialize(self):
        if not self.initialized:
            return
        # initialize the loader
        self.loader = Loader(self.segmentations_path, self.annotations_path, self.images_path)

    def open_annotations(self, path):
        self.annotations_path = path

    def open_segmentations(self, path):
        self.segmentations_path = path

    def open_images(self, path):
        self.images_path = path

    @property
    def current_item(self):
        if not self.loader:
            return None
        return self.loader[self.current_index]

    def next_item(self):
        if not self.loader:
            return None
        self.current_index = (self.current_index + 1) % len(self.loader)

    def previous_item(self):
        if not self.loader:
            return None
        if self.current_index == 0:
            self.current_index = len(self.loader) - 1
        else:
            self.current_index -= 1


if __name__ == "__main__":
    loader = Loader(
        Path("/home/horus/Downloads/test_annotations/segmentations"),
        Path("/home/horus/Downloads/test_annotations/images_and_ann"),
        Path("/home/horus/Downloads/test_annotations/images_and_ann")
    )
    loader[0]