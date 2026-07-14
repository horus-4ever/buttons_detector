from PIL import Image, ImageDraw
from dataformat import Annotation, Pair
from pathlib import Path
from argparse import ArgumentParser
import json


def nbbox_to_bbox(bbox, width, height):
    """
    Converts a normalized bounding box into real coordinates bounding box.
    """
    x1, y1, x2, y2 = bbox
    x1, y1 = x1 * width, y1 * height
    x2, y2 = x2 * width, y2 * height
    return (x1, y1, x2, y2)


def _draw_one_pair(draw: ImageDraw.ImageDraw, pair: Pair, w, h):
    """
    Draws a single pair <button, fastener> on the image.
    """
    button = pair.button
    fastener = pair.fastener
    draw.rectangle(nbbox_to_bbox(button.bbox.to_x1y1x2y2(), w, h), outline="green", width=2)
    draw.rectangle(nbbox_to_bbox(fastener.bbox.to_x1y1x2y2(), w, h), outline="orange", width=2)


def visualize_one(annotation: Annotation, annotation_path: Path) -> Image.Image:
    """
    Visualizes a single annotation by drawing the bounding box of the clothing item on the image.
    """
    image = Image.open(annotation_path / "images" / annotation.image.url)
    w, h = image.size
    draw = ImageDraw.Draw(image)
    pairs = annotation.cloth.pairs
    for pair in pairs:
        _draw_one_pair(draw, pair, w, h)
    return image


def init_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--root", type="str", required=True, help="Directory root containing annotations/ and images/.")
    parser.add_argument("--name", type="str", required=True, help="Path to the JSON file containing the configuration.")
    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    root = Path(args.root)
    json_name = args.name
    json_path = root / "annotations" / json_name
    # load the json
    with open(json_path) as file:
        json_data = json.load(file)
        ann = Annotation.from_json(json_data)
        image = visualize_one(ann, root)
        image.save("out.png")