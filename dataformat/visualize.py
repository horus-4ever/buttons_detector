from PIL import Image, ImageDraw
from .dataformat import Annotation, Pair


def _draw_one_pair(draw: ImageDraw.ImageDraw, pair: Pair):
    """
    Draws a single pair <button, fastener> on the image.
    """
    button = pair.button
    fastener = pair.fastener
    draw.rectangle(button.bbox.to_x1y1x2y2(), outline="green", width=2)
    draw.rectangle(fastener.bbox.to_x1y1x2y2(), outline="orange", width=2)


def visualize_one(annotation: Annotation) -> Image.Image:
    """
    Visualizes a single annotation by drawing the bounding box of the clothing item on the image.
    """
    image = Image.open(annotation.image.url)
    draw = ImageDraw.Draw(image)
    pairs = annotation.clothing_item.pairs
    for pair in pairs:
        _draw_one_pair(draw, pair)
    return image