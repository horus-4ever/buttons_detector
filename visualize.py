import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms
from mpl_toolkits.axes_grid1 import ImageGrid

from prtr import build_model_from


DATASET_ROOT = Path("dataset")
IMAGES_DIR = DATASET_ROOT / "images"

CHECKPOINT_DIR = Path("good_runs")
OUTPUT_DIR = Path("viz_outputs")

DEVICE = torch.device("cpu" if torch.cuda.is_available() else "cpu")

INFERENCE_SIZE = 512

# DETR-style class layout:
#   class 0 = button
#   final class = no-object
BUTTON_CLASS_ID = 0


@dataclass
class Prediction:
    class_id: int
    pos_x: float
    pos_y: float
    confidence: float


def load_model(model_config_path: str | Path, model_weights_path: str | Path, device: torch.device):
    model = build_model_from(str(model_config_path))
    checkpoint = torch.load(model_weights_path, map_location=device)
    if "model_state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint does not contain 'model_state_dict'. "
            f"Available keys: {list(checkpoint.keys())}"
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model



def load_image(path: Path) -> Image.Image:
    image_path = path
    image = Image.open(image_path).convert("RGB")
    return image


def iter_image_files(directory: Path) -> list[Path]:
    image_files: list[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        image_files.extend(sorted(directory.glob(ext)))
    return sorted(image_files)


def preprocess_image(image: Image.Image, inference_size: int) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((inference_size, inference_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])
    return transform(image).unsqueeze(0)


def normalized_to_pixel_xy(xy: torch.Tensor, width: int, height: int) -> tuple[float, float]:
    x = float(xy[0]) * width
    y = float(xy[1]) * height
    return x, y


@torch.no_grad()
def run_model(model, image: Image.Image, device: torch.device, inference_size: int):
    width, height = image.size
    x = preprocess_image(image, inference_size).to(device)
    # masks: [B, H_img, W_img]
    masks = torch.zeros([1, height, width], dtype=torch.bool)
    outputs = model(x, masks)
    return outputs


def get_predictions(image, outputs):
    width, height = image.size
    # put on the CPU, and we have only one batch
    pred_logits = outputs["pred_logits"][0].detach().cpu()      # [Q, C + 1]
    predictions = outputs["pred_buttons"][0].detach().cpu()    # [Q,  RpQ, 2]
    pred_buttons = predictions[:, 0, :] # take the button predictions
    pred_holes = predictions[:, 1, :] # take the hole predictions
    # transforms to probabilities
    pred_probs = pred_logits.softmax(dim=-1) # [Q, C + 1]
    pred_classes = pred_probs.argmax(dim=-1)
    pred_scores = pred_probs.max(dim=-1).values
    # now loop over the queries and get the predictions
    num_queries, num_classes = pred_logits.size()
    buttons_predictions = []
    holes_predictions = []
    for query_idx in range(num_queries):
        class_id = int(pred_classes[query_idx])
        confidence = float(pred_probs[query_idx, BUTTON_CLASS_ID])
        xy_norm_tensor = pred_buttons[query_idx]
        pos_x, pos_y = normalized_to_pixel_xy(xy_norm_tensor, width, height)
        buttons_predictions.append(Prediction(class_id, pos_x, pos_y, confidence))
        # now the same for the holes
        xy_norm_tensor = pred_holes[query_idx]
        pos_x, pos_y = normalized_to_pixel_xy(xy_norm_tensor, width, height)
        holes_predictions.append(Prediction(class_id, pos_x, pos_y, confidence))
    return buttons_predictions, holes_predictions


def get_attention_maps(attn_weights):
    """
    - attn_weights: decoder_layers * [batch, query_len, heads, num_levels, num_points]

    - output: [query_len, heads, num_levels, num_points]
    Keeps only the last decoder layer attention weights.
    """
    attn_weights = attn_weights[-1][0] # get the attention weigths of the last decoder layer, first image
    return attn_weights


def visualize_attention(image: Image.Image, attn_maps, reference_points, sampling_locations, predictions) -> list[Image.Image]:
    """
    - attn_maps: [query_len * RpQ, heads, num_levels, num_points]
    - spatial_shapes: [num_levels, 2]
    - sampling_locations: [batch, query_len, heads, num_levels, num_points, 2]
    - reference_points: [num_queries, RpQ, 2]
    """
    image = image.convert("L").convert("RGBA")
    W_img, H_img = image.size
    num_queries, num_heads, num_levels, num_points = attn_maps.size()
    # --> [query_len, heads, num_levels, num_points, 2]
    sampling_locations = sampling_locations[0] # first image of the batch
    # flatten that
    sampling_locations = sampling_locations.reshape(num_queries, -1, 2)
    attn_maps = attn_maps.reshape(num_queries, -1)
    # loop over the queries to have each attention per query
    query_attn_maps_images = []
    for attn_map, locations, ref_points, prediction in zip(attn_maps, sampling_locations, reference_points, predictions):
        if prediction.class_id != BUTTON_CLASS_ID:
            continue
        # draw the attention
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        image_draw = ImageDraw.Draw(overlay)
        for (posx, posy), value in zip(locations, attn_map):
            px = float(posx * W_img)
            py = float(posy * H_img)
            alpha = int(40 + 215 * float(value))
            radius = 5
            image_draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                fill=(255, 0, 0, alpha),
            )
        for ref_point in ref_points:
            x, y = ref_point
            px, py = float(x * W_img), float(y * H_img)
            radius = 5
            image_draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                fill=(0, 255, 255, 255),
            )
        # then draw the predictions
        if prediction.class_id == BUTTON_CLASS_ID:
            x, y = prediction.pos_x, prediction.pos_y
            radius = 10
            image_draw.ellipse(
                (x - radius, y - radius, x + radius, y + radius),
                fill=(0, 255, 0, 255)
            )
        attn_map_image = Image.alpha_composite(image, overlay).convert("RGB")
        query_attn_maps_images.append(attn_map_image)
    return query_attn_maps_images


def visualize_predictions(image: Image.Image, button_predictions, hole_predictions):
    W, H = image.size
    result_image= image.copy()
    blackboard = ImageDraw.Draw(result_image)
    for button_prediction, hole_prediction in zip(button_predictions, hole_predictions):
        if button_prediction.class_id != BUTTON_CLASS_ID:
            continue
        button_x, button_y = button_prediction.pos_x, button_prediction.pos_y
        hole_x, hole_y = hole_prediction.pos_x, hole_prediction.pos_y
        radius = 5
        blackboard.ellipse(
            (button_x - radius, button_y - radius, button_x + radius, button_y + radius),
            fill=(0, 255, 0, 255)
        )
        blackboard.ellipse(
            (hole_x - radius, hole_y - radius, hole_x + radius, hole_y + radius),
            fill=(255, 200, 0, 255)
        )
        blackboard.line(
            (button_x, button_y, hole_x, hole_y), fill=(0, 0, 0, 0), width=4
        )
    return result_image


def create_plot(images):
    fig = plt.figure(figsize=(5, 2))
    images = [img.resize((512, 512)) for img in images]
    # Create an ImageGrid with a custom padding (axes_pad) in inches
    grid = ImageGrid(fig, 111,          # similar to subplot(111)
                    nrows_ncols=(2, 5), # 2x2 grid
                    axes_pad=0.1,       # pad between images
                    )

    for ax, img in zip(grid, images):
        ax.imshow(img)
        ax.axis('off')
    return fig


def visualize_one(
    path: Path,
    model,
    output_dir: Path,
    device: torch.device,
    inference_size: int,
    save_attention_maps: bool = True,
    min_button_score: Optional[float] = None,
):
    image = load_image(path)
    image_name = path.stem

    outputs = run_model(model, image, device, inference_size)
    reference_points = outputs["reference_points"] # [num_queries, RpQ, 2]
    # get the predictions
    button_predictions, hole_predictions = get_predictions(image, outputs)
    print(len(list(filter(lambda p: p.class_id == BUTTON_CLASS_ID, button_predictions))))
    # normalize the attention maps
    # [query_len * RpQ, heads, num_levels, num_points], where query_len = 10
    attn_maps = get_attention_maps(outputs["attn_maps"])
    # get the spatial shapes and then visualize the attention
    spatial_shapes = outputs["spatial_shapes"]
    sampling_locations = outputs["sampling_locations"]
    attn_images = visualize_attention(image, attn_maps, reference_points, sampling_locations, button_predictions)
    fig = create_plot(attn_images)
    fig.savefig(Path(f"visualize/{image_name}_attn.png"), dpi=300)
    # now save the whole image
    img_predictions = visualize_predictions(image, button_predictions, hole_predictions)
    img_predictions.save(Path(f"visualize/{image_name}_pred.png"))

def visualize_directory(
    directory: Path,
    model,
    output_dir: Path,
    device: torch.device,
    inference_size: int,
    save_attention_maps: bool,
    min_button_score: Optional[float],
):
    image_files = iter_image_files(directory)

    if not image_files:
        raise RuntimeError(f"No image files found in: {directory}")

    for image_path in image_files:
        visualize_one(
            path=image_path,
            model=model,
            output_dir=output_dir,
            device=device,
            inference_size=inference_size,
            save_attention_maps=save_attention_maps,
            min_button_score=min_button_score,
        )


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize DETR-style PRTR button predictions and decoder attention maps."
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        default="good_run_8",
        help="Model name without extension, looked up in CHECKPOINT_DIR.",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="dataset/real",
        help="Image stem, image path, or directory.",
    )

    parser.add_argument(
        "--inference-size",
        type=int,
        default=INFERENCE_SIZE,
        help="Input resolution used at inference.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where visualizations are saved.",
    )

    parser.add_argument(
        "--attention",
        dest="attention",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save decoder attention maps. Use --no-attention to disable.",
    )

    parser.add_argument(
        "--min-button-score",
        type=float,
        default=None,
        help=(
            "Optional extra display filter. "
            "By default, every query whose argmax class is button is displayed."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    model_config_path = CHECKPOINT_DIR / f"{args.model}.json"
    model_weights_path = CHECKPOINT_DIR / f"{args.model}.pt"

    model_config_path = Path("model.json")
    model_weights_path = Path("checkpoints/last.pt")

    if not model_config_path.exists():
        raise FileNotFoundError(f"Model config not found: {model_config_path}")

    if not model_weights_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_weights_path}")

    model = load_model(
        model_config_path=model_config_path,
        model_weights_path=model_weights_path,
        device=DEVICE,
    )

    input_path = Path(args.input)

    if input_path.exists() and input_path.is_dir():
        visualize_directory(
            directory=input_path,
            model=model,
            output_dir=args.output_dir,
            device=DEVICE,
            inference_size=args.inference_size,
            save_attention_maps=args.attention,
            min_button_score=args.min_button_score,
        )
    else:
        visualize_one(
            path=args.input,
            model=model,
            output_dir=args.output_dir,
            device=DEVICE,
            inference_size=args.inference_size,
            save_attention_maps=args.attention,
            min_button_score=args.min_button_score,
        )


if __name__ == "__main__":
    main()