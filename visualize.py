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

from prtr import build_model_from


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

DATASET_ROOT = Path("dataset")
IMAGES_DIR = DATASET_ROOT / "images"

CHECKPOINT_DIR = Path("good_runs")
OUTPUT_DIR = Path("viz_outputs")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INFERENCE_SIZE = 512

# DETR-style class layout:
#   class 0 = button
#   final class = no-object
BUTTON_CLASS_ID = 0


# ------------------------------------------------------------
# Data structures
# ------------------------------------------------------------

@dataclass(frozen=True)
class QueryPrediction:
    query_idx: int
    class_id: int
    class_score: float
    button_score: float
    xy_norm: tuple[float, float]
    xy_px: tuple[float, float]

    @property
    def is_button(self) -> bool:
        return self.class_id == BUTTON_CLASS_ID


@dataclass(frozen=True)
class InferenceResult:
    predictions: list[QueryPrediction]
    pred_logits: torch.Tensor
    pred_buttons: torch.Tensor
    pred_probs: torch.Tensor
    attn_maps: Optional[torch.Tensor | list | tuple]
    raw_outputs: dict


# ------------------------------------------------------------
# Model loading
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Image IO
# ------------------------------------------------------------

def resolve_image_path(name_or_path: str | Path) -> Path:
    p = Path(name_or_path)

    if p.exists():
        if not p.is_file():
            raise FileNotFoundError(f"Expected an image file, got directory: {p}")
        return p

    image_path = IMAGES_DIR / f"{name_or_path}.png"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    return image_path


def load_image(name_or_path: str | Path) -> tuple[Image.Image, Path]:
    image_path = resolve_image_path(name_or_path)
    image = Image.open(image_path).convert("RGB")
    return image, image_path


def iter_image_files(directory: Path) -> list[Path]:
    image_files: list[Path] = []

    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        image_files.extend(sorted(directory.glob(ext)))

    return sorted(image_files)


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# Prediction helpers
# ------------------------------------------------------------

def normalized_to_pixel_xy(xy: torch.Tensor, width: int, height: int) -> tuple[float, float]:
    x = float(xy[0]) * width
    y = float(xy[1]) * height
    return x, y


@torch.no_grad()
def run_model(
    model,
    image: Image.Image,
    device: torch.device,
    inference_size: int,
) -> InferenceResult:
    width, height = image.size

    x = preprocess_image(image, inference_size).to(device)
    outputs = model(x)

    if "pred_logits" not in outputs:
        raise KeyError("Model output is missing 'pred_logits'.")

    if "pred_buttons" not in outputs:
        raise KeyError("Model output is missing 'pred_buttons'.")

    pred_logits = outputs["pred_logits"][0].detach().cpu()      # [Q, C + 1]
    pred_buttons = outputs["pred_buttons"][0].detach().cpu()    # [Q, 2]

    pred_probs = pred_logits.softmax(dim=-1)
    pred_classes = pred_probs.argmax(dim=-1)
    pred_scores = pred_probs.max(dim=-1).values

    predictions: list[QueryPrediction] = []

    for query_idx in range(pred_logits.shape[0]):
        class_id = int(pred_classes[query_idx])
        class_score = float(pred_scores[query_idx])
        button_score = float(pred_probs[query_idx, BUTTON_CLASS_ID])

        xy_norm_tensor = pred_buttons[query_idx]
        xy_px = normalized_to_pixel_xy(xy_norm_tensor, width, height)

        predictions.append(
            QueryPrediction(
                query_idx=query_idx,
                class_id=class_id,
                class_score=class_score,
                button_score=button_score,
                xy_norm=(float(xy_norm_tensor[0]), float(xy_norm_tensor[1])),
                xy_px=xy_px,
            )
        )

    return InferenceResult(
        predictions=predictions,
        pred_logits=pred_logits,
        pred_buttons=pred_buttons,
        pred_probs=pred_probs,
        attn_maps=outputs.get("attn_maps"),
        raw_outputs=outputs,
    )


def selected_button_predictions(
    predictions: Iterable[QueryPrediction],
    min_button_score: Optional[float] = None,
) -> list[QueryPrediction]:
    selected = [p for p in predictions if p.is_button]

    if min_button_score is not None:
        selected = [p for p in selected if p.button_score >= min_button_score]

    return selected


# ------------------------------------------------------------
# Drawing
# ------------------------------------------------------------

def draw_button_predictions(
    image: Image.Image,
    predictions: list[QueryPrediction],
    radius: int = 8,
    color: str = "red",
) -> Image.Image:
    image = image.copy()
    draw = ImageDraw.Draw(image)

    for pred in predictions:
        x, y = pred.xy_px

        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            outline=color,
            width=3,
        )

        # draw.line((x - radius, y, x + radius, y), fill=color, width=2)
        # draw.line((x, y - radius, x, y + radius), fill=color, width=2)

        label = f"q{pred.query_idx} {pred.button_score:.2f}"
        draw.text((x + 8, y + 8), label, fill=color)

    return image


# ------------------------------------------------------------
# Attention map handling
# ------------------------------------------------------------

def _last_tensor(x):
    """
    Handles attention outputs that may be tensors, lists, or tuples.
    Uses the final decoder layer when the model returns a stack/list.
    """
    while isinstance(x, (list, tuple)):
        if len(x) == 0:
            raise ValueError("Attention map list/tuple is empty.")
        x = x[-1]

    if not isinstance(x, torch.Tensor):
        raise TypeError(f"Unsupported attention map type: {type(x)}")

    return x.detach().cpu()


def _reshape_flat_attention(attn: torch.Tensor, num_queries: int) -> torch.Tensor:
    """
    Converts flattened attention [Q, S] into [Q, H, W] when S is square.
    """
    if attn.ndim != 2:
        raise ValueError(f"Expected flattened attention with shape [Q, S], got {tuple(attn.shape)}.")

    if attn.shape[0] != num_queries:
        if attn.shape[1] == num_queries:
            attn = attn.T
        else:
            raise ValueError(
                f"Cannot identify query dimension in flattened attention shape {tuple(attn.shape)}."
            )

    spatial_tokens = attn.shape[1]
    side = math.isqrt(spatial_tokens)

    if side * side != spatial_tokens:
        raise ValueError(
            f"Flattened attention has {spatial_tokens} spatial tokens, which is not square. "
            f"Cannot infer [H, W] automatically."
        )

    return attn.reshape(num_queries, side, side)


def normalize_attention_maps(attn_maps, num_queries: int, batch_idx: int = 0) -> torch.Tensor:
    """
    Converts common attention shapes into [Q, Hf, Wf].

    Supported examples:
      [Q, Hf, Wf]
      [Q, heads, Hf, Wf]
      [heads, Q, Hf, Wf]
      [B, Q, Hf, Wf]
      [B, heads, Q, Hf, Wf]
      [B, Q, heads, Hf, Wf]
      [L, B, heads, Q, Hf, Wf]
      [L, B, Q, heads, Hf, Wf]
      [Q, S] where S is square
    """
    attn = _last_tensor(attn_maps)

    # [L, B, heads, Q, H, W] or [L, B, Q, heads, H, W]
    if attn.ndim == 6:
        attn = attn[-1]  # final layer -> [B, heads, Q, H, W] or [B, Q, heads, H, W]

    # [B, heads, Q, H, W] or [B, Q, heads, H, W]
    if attn.ndim == 5:
        if attn.shape[0] != num_queries:
            attn = attn[min(batch_idx, attn.shape[0] - 1)]
        else:
            # Rare case: [Q, something, something, H, W].
            # Average the second dimension if needed below.
            pass

    # [Q, heads, H, W], [heads, Q, H, W], or [B, Q, H, W]
    if attn.ndim == 4:
        if attn.shape[0] == num_queries:
            # [Q, heads, H, W]
            attn = attn.mean(dim=1)
        elif attn.shape[1] == num_queries:
            # [heads, Q, H, W] or [B, Q, H, W]
            attn = attn.mean(dim=0)
        else:
            # Last fallback: average leading dimension.
            attn = attn.mean(dim=0)

    # [Q, H, W]
    if attn.ndim == 3:
        if attn.shape[0] != num_queries:
            raise ValueError(
                f"Expected normalized attention shape [Q, H, W] with Q={num_queries}, "
                f"got {tuple(attn.shape)}."
            )
        return attn

    # [Q, S]
    if attn.ndim == 2:
        return _reshape_flat_attention(attn, num_queries)

    raise ValueError(f"Unsupported attention map shape: {tuple(attn.shape)}.")


def minmax_normalize_map(attn_map: torch.Tensor) -> torch.Tensor:
    attn_map = attn_map.float()

    min_val = attn_map.min()
    max_val = attn_map.max()

    denom = max_val - min_val
    if float(denom) < 1e-8:
        return torch.zeros_like(attn_map)

    return (attn_map - min_val) / denom


def visualize_attention(
    image: Image.Image,
    attn_maps,
    predictions: list[QueryPrediction],
    output_path: Path,
):
    num_queries = len(predictions)
    query_maps = normalize_attention_maps(attn_maps, num_queries=num_queries)  # [Q, Hf, Wf]

    width, height = image.size

    button_query_ids = {p.query_idx for p in predictions if p.is_button}
    prediction_by_query = {p.query_idx: p for p in predictions}

    ncols = 5
    nrows = math.ceil(num_queries / ncols)

    figure, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4 * ncols, 4 * nrows),
        squeeze=False,
    )

    grayscale_image = image.convert("L")

    for i in range(nrows * ncols):
        row = i // ncols
        col = i % ncols
        ax = axes[row][col]

        if i >= num_queries:
            ax.axis("off")
            continue

        pred = prediction_by_query[i]
        is_button_query = i in button_query_ids

        attn_map = query_maps[i].unsqueeze(0).unsqueeze(0)

        attn_map = F.interpolate(
            attn_map,
            size=(height, width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]

        attn_map = minmax_normalize_map(attn_map).numpy()

        if is_button_query:
            ax.imshow(image)
            ax.imshow(attn_map, cmap="jet", alpha=0.35)

            x, y = pred.xy_px
            ax.scatter([x], [y], marker="x", s=120, linewidths=4, color="#00ff00")

            title = f"Query {i} | BUTTON | p={pred.button_score:.2f}"
        else:
            ax.imshow(grayscale_image, cmap="gray")
            ax.imshow(attn_map, cmap="gray", alpha=0.45)

            title = f"Query {i} | class {pred.class_id} | unused"

        ax.set_title(title)
        ax.axis("off")

    figure.suptitle("Decoder attention maps", fontsize=20)
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


# ------------------------------------------------------------
# Main visualization
# ------------------------------------------------------------

def visualize_one(
    name_or_path: str | Path,
    model,
    output_dir: Path,
    device: torch.device,
    inference_size: int,
    save_attention_maps: bool = True,
    min_button_score: Optional[float] = None,
):
    image, image_path = load_image(name_or_path)
    name = image_path.stem

    infer = run_model(
        model=model,
        image=image,
        device=device,
        inference_size=inference_size,
    )

    button_predictions = selected_button_predictions(
        infer.predictions,
        min_button_score=min_button_score,
    )

    print(f"\nImage: {image_path}")
    print(f"Predicted button queries: {len(button_predictions)}")

    for pred in button_predictions:
        x_px, y_px = pred.xy_px
        x_norm, y_norm = pred.xy_norm

        print(
            f"  query {pred.query_idx}: "
            f"x={x_px:.1f}, y={y_px:.1f} | "
            f"x_norm={x_norm:.4f}, y_norm={y_norm:.4f} | "
            f"button_score={pred.button_score:.3f}"
        )

    vis = draw_button_predictions(
        image=image,
        predictions=button_predictions,
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    vis_path = output_dir / f"{name}_predictions.png"
    vis.save(vis_path)
    print(f"Saved prediction visualization to: {vis_path}")

    if save_attention_maps:
        if infer.attn_maps is None:
            print("No attention maps found in model output; skipping attention visualization.")
        else:
            attn_path = output_dir / f"{name}_attention.png"
            visualize_attention(
                image=image,
                attn_maps=infer.attn_maps,
                predictions=infer.predictions,
                output_path=attn_path,
            )
            print(f"Saved attention visualization to: {attn_path}")


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
            name_or_path=image_path,
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
            name_or_path=args.input,
            model=model,
            output_dir=args.output_dir,
            device=DEVICE,
            inference_size=args.inference_size,
            save_attention_maps=args.attention,
            min_button_score=args.min_button_score,
        )


if __name__ == "__main__":
    main()