import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision import transforms

from prtr import build_model_from

from criterion import HungarianMatcher


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DATASET_ROOT = Path("dataset")
IMAGES_DIR = DATASET_ROOT / "images"
ANNOTATIONS_DIR = DATASET_ROOT / "annotations"

CHECKPOINT_DIR = Path("good_runs")
OUTPUT_DIR = Path("viz_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 1
NUM_QUERIES = 10

# Keep this the same as training unless you intentionally changed it.
INFERENCE_SIZE = 512

# Final display threshold only.
CONF_THRESHOLD = 0.9

# Must match training matcher settings.
MATCH_COST_CLASS = 1.0
MATCH_COST_COORD = 5.0


# ------------------------------------------------------------
# Model / matcher
# ------------------------------------------------------------
def load_model(model_config_path: str, model_weights_path: str, device):
    model = build_model_from(model_config_path)
    ckpt = torch.load(model_weights_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def build_matcher():
    return HungarianMatcher(
        cost_class=MATCH_COST_CLASS,
        cost_coord=MATCH_COST_COORD,
    )


# ------------------------------------------------------------
# IO helpers
# ------------------------------------------------------------
def resolve_paths(name_or_path: str):
    """
    Accepts either:
      - a dataset stem, e.g. "cloth_5_buttons_00000005"
      - a full path to an image, e.g. "dataset/real/img_01.png"

    Returns:
      image_path, annotation_path_or_none
    """
    p = Path(name_or_path)

    if p.exists():
        image_path = p

        # First try same folder / same stem json
        candidate = p.with_suffix(".json")
        if candidate.exists():
            ann_path = candidate
        else:
            # Then try dataset annotations folder
            candidate = ANNOTATIONS_DIR / f"{p.stem}.json"
            ann_path = candidate if candidate.exists() else None

        return image_path, ann_path

    image_path = IMAGES_DIR / f"{name_or_path}.png"
    ann_path = ANNOTATIONS_DIR / f"{name_or_path}.json"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not ann_path.exists():
        ann_path = None

    return image_path, ann_path


def load_image_and_annotation(name_or_path: str):
    image_path, ann_path = resolve_paths(name_or_path)

    image = Image.open(image_path).convert("RGB")
    ann = None
    if ann_path is not None and ann_path.exists():
        with open(ann_path, "r", encoding="utf-8") as f:
            ann = json.load(f)

    return image, ann, image_path


# ------------------------------------------------------------
# Preprocessing
# ------------------------------------------------------------
def preprocess_image(image: Image.Image):
    tfm = transforms.Compose([
        transforms.Resize((INFERENCE_SIZE, INFERENCE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    x = tfm(image).unsqueeze(0)  # [1, 3, H, W]
    return x


# ------------------------------------------------------------
# Annotation helpers
# ------------------------------------------------------------
def gt_buttons_to_pixels(annotation: dict):
    coords = []
    for b in annotation.get("buttons", []):
        coords.append((float(b["x_px"]), float(b["y_px"])))
    return coords


def gt_buttons_to_normalized(annotation: dict):
    width = float(annotation["width"])
    height = float(annotation["height"])

    coords = []
    for b in annotation.get("buttons", []):
        x = float(b["x_px"]) / width
        y = float(b["y_px"]) / height
        coords.append([x, y])

    if len(coords) == 0:
        return torch.zeros((0, 2), dtype=torch.float32)

    return torch.tensor(coords, dtype=torch.float32)


def prediction_to_pixels(pred_buttons: torch.Tensor, width: int, height: int):
    coords = []
    for xy in pred_buttons:
        x = float(xy[0]) * width
        y = float(xy[1]) * height
        coords.append((x, y))
    return coords


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------
@torch.no_grad()
def run_model(model, image: Image.Image, device):
    x = preprocess_image(image).to(device)
    outputs = model(x)

    pred_logits = outputs["pred_logits"][0].detach().cpu()    # [Q, C+1]
    pred_buttons = outputs["pred_buttons"][0].detach().cpu()  # [Q, 2]

    probs = pred_logits.softmax(dim=-1)
    button_scores = probs[:, 0]
    pred_classes = probs.argmax(dim=-1)

    keep = (pred_classes == 0) & (button_scores >= CONF_THRESHOLD)

    result = {
        "all_logits": pred_logits,
        "all_buttons": pred_buttons,
        "button_scores": button_scores,
        "pred_classes": pred_classes,
        "keep_mask": keep,
        "kept_logits": pred_logits[keep],
        "kept_buttons": pred_buttons[keep],
        "kept_scores": button_scores[keep],
        "raw_outputs": outputs,   # keep full batched outputs for matcher / attention
    }
    return result


# ------------------------------------------------------------
# Hungarian matching (same matcher as training)
# ------------------------------------------------------------
@torch.no_grad()
def match_all_queries_to_gt(matcher, raw_outputs, gt_buttons_norm: torch.Tensor, device):
    """
    Uses the exact same HungarianMatcher class as training.

    Returns:
        list of (pred_idx, gt_idx)
    """
    if gt_buttons_norm.numel() == 0:
        return []

    targets = [{
        "labels": torch.zeros((gt_buttons_norm.shape[0],), dtype=torch.long, device=device),
        "buttons": gt_buttons_norm.to(device),
    }]

    outputs_for_matcher = {
        "pred_logits": raw_outputs["pred_logits"],
        "pred_buttons": raw_outputs["pred_buttons"],
    }

    indices = matcher(outputs_for_matcher, targets)
    pred_idx, gt_idx = indices[0]

    return list(zip(pred_idx.tolist(), gt_idx.tolist()))


# ------------------------------------------------------------
# Drawing
# ------------------------------------------------------------
def draw_points_and_matches(
    image: Image.Image,
    gt_points=None,
    matched_pred_points=None,
    confident_pred_points=None,
    matches=None,
    matched_scores=None,
    confident_scores=None,
    gt_color="lime",
    matched_color="red",
    confident_color="cyan",
    line_color="yellow",
    radius=8,
):
    """
    matched_pred_points:
        predictions selected by Hungarian matching
    confident_pred_points:
        thresholded final predictions
    matches:
        list of (pred_idx_in_matched_pred_points, gt_idx)
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)

    # GT
    if gt_points is not None:
        for i, (x, y) in enumerate(gt_points):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         outline=gt_color, width=3)
            draw.line((x - radius, y, x + radius, y), fill=gt_color, width=2)
            draw.line((x, y - radius, x, y + radius), fill=gt_color, width=2)
            draw.text((x + 8, y + 8), f"gt{i}", fill=gt_color)

    # Confident predictions
    if confident_pred_points is not None:
        for i, (x, y) in enumerate(confident_pred_points):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         outline=confident_color, width=2)
            if confident_scores is not None:
                draw.text((x + 8, y - 18), f"{confident_scores[i]:.2f}", fill=confident_color)

    # Matched predictions
    if matched_pred_points is not None:
        for i, (x, y) in enumerate(matched_pred_points):
            draw.ellipse((x - radius, y - radius, x + radius, y + radius),
                         outline=matched_color, width=3)
            draw.text((x + 8, y - 4), f"m{i}", fill=matched_color)
            if matched_scores is not None:
                draw.text((x + 8, y + 12), f"{matched_scores[i]:.2f}", fill=matched_color)

    # Match lines
    if matches is not None and gt_points is not None and matched_pred_points is not None:
        for matched_pred_idx, gt_idx in matches:
            px, py = matched_pred_points[matched_pred_idx]
            gx, gy = gt_points[gt_idx]
            draw.line((px, py, gx, gy), fill=line_color, width=2)

    return image


# ------------------------------------------------------------
# Attention visualization
# ------------------------------------------------------------
def _normalize_attention_maps(attn_maps: torch.Tensor):
    """
    Tries to convert attention maps into shape [Q, Hf, Wf].
    """
    attn_maps = attn_maps.detach().cpu()

    if attn_maps.ndim == 3:
        # already [Q, Hf, Wf]
        return attn_maps

    if attn_maps.ndim == 4:
        # try to reduce head dimension
        # common cases:
        # [Q, heads, Hf, Wf]
        # [heads, Q, Hf, Wf]
        if attn_maps.shape[0] == NUM_QUERIES:
            return attn_maps.mean(dim=1)
        if attn_maps.shape[1] == NUM_QUERIES:
            return attn_maps.mean(dim=0)

        # fallback: average first dim
        return attn_maps.mean(dim=0)

    raise ValueError(f"Unsupported attention map shape: {tuple(attn_maps.shape)}")


def visualize_attention(image: Image.Image, model, name: str):
    if not hasattr(model.transformer.decoder, "attn_maps"):
        print("No decoder attention maps found on model.transformer.decoder.attn_maps")
        return

    if len(model.transformer.decoder.attn_maps) == 0:
        print("decoder.attn_maps is empty")
        return

    attn_maps = model.transformer.decoder.attn_maps[-1]
    # attn_maps = _normalize_attention_maps(attn_maps)  # [Q, Hf, Wf]

    width, height = image.size
    q = len(attn_maps)

    ncols = 5
    nrows = math.ceil(q / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))

    if nrows == 1 and ncols == 1:
        axes = [[axes]]
    elif nrows == 1:
        axes = [axes]
    elif ncols == 1:
        axes = [[ax] for ax in axes]

    for i in range(nrows * ncols):
        r = i // ncols
        c = i % ncols
        ax = axes[r][c]

        if i >= q:
            ax.axis("off")
            continue

        attn_map = attn_maps[i].unsqueeze(0).unsqueeze(0)  # [1,1,Hf,Wf]
        attn_map = F.interpolate(
            attn_map,
            size=(height, width),
            mode="bilinear",
            align_corners=False
        )[0, 0].numpy()

        ax.imshow(image)
        ax.imshow(attn_map, cmap="jet", alpha=0.35)
        ax.set_title(f"Query {i}")
        ax.axis("off")

    figure.suptitle("Decoder attention maps", fontsize=20)
    figure.tight_layout()

    out_path = OUTPUT_DIR / f"{name}_attention.png"
    figure.savefig(out_path, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved attention map to: {out_path}")


# ------------------------------------------------------------
# Main visualization
# ------------------------------------------------------------
def visualize_one(name_or_path: str, model, matcher, save_attention_maps: bool = True):
    image, ann, image_path = load_image_and_annotation(name_or_path)
    width, height = image.size
    name = image_path.stem

    infer = run_model(model, image, DEVICE)

    all_logits = infer["all_logits"]
    all_buttons = infer["all_buttons"]
    kept_logits = infer["kept_logits"]
    kept_buttons = infer["kept_buttons"]
    kept_scores = infer["kept_scores"]
    button_scores = infer["button_scores"]

    kept_points_px = prediction_to_pixels(kept_buttons, width, height)

    gt_points_px = None
    gt_buttons_norm = None
    training_matches = []
    matched_pred_points_px = []
    matched_scores = []

    if ann is not None:
        gt_points_px = gt_buttons_to_pixels(ann)
        gt_buttons_norm = gt_buttons_to_normalized(ann)

        training_matches = match_all_queries_to_gt(
            matcher=matcher,
            raw_outputs=infer["raw_outputs"],
            gt_buttons_norm=gt_buttons_norm,
            device=DEVICE,
        )

        # Convert matched queries to pixel coords, preserving match order
        matched_query_indices = [pred_idx for pred_idx, _ in training_matches]
        matched_pred_buttons = all_buttons[matched_query_indices]
        matched_pred_points_px = prediction_to_pixels(matched_pred_buttons, width, height)
        matched_scores = button_scores[matched_query_indices].tolist()

        # For drawing, remap matches so pred index is local to matched_pred_points_px list
        remapped_matches = [(i, gt_idx) for i, (_, gt_idx) in enumerate(training_matches)]
    else:
        remapped_matches = None

    print(f"\nImage: {image_path}")
    print(f"Confident predictions kept: {len(kept_points_px)}")

    for i, (pt, score) in enumerate(zip(kept_points_px, kept_scores.tolist())):
        print(f"  kept {i}: x={pt[0]:.1f}, y={pt[1]:.1f}, score={score:.3f}")

    if ann is not None:
        print(f"GT buttons: {len(gt_points_px)}")
        print("Training-style Hungarian matches (all queries -> GT):")
        for i, (pred_idx, gt_idx) in enumerate(training_matches):
            px = float(all_buttons[pred_idx, 0]) * width
            py = float(all_buttons[pred_idx, 1]) * height
            gx, gy = gt_points_px[gt_idx]
            score = float(button_scores[pred_idx])
            print(
                f"  match {i}: query {pred_idx} -> gt {gt_idx} | "
                f"pred=({px:.1f}, {py:.1f}) gt=({gx:.1f}, {gy:.1f}) score={score:.3f}"
            )

    vis = draw_points_and_matches(
        image=image,
        gt_points=gt_points_px,
        matched_pred_points=matched_pred_points_px if ann is not None else None,
        confident_pred_points=kept_points_px,
        matches=remapped_matches if ann is not None else None,
        matched_scores=matched_scores if ann is not None else None,
        confident_scores=kept_scores.tolist(),
    )

    out_path = OUTPUT_DIR / f"{name}_viz.png"
    vis.save(out_path)
    print(f"Saved visualization to: {out_path}")

    if save_attention_maps:
        visualize_attention(image, model, name)


def visualize_directory(directory: Path, model, matcher, save_attention_maps: bool = False):
    image_files = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
        image_files.extend(sorted(directory.glob(ext)))

    if not image_files:
        raise RuntimeError(f"No image files found in: {directory}")

    for image_path in image_files:
        visualize_one(str(image_path), model, matcher, save_attention_maps=save_attention_maps)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Visualize PRTR predictions and training-style Hungarian matches.")
    parser.add_argument("-m", "--model", type=str, default="good_run_8",
                        help="Model name without extension, looked up in CHECKPOINT_DIR")
    parser.add_argument("-i", "--input", type=str, default="dataset/real",
                        help="Image stem, image path, or directory")
    parser.add_argument("--attention", action="store_true",
                        help="Also save decoder attention maps")
    args = parser.parse_args()

    model_name = args.model
    model_config_path = CHECKPOINT_DIR / f"{model_name}.json"
    model_weights_path = CHECKPOINT_DIR / f"{model_name}.pt"

    model = load_model(str(model_config_path), str(model_weights_path), DEVICE)
    matcher = build_matcher()

    input_path = Path(args.input)
    if input_path.exists() and input_path.is_dir():
        visualize_directory(input_path, model, matcher, save_attention_maps=args.attention)
    else:
        visualize_one(args.input, model, matcher, save_attention_maps=args.attention)