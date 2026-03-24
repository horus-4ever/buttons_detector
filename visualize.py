import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from scipy.optimize import linear_sum_assignment
from torchvision import transforms

from prtr import PRTR


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
DATASET_ROOT = Path("dataset")
IMAGES_DIR = DATASET_ROOT / "images"
ANNOTATIONS_DIR = DATASET_ROOT / "annotations"

CHECKPOINT_PATH = Path("checkpoints/best.pt")
OUTPUT_DIR = Path("viz_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

NUM_CLASSES = 1      # only "button"
NUM_QUERIES = 10
CONF_THRESHOLD = 0.7

# Use the same relative importance as in training if possible
MATCH_COST_CLASS = 1.0
MATCH_COST_COORD = 5.0


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def load_model(checkpoint_path: Path, num_classes: int, num_queries: int, device):
    model = PRTR(num_classes=num_classes, num_queries=num_queries)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def load_image_and_annotation(name: str):
    img_path = IMAGES_DIR / f"{name}.png"
    ann_path = ANNOTATIONS_DIR / f"{name}.json"

    if Path(name).exists():
        img_path = Path(name)
        ann_path = Path(name.replace(".png", ".json"))

    image = Image.open(img_path).convert("RGB")
    with open(ann_path, "r", encoding="utf-8") as f:
        ann = json.load(f)

    return image, ann


def preprocess_image(image: Image.Image):
    tfm = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    x = tfm(image).unsqueeze(0)   # [1, 3, H, W]
    return x


def prediction_to_pixels(pred_buttons, width: int, height: int):
    """
    pred_buttons: [N, 2] normalized in [0,1]
    returns pixel coords [(x_px, y_px), ...]
    """
    coords = []
    for xy in pred_buttons:
        x = float(xy[0]) * width
        y = float(xy[1]) * height
        coords.append((x, y))
    return coords


def gt_buttons_to_pixels(annotation: dict):
    coords = []
    for b in annotation.get("buttons", []):
        coords.append((float(b["x_px"]), float(b["y_px"])))
    return coords


def gt_buttons_to_normalized(annotation: dict):
    """
    Returns GT buttons in the same normalized image convention
    as pred_buttons: x in [0,1], y in [0,1] from top to bottom.
    """
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


def draw_points_and_matches(
    image: Image.Image,
    pred_points,
    gt_points=None,
    matches=None,
    pred_color="red",
    gt_color="lime",
    line_color="yellow",
    radius=8
):
    """
    pred_points: list of (x, y)
    gt_points: list of (x, y)
    matches: list of (pred_idx, gt_idx)
    """
    image = image.copy()
    draw = ImageDraw.Draw(image)

    # Draw GT first
    if gt_points is not None:
        for x, y in gt_points:
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=gt_color, width=3)
            draw.line((x - radius, y, x + radius, y), fill=gt_color, width=2)
            draw.line((x, y - radius, x, y + radius), fill=gt_color, width=2)

    # Draw prediction points
    for x, y in pred_points:
        draw.ellipse((x - radius, y - radius, x + radius, y + radius), outline=pred_color, width=3)

    # Draw assignment lines
    if matches is not None and gt_points is not None:
        for pred_idx, gt_idx in matches:
            px, py = pred_points[pred_idx]
            gx, gy = gt_points[gt_idx]
            draw.line((px, py, gx, gy), fill=line_color, width=2)

    return image


# ------------------------------------------------------------
# Hungarian matching for visualization
# ------------------------------------------------------------
def match_predictions_to_gt(
    pred_logits: torch.Tensor,
    pred_buttons: torch.Tensor,
    gt_buttons_norm: torch.Tensor,
    cost_class: float = 1.0,
    cost_coord: float = 5.0,
):
    """
    pred_logits:  [N, 2] or [N, num_classes+1]
    pred_buttons: [N, 2] normalized
    gt_buttons_norm: [M, 2] normalized

    Returns:
        matches: list of (pred_idx, gt_idx)
    """
    if pred_buttons.numel() == 0 or gt_buttons_norm.numel() == 0:
        return []

    probs = pred_logits.softmax(dim=-1)   # [N, C+1]

    # Only one real class: button = class 0
    tgt_labels = torch.zeros((gt_buttons_norm.shape[0],), dtype=torch.long, device=pred_logits.device)

    # cost_class: [N, M]
    cost_class_mat = -probs[:, tgt_labels]

    # cost_coord: [N, M]
    cost_coord_mat = torch.cdist(pred_buttons, gt_buttons_norm.to(pred_buttons.device), p=1)

    C = cost_class * cost_class_mat + cost_coord * cost_coord_mat
    C = C.detach().cpu().numpy()

    pred_ind, gt_ind = linear_sum_assignment(C)
    return list(zip(pred_ind.tolist(), gt_ind.tolist()))


# ------------------------------------------------------------
# Inference
# ------------------------------------------------------------
@torch.no_grad()
def run_inference(model, image: Image.Image, device, conf_threshold=0.7):
    x = preprocess_image(image).to(device)

    outputs = model(x)
    pred_logits = outputs["pred_logits"][0]     # [Q, num_classes+1]
    pred_buttons = outputs["pred_buttons"][0]   # [Q, 2]

    probs = pred_logits.softmax(dim=-1)

    # class 0 = button, class 1 = no-object
    button_scores = probs[:, 0]
    pred_classes = probs.argmax(dim=-1)

    # keep predictions that:
    # 1) predict class 0
    # 2) have enough confidence
    keep = (pred_classes == 0) & (button_scores >= conf_threshold)

    kept_logits = pred_logits[keep].cpu()
    kept_scores = button_scores[keep].cpu()
    kept_buttons = pred_buttons[keep].cpu()

    return kept_logits, kept_buttons, kept_scores


def visualize_one(name: str):
    model = load_model(CHECKPOINT_PATH, NUM_CLASSES, NUM_QUERIES, DEVICE)

    image, ann = load_image_and_annotation(name)
    width, height = image.size

    pred_logits, pred_buttons, pred_scores = run_inference(model, image, DEVICE, CONF_THRESHOLD)

    gt_points_px = gt_buttons_to_pixels(ann)
    gt_buttons_norm = gt_buttons_to_normalized(ann)

    pred_points_px = prediction_to_pixels(pred_buttons, width, height)

    matches = match_predictions_to_gt(
        pred_logits=pred_logits,
        pred_buttons=pred_buttons,
        gt_buttons_norm=gt_buttons_norm,
        cost_class=MATCH_COST_CLASS,
        cost_coord=MATCH_COST_COORD,
    )

    print(f"Image: {name}")
    print(f"GT buttons: {len(gt_points_px)}")
    print(f"Predicted buttons kept: {len(pred_points_px)}")
    for i, (pt, score) in enumerate(zip(pred_points_px, pred_scores.tolist())):
        print(f"  pred {i}: x={pt[0]:.1f}, y={pt[1]:.1f}, score={score:.3f}")

    print("Matches (pred_idx -> gt_idx):")
    for pred_idx, gt_idx in matches:
        px, py = pred_points_px[pred_idx]
        gx, gy = gt_points_px[gt_idx]
        print(
            f"  pred {pred_idx} -> gt {gt_idx} | "
            f"pred=({px:.1f}, {py:.1f}) gt=({gx:.1f}, {gy:.1f})"
        )

    vis = draw_points_and_matches(
        image,
        pred_points=pred_points_px,
        gt_points=gt_points_px,
        matches=matches
    )

    if Path(name).exists():
        name = Path(name).stem
    out_path = OUTPUT_DIR / f"{name}_viz.png"
    vis.save(out_path)
    print(f"Saved visualization to: {out_path}")


def visualize_random_sample():
    ann_files = sorted(ANNOTATIONS_DIR.glob("*.json"))
    if not ann_files:
        raise RuntimeError("No annotation files found")

    name = ann_files[0].stem
    visualize_one(name)


if __name__ == "__main__":
    visualize_one(f"dataset/images/cloth_5_buttons_00000000.png")
    visualize_one(f"cloth_5_buttons_00000002")