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
from criterion import HungarianMatcher, SetCriterion


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


def load_image(annotation_path: Path, path: Path):
    # first, get the image
    image_path = path
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    # then, get the annotations
    coordinates = []
    with open(annotation_path, "r") as file:
        json_data = json.load(file)
        buttons = json_data["buttons"]
        for b in buttons:
            center = b # ["center"]
            x = float(center["x_px"]) / float(width)
            y = 1.0 - float(center["y_px"]) / float(height)
            coordinates.append([x, y])
    # convert to torch tensors
    coordinates = torch.tensor(coordinates) # [nb_gt, 2]
    labels = torch.zeros(coordinates.size()[0], dtype=torch.long) # [nb_gt]
    # now simply but that into a list and inside a dict
    target = {
        "buttons": coordinates,
        "labels": labels
    }
    return image, target


def iter_image_files(directory: Path):
    annotation_path = directory / "annotations"
    images_path = directory / "images"
    result = []
    for annotation_file in annotation_path.glob("*.json"):
        for ext in (".png", ".jpg", ".jpeg", ".bmp", ".webp"):
            image_file = images_path / f"{annotation_file.stem}{ext}"
            if image_file.exists():
                result.append((annotation_file, image_file))
    return result


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
    pred_buttons = outputs["pred_buttons"][0].detach().cpu()    # [Q, 2]
    # transforms to probabilities
    pred_probs = pred_logits.softmax(dim=-1) # [Q, C + 1]
    pred_classes = pred_probs.argmax(dim=-1)
    pred_scores = pred_probs.max(dim=-1).values
    # now loop over the queries and get the predictions
    num_queries, num_classes = pred_logits.size()
    predictions = []
    for query_idx in range(num_queries):
        class_id = int(pred_classes[query_idx])
        confidence = float(pred_probs[query_idx, BUTTON_CLASS_ID])
        xy_norm_tensor = pred_buttons[query_idx]
        pos_x, pos_y = normalized_to_pixel_xy(xy_norm_tensor, width, height)
        predictions.append(Prediction(class_id, pos_x, pos_y, confidence))
    return predictions


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


def evaluate_one(
    annotation_path: Path,
    image_path: Path,
    model,
    device: torch.device,
    inference_size: int,
    cost_class: float,
    cost_coord: float,
):
    """
    For the evaluation, since we don't have bounding boxes,
    we will consider that a prediction is valid if it is in a 10 pixel radius.
    We also define a confidence threshold of 95%.
    """
    image, annotation = load_image(annotation_path, image_path)
    image_name = image_path.stem
    W, H = image.size
    # run the model and get the output
    outputs = run_model(model, image, device, inference_size)
    predictions = outputs["pred_buttons"][0]
    pred_probabilities = outputs["pred_logits"][0].softmax(-1)
    button_probabilities = pred_probabilities[:, 0]
    # sort by confidence
    order = torch.argsort(button_probabilities, descending=True)
    predictions = predictions[order]
    button_probabilities = button_probabilities[order]
    # now get the targets
    targets = annotation["buttons"]
    matched_targets = torch.zeros(len(targets), dtype=torch.bool)
    # now go over the annotations
    TP = 0
    FP = 0
    FN = 0
    for confidence, (x, y) in zip(button_probabilities, predictions):
        if confidence < 0.95:
            continue
        x, y = x * W, y * H
        best_i = None
        best_distance = float("inf")
        for i, (target_x, target_y) in enumerate(targets):
            target_x, target_y = target_x * W, target_y * H
            if matched_targets[i]:
                continue
            distance = math.dist((x, y), (target_x, target_y))
            if distance <= 10 and distance < best_distance:
                best_distance = distance
                best_i = i
        if best_i is not None:
            TP += 1
            matched_targets[best_i] = True
        else:
            FP += 1
    FN = (~matched_targets).sum()
    return TP, FP, FN


def evaluate_directory(
    directory: Path,
    model,
    device: torch.device,
    inference_size: int,
    cost_class: float,
    cost_coord: float,
):
    image_files = iter_image_files(directory)
    if not image_files:
        raise RuntimeError(f"No image files found in: {directory}")
    # now perform a few operation to get the losses
    # first, report the mean button loss of the buttons
    mean_button_loss = 0.0
    TPs, FPs, FNs = 0, 0, 0
    nb_image = len(image_files)
    for i, (annotation_path, image_path) in enumerate(image_files):
        TP, FP, FN = evaluate_one(
            annotation_path=annotation_path,
            image_path=image_path,
            model=model,
            device=device,
            inference_size=inference_size,
            cost_class=cost_class,
            cost_coord=cost_coord
        )
        TPs += TP
        FPs += FP
        FNs += FN
        precision = TPs / (TPs + FPs)
        recall = TPs / (TPs + FNs)
        F1 = (2 * TPs) / (2 * TPs + FPs + FNs)
        print(f"Eval[{i:<5}/{nb_image}]: precision={precision:.4f}, recall={recall:.4f}, F1={F1:.4f}", end="\r")
    # then report precision, recall and F1
    precision = TPs / (TPs + FPs)
    recall = TPs / (TPs + FNs)
    F1 = (2 * TPs) / (2 * TPs + FPs + FNs)
    print("Evaluation report:")
    print(f"- precision: {precision:.4f}")
    print(f"- recall: {recall:.4f}")
    print(f"- F1: {F1:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate DETR-style model."
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
        default="dataset_finetune",
        help="Image stem, image path, or directory.",
    )

    parser.add_argument(
        "--inference-size",
        type=int,
        default=INFERENCE_SIZE,
        help="Input resolution used at inference.",
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

    # get the class cost and coord cost
    path = Path(model_config_path)
    with open(path, "r") as file:
        data = json.load(file)
        cost_class = data["training_parameters"]["cost_class"]
        cost_coord = data["training_parameters"]["cost_coord"]

    input_path = Path(args.input)

    if input_path.exists() and input_path.is_dir():
        evaluate_directory(
            directory=input_path,
            model=model,
            device=DEVICE,
            inference_size=args.inference_size,
            cost_class=cost_class,
            cost_coord=cost_coord
        )

# # Mean button loss (mean norm 2 distance): 0.0729093998670578

if __name__ == "__main__":
    main()