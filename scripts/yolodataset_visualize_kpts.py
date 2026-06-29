#!/usr/bin/env python3
"""
Visualize YOLO-pose annotations.

Expected YOLO-pose label format:

    class x_center y_center width height kp1_x kp1_y kp2_x kp2_y ...

or with visibility:

    class x_center y_center width height kp1_x kp1_y kp1_v kp2_x kp2_y kp2_v ...

All bbox and keypoint coordinates are expected to be normalized in [0, 1].

Examples:

    python visualize_yolo_pose.py \
        --image dataset/images/train/example.jpg \
        --kpt-dim 2 \
        --show

    python visualize_yolo_pose.py \
        --image dataset/images/train/example.jpg \
        --label dataset/labels/train/example.txt \
        --kpt-dim 2 \
        --out debug.jpg

    python visualize_yolo_pose.py \
        --images-dir dataset/images/train \
        --labels-dir dataset/labels/train \
        --random \
        --kpt-dim 2 \
        --show
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass
class PoseObject:
    class_index: int
    bbox_xywh: tuple[float, float, float, float]
    keypoints: list[tuple[float, float, float | None]]


def infer_label_path(image_path: Path, labels_dir: Path | None = None) -> Path:
    """
    Infer label path from image path.

    Supports the common YOLO structure:

        dataset/images/train/foo.jpg
        dataset/labels/train/foo.txt
    """
    if labels_dir is not None:
        return labels_dir / f"{image_path.stem}.txt"

    parts = list(image_path.parts)

    for i, part in enumerate(parts):
        if part == "images":
            parts[i] = "labels"
            return Path(*parts).with_suffix(".txt")

    return image_path.with_suffix(".txt")


def parse_pose_label_file(label_path: Path, kpt_dim: int) -> list[PoseObject]:
    """
    Parse one YOLO-pose label file.

    kpt_dim:
        2 -> x y
        3 -> x y visibility
    """
    if kpt_dim not in (2, 3):
        raise ValueError(f"kpt_dim must be 2 or 3, got {kpt_dim}")

    if not label_path.exists():
        raise FileNotFoundError(f"Label file does not exist: {label_path}")

    objects: list[PoseObject] = []

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    for line_index, line in enumerate(lines, start=1):
        values = line.split()

        if len(values) < 5:
            raise ValueError(
                f"Invalid label line {line_index} in {label_path}: "
                f"expected at least 5 values, got {len(values)}"
            )

        class_index = int(float(values[0]))
        numbers = list(map(float, values[1:]))

        bbox_xywh = tuple(numbers[:4])
        raw_kpts = numbers[4:]

        if len(raw_kpts) % kpt_dim != 0:
            raise ValueError(
                f"Invalid keypoint count on line {line_index} in {label_path}. "
                f"Got {len(raw_kpts)} keypoint values, which is not divisible by "
                f"kpt_dim={kpt_dim}."
            )

        keypoints: list[tuple[float, float, float | None]] = []

        for i in range(0, len(raw_kpts), kpt_dim):
            x = raw_kpts[i]
            y = raw_kpts[i + 1]
            visibility = raw_kpts[i + 2] if kpt_dim == 3 else None
            keypoints.append((x, y, visibility))

        objects.append(
            PoseObject(
                class_index=class_index,
                bbox_xywh=bbox_xywh,  # YOLO center-x, center-y, width, height
                keypoints=keypoints,
            )
        )

    return objects


def normalized_xywh_to_xyxy(
    bbox_xywh: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> tuple[int, int, int, int]:
    """
    Convert YOLO normalized cx cy w h bbox to pixel x1 y1 x2 y2.
    """
    cx, cy, w, h = bbox_xywh

    x1 = (cx - w / 2) * image_width
    y1 = (cy - h / 2) * image_height
    x2 = (cx + w / 2) * image_width
    y2 = (cy + h / 2) * image_height

    return round(x1), round(y1), round(x2), round(y2)


def draw_pose_annotations(
    image: np.ndarray,
    objects: list[PoseObject],
    names: list[str] | None = None,
    skeleton: list[tuple[int, int]] | None = None,
    point_radius: int = 5,
    line_thickness: int = 2,
    draw_index: bool = True,
) -> np.ndarray:
    """
    Draw YOLO-pose annotations onto an image.

    image:
        OpenCV BGR image.

    skeleton:
        Optional list of zero-based keypoint connections, e.g. [(0, 1)].
    """
    output = image.copy()
    image_height, image_width = output.shape[:2]

    box_color = (0, 255, 0)
    keypoint_color = (0, 0, 255)
    skeleton_color = (255, 0, 0)
    text_color = (255, 255, 255)
    text_bg_color = (0, 0, 0)

    for obj_index, obj in enumerate(objects):
        x1, y1, x2, y2 = normalized_xywh_to_xyxy(
            obj.bbox_xywh,
            image_width=image_width,
            image_height=image_height,
        )

        cv2.rectangle(
            output,
            (x1, y1),
            (x2, y2),
            box_color,
            thickness=line_thickness,
        )

        if names is not None and 0 <= obj.class_index < len(names):
            label = names[obj.class_index]
        else:
            label = str(obj.class_index)

        label = f"{label}:{obj_index}"

        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            1,
        )

        text_x = x1
        text_y = max(0, y1 - 5)

        cv2.rectangle(
            output,
            (text_x, text_y - text_size[1] - baseline),
            (text_x + text_size[0], text_y + baseline),
            text_bg_color,
            thickness=-1,
        )

        cv2.putText(
            output,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            text_color,
            thickness=1,
            lineType=cv2.LINE_AA,
        )

        pixel_keypoints: list[tuple[int, int, bool]] = []

        for kp_index, (x_norm, y_norm, visibility) in enumerate(obj.keypoints):
            visible = True

            if visibility is not None:
                # In YOLO pose, v=0 usually means not labeled / not visible.
                visible = visibility > 0

            x_px = round(x_norm * image_width)
            y_px = round(y_norm * image_height)

            pixel_keypoints.append((x_px, y_px, visible))

            if not visible:
                continue

            cv2.circle(
                output,
                (x_px, y_px),
                point_radius,
                keypoint_color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

            if draw_index:
                cv2.putText(
                    output,
                    str(kp_index),
                    (x_px + point_radius + 2, y_px - point_radius - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    text_color,
                    thickness=1,
                    lineType=cv2.LINE_AA,
                )

        if skeleton is not None:
            for kp_a, kp_b in skeleton:
                if kp_a >= len(pixel_keypoints) or kp_b >= len(pixel_keypoints):
                    continue

                x_a, y_a, visible_a = pixel_keypoints[kp_a]
                x_b, y_b, visible_b = pixel_keypoints[kp_b]

                if not visible_a or not visible_b:
                    continue

                cv2.line(
                    output,
                    (x_a, y_a),
                    (x_b, y_b),
                    skeleton_color,
                    thickness=line_thickness,
                    lineType=cv2.LINE_AA,
                )

    return output


def parse_names(names_string: str | None) -> list[str] | None:
    if names_string is None:
        return None

    return [name.strip() for name in names_string.split(",") if name.strip()]


def parse_skeleton(skeleton_string: str | None) -> list[tuple[int, int]] | None:
    """
    Parse skeleton string like:

        "0-1,1-2,2-3"

    Keypoint indices are zero-based.
    """
    if skeleton_string is None:
        return None

    edges: list[tuple[int, int]] = []

    for item in skeleton_string.split(","):
        item = item.strip()

        if not item:
            continue

        a, b = item.split("-")
        edges.append((int(a), int(b)))

    return edges


def choose_image(
    image_path: Path | None,
    images_dir: Path | None,
    random_image: bool,
) -> Path:
    if image_path is not None:
        return image_path

    if images_dir is None:
        raise ValueError("Either --image or --images-dir must be provided.")

    image_extensions = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    image_files = [
        p for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in image_extensions
    ]

    if not image_files:
        raise FileNotFoundError(f"No image files found in: {images_dir}")

    image_files = sorted(image_files)

    if random_image:
        return random.choice(image_files)

    return image_files[0]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize one YOLO-pose image/label pair."
    )

    parser.add_argument(
        "--image",
        type=Path,
        help="Path to one image file.",
    )

    parser.add_argument(
        "--label",
        type=Path,
        help="Path to the corresponding YOLO-pose .txt label file. "
             "If omitted, the script tries to infer it.",
    )

    parser.add_argument(
        "--images-dir",
        type=Path,
        help="Directory containing images. Used when --image is omitted.",
    )

    parser.add_argument(
        "--labels-dir",
        type=Path,
        help="Directory containing labels. Used for label inference.",
    )

    parser.add_argument(
        "--random",
        action="store_true",
        help="Pick a random image from --images-dir.",
    )

    parser.add_argument(
        "--kpt-dim",
        type=int,
        choices=[2, 3],
        required=True,
        help="Keypoint dimension. Use 2 for x y, or 3 for x y visibility.",
    )

    parser.add_argument(
        "--names",
        type=str,
        default=None,
        help='Optional comma-separated class names, e.g. "button,hole".',
    )

    parser.add_argument(
        "--skeleton",
        type=str,
        default=None,
        help='Optional zero-based skeleton edges, e.g. "0-1,1-2".',
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output image path.",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show the visualization in an OpenCV window.",
    )

    parser.add_argument(
        "--no-index",
        action="store_true",
        help="Do not draw keypoint indices.",
    )

    args = parser.parse_args()

    image_path = choose_image(
        image_path=args.image,
        images_dir=args.images_dir,
        random_image=args.random,
    )

    label_path = args.label or infer_label_path(
        image_path=image_path,
        labels_dir=args.labels_dir,
    )

    image = cv2.imread(str(image_path))

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    objects = parse_pose_label_file(
        label_path=label_path,
        kpt_dim=args.kpt_dim,
    )

    names = parse_names(args.names)
    skeleton = parse_skeleton(args.skeleton)

    vis = draw_pose_annotations(
        image=image,
        objects=objects,
        names=names,
        skeleton=skeleton,
        draw_index=not args.no_index,
    )

    print(f"Image: {image_path}")
    print(f"Label: {label_path}")
    print(f"Objects: {len(objects)}")

    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.out), vis)
        print(f"Saved: {args.out}")

    if args.show:
        cv2.imshow("YOLO-pose annotations", vis)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if args.out is None and not args.show:
        default_out = image_path.with_name(f"{image_path.stem}_pose_vis.jpg")
        cv2.imwrite(str(default_out), vis)
        print(f"Saved: {default_out}")


if __name__ == "__main__":
    main()

"""
python3 scripts/yolodataset_visualize_kpts.py \
    --image /home/tomtom/Documents/DATASET_5_YOLO_KPTS/images/train/cloth_5_buttons_00000252.png \
    --label /home/tomtom/Documents/DATASET_5_YOLO_KPTS/labels/train/cloth_5_buttons_00000252.txt \
    --kpt-dim 2 \
    --names button hole \
    --skeleton "0-1" \
    --show
"""