from abc import ABC, abstractmethod
import math
import random
from typing import Callable, List, Sequence, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


Label = Tuple[float, float]
Labels = List[Label]


def _check_paired_labels(button_labels: Sequence[Label], hole_labels: Sequence[Label]) -> None:
    if len(button_labels) != len(hole_labels):
        raise ValueError(
            "button_labels and hole_labels must have the same length. "
            f"Got {len(button_labels)} buttons and {len(hole_labels)} holes."
        )


def _is_normalized_point_visible(point: Label) -> bool:
    x, y = point
    return 0.0 <= x <= 1.0 and 0.0 <= y <= 1.0


def _keep_pairs_for_visible_buttons(
    button_labels: Sequence[Label],
    hole_labels: Sequence[Label],
    is_button_visible: Callable[[Label], bool],
) -> Tuple[Labels, Labels]:
    """
    Important rule:
    - if button[i] is visible, keep button[i] and hole[i]
    - if button[i] is not visible, drop both button[i] and hole[i]
    - hole visibility is intentionally NOT checked
    """
    _check_paired_labels(button_labels, hole_labels)

    new_button_labels: Labels = []
    new_hole_labels: Labels = []

    for button, hole in zip(button_labels, hole_labels):
        if is_button_visible(button):
            new_button_labels.append(button)
            new_hole_labels.append(hole)

    return new_button_labels, new_hole_labels


def _transform_pairs_keep_visible_buttons(
    button_labels: Sequence[Label],
    hole_labels: Sequence[Label],
    transform_point: Callable[[Label], Label],
) -> Tuple[Labels, Labels]:
    """
    Apply the same geometric transform to buttons and holes,
    but filter pairs only by transformed button visibility.
    """
    _check_paired_labels(button_labels, hole_labels)

    new_button_labels: Labels = []
    new_hole_labels: Labels = []

    for button, hole in zip(button_labels, hole_labels):
        transformed_button = transform_point(button)
        transformed_hole = transform_point(hole)

        if _is_normalized_point_visible(transformed_button):
            new_button_labels.append(transformed_button)
            new_hole_labels.append(transformed_hole)

    return new_button_labels, new_hole_labels


def rect_contains_point(point, rect) -> bool:
    px, py = point
    x1, y1, x2, y2 = rect

    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1

    return x1 <= px <= x2 and y1 <= py <= y2


def _gray_fill_for_image(image: Image.Image, value: int):
    if image.mode in ("1", "L", "I", "F"):
        return value
    if image.mode == "RGBA":
        return value, value, value, 255
    return value, value, value


class Transform(ABC):
    def __init__(self):
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = max(0, epoch)

    @abstractmethod
    def __call__(self, image, button_labels, hole_labels):
        return image, list(button_labels), list(hole_labels)


class RandomSafeErasing(Transform):
    def __init__(
        self,
        p: float = 0.2,
        min_width: float = 0.02,
        max_width: float = 0.2,
        min_height: float = 0.02,
        max_height: float = 0.2,
        safety_radius: float = 0.04,
        max_trials: int = 20,
    ):
        super().__init__()
        self.p = p
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.safety_radius = safety_radius
        self.max_trials = max_trials

    def _is_correct(self, buttons_to_avoid, x0, y0, x1, y1, W, H):
        r_x = self.safety_radius * W
        r_y = self.safety_radius * H

        for bx, by in buttons_to_avoid:
            px = bx * W
            py = by * H

            safe_x0 = px - r_x
            safe_y0 = py - r_y
            safe_x1 = px + r_x
            safe_y1 = py + r_y

            overlaps = not (
                x1 < safe_x0
                or x0 > safe_x1
                or y1 < safe_y0
                or y0 > safe_y1
            )

            if overlaps:
                return False

        return True

    def __call__(self, image, button_labels, hole_labels):
        _check_paired_labels(button_labels, hole_labels)

        if random.random() > self.p:
            return image, list(button_labels), list(hole_labels)

        W, H = image.size
        out = image.copy()
        draw = ImageDraw.Draw(out)

        how_much = random.randrange(1, 3)
        done = 0

        for _ in range(self.max_trials):
            if done >= how_much:
                break

            erase_w = max(1, int(random.uniform(self.min_width, self.max_width) * W))
            erase_h = max(1, int(random.uniform(self.min_height, self.max_height) * H))

            if erase_w >= W or erase_h >= H:
                continue

            x0 = random.randint(0, W - erase_w)
            y0 = random.randint(0, H - erase_h)
            x1 = x0 + erase_w
            y1 = y0 + erase_h

            # Safe erasing avoids buttons only.
            # Holes may be occluded and are still kept if their paired button is visible.
            if not self._is_correct(button_labels, x0, y0, x1, y1, W, H):
                continue

            draw.rectangle([x0, y0, x1, y1], fill=_gray_fill_for_image(image, 0))
            done += 1

        return out, list(button_labels), list(hole_labels)


class RandomButtonErasing(Transform):
    def __init__(
        self,
        p: float = 0.25,
        min_size: float = 0.05,
        max_size: float = 0.2,
    ):
        super().__init__()
        self.p = p
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, button_labels, hole_labels):
        _check_paired_labels(button_labels, hole_labels)

        if random.random() > self.p or not button_labels:
            return image, list(button_labels), list(hole_labels)

        W, H = image.size
        chosen_button = random.choice(button_labels)
        x, y = chosen_button

        width = random.uniform(self.min_size, self.max_size) * W
        height = random.uniform(self.min_size, self.max_size) * H

        center_x = x * W
        center_y = y * H

        rectangle = [
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        ]

        v = random.randint(0, 160)
        fill = _gray_fill_for_image(image, v)

        out = image.copy()
        draw = ImageDraw.Draw(out)
        draw.rectangle(rectangle, fill=fill)

        def is_button_visible(button: Label) -> bool:
            px = button[0] * W
            py = button[1] * H
            return not rect_contains_point((px, py), rectangle)

        new_button_labels, new_hole_labels = _keep_pairs_for_visible_buttons(
            button_labels,
            hole_labels,
            is_button_visible,
        )

        return out, new_button_labels, new_hole_labels


class ComposeWithLabels(Transform):
    def __init__(self, transformations):
        super().__init__()
        self.transformations = transformations

    def set_epoch(self, epoch: int):
        super().set_epoch(epoch)
        for transformation in self.transformations:
            if hasattr(transformation, "set_epoch"):
                transformation.set_epoch(epoch)

    def __call__(self, image, button_labels, hole_labels):
        _check_paired_labels(button_labels, hole_labels)

        button_labels = list(button_labels)
        hole_labels = list(hole_labels)

        for transformation in self.transformations:
            image, button_labels, hole_labels = transformation(
                image,
                button_labels,
                hole_labels,
            )
            _check_paired_labels(button_labels, hole_labels)

        return image, button_labels, hole_labels


class ComposeWrapper(Transform):
    def __init__(self, transformation):
        super().__init__()
        self.transformation = transformation

    def __call__(self, image, button_labels, hole_labels):
        _check_paired_labels(button_labels, hole_labels)
        image = self.transformation(image)
        return image, list(button_labels), list(hole_labels)


class RandomHorizontalFlip(Transform):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def __call__(self, image, button_labels, hole_labels):
        _check_paired_labels(button_labels, hole_labels)

        if random.random() >= self.p:
            return image, list(button_labels), list(hole_labels)

        flipped = image.transpose(Image.FLIP_LEFT_RIGHT)

        def transform_point(point: Label) -> Label:
            x, y = point
            return 1.0 - x, y

        new_button_labels, new_hole_labels = _transform_pairs_keep_visible_buttons(
            button_labels,
            hole_labels,
            transform_point,
        )

        return flipped, new_button_labels, new_hole_labels


class RandomHorizontalTranslation(Transform):
    def __init__(self, p: float = 0.5, min: float = -0.3, max: float = 0.3):
        super().__init__()
        self.p = p
        self.min = min
        self.max = max

    def __call__(self, image, button_labels, hole_labels):
        _check_paired_labels(button_labels, hole_labels)

        if random.random() >= self.p:
            return image, list(button_labels), list(hole_labels)

        W, H = image.size
        shift = random.uniform(self.min, self.max)

        translated = Image.new(image.mode, (W, H), _gray_fill_for_image(image, 0))
        translated.paste(image, (int(shift * W), 0))

        def transform_point(point: Label) -> Label:
            x, y = point
            return x + shift, y

        new_button_labels, new_hole_labels = _transform_pairs_keep_visible_buttons(
            button_labels,
            hole_labels,
            transform_point,
        )

        return translated, new_button_labels, new_hole_labels


class RandomVerticalTranslation(Transform):
    def __init__(self, p: float = 0.5, min: float = -0.3, max: float = 0.3):
        super().__init__()
        self.p = p
        self.min = min
        self.max = max

    def __call__(self, image, button_labels, hole_labels):
        _check_paired_labels(button_labels, hole_labels)

        if random.random() >= self.p:
            return image, list(button_labels), list(hole_labels)

        W, H = image.size
        shift = random.uniform(self.min, self.max)

        translated = Image.new(image.mode, (W, H), _gray_fill_for_image(image, 0))
        translated.paste(image, (0, int(shift * H)))

        def transform_point(point: Label) -> Label:
            x, y = point
            return x, y + shift

        new_button_labels, new_hole_labels = _transform_pairs_keep_visible_buttons(
            button_labels,
            hole_labels,
            transform_point,
        )

        return translated, new_button_labels, new_hole_labels


class SaveImage(Transform):
    def __init__(self, output_dir: str = "tests", radius: int = 4):
        super().__init__()
        self.output_dir = output_dir
        self.radius = radius

    def __call__(self, image, button_labels, hole_labels):
        _check_paired_labels(button_labels, hole_labels)

        out = image.copy()
        draw = ImageDraw.Draw(out)
        W, H = out.size

        for i, (button, hole) in enumerate(zip(button_labels, hole_labels)):
            bx, by = button
            hx, hy = hole

            bpx = bx * W
            bpy = by * H
            hpx = hx * W
            hpy = hy * H

            draw.ellipse(
                (
                    bpx - self.radius,
                    bpy - self.radius,
                    bpx + self.radius,
                    bpy + self.radius,
                ),
                fill=(255, 0, 0),
                outline=(255, 255, 255),
            )
            draw.text((bpx + 6, bpy - 6), f"B{i}", fill=(255, 0, 0))

            draw.ellipse(
                (
                    hpx - self.radius,
                    hpy - self.radius,
                    hpx + self.radius,
                    hpy + self.radius,
                ),
                fill=(0, 0, 255),
                outline=(255, 255, 255),
            )
            draw.text((hpx + 6, hpy - 6), f"H{i}", fill=(0, 0, 255))

        out.save(f"{self.output_dir}/{random.randrange(500)}.png")
        return image, list(button_labels), list(hole_labels)


class RandomRotation(Transform):
    def __init__(
        self,
        p: float = 0.5,
        min_angle: float = -90.0,
        max_angle: float = 90.0,
        fill=(0, 0, 0),
    ):
        super().__init__()
        self.p = p
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.fill = fill

    def __call__(self, image, button_labels, hole_labels):
        _check_paired_labels(button_labels, hole_labels)

        if random.random() >= self.p:
            return image, list(button_labels), list(hole_labels)

        W, H = image.size
        angle = random.uniform(self.min_angle, self.max_angle)

        rotated = image.rotate(
            angle,
            resample=Image.BILINEAR,
            expand=False,
            fillcolor=self.fill,
        )

        cx = W / 2.0
        cy = H / 2.0
        theta = -angle * math.pi / 180.0
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        def transform_point(point: Label) -> Label:
            x, y = point

            px = x * W
            py = y * H

            dx = px - cx
            dy = py - cy

            new_dx = dx * cos_t - dy * sin_t
            new_dy = dx * sin_t + dy * cos_t

            new_px = cx + new_dx
            new_py = cy + new_dy

            return new_px / W, new_py / H

        new_button_labels, new_hole_labels = _transform_pairs_keep_visible_buttons(
            button_labels,
            hole_labels,
            transform_point,
        )

        return rotated, new_button_labels, new_hole_labels


class RandomSafeCrop(Transform):
    def __init__(
        self,
        p: float = 0.4,
        min_width: float = 0.7,
        max_width: float = 1.0,
        min_height: float = 0.7,
        max_height: float = 1.0,
    ):
        super().__init__()
        self.p = p
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height

    def __call__(self, image, button_labels, hole_labels):
        _check_paired_labels(button_labels, hole_labels)

        if random.random() >= self.p:
            return image, list(button_labels), list(hole_labels)

        W, H = image.size

        crop_w = int(random.uniform(self.min_width, self.max_width) * W)
        crop_h = int(random.uniform(self.min_height, self.max_height) * H)

        crop_w = max(1, min(crop_w, W))
        crop_h = max(1, min(crop_h, H))

        if crop_w == W and crop_h == H:
            return image, list(button_labels), list(hole_labels)

        left = random.randint(0, W - crop_w)
        top = random.randint(0, H - crop_h)
        right = left + crop_w
        bottom = top + crop_h

        cropped = image.crop((left, top, right, bottom))

        def transform_point(point: Label) -> Label:
            x, y = point
            px = x * W
            py = y * H
            return (px - left) / crop_w, (py - top) / crop_h

        new_button_labels, new_hole_labels = _transform_pairs_keep_visible_buttons(
            button_labels,
            hole_labels,
            transform_point,
        )

        return cropped, new_button_labels, new_hole_labels


class RandomZoomOut(Transform):
    def __init__(
        self,
        p: float = 0.5,
        min_scale: float = 0.7,
        max_scale: float = 1.0,
    ):
        super().__init__()
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image, button_labels, hole_labels):
        _check_paired_labels(button_labels, hole_labels)

        if random.random() >= self.p:
            return image, list(button_labels), list(hole_labels)

        W, H = image.size
        scale = random.uniform(self.min_scale, self.max_scale)

        new_W = max(1, int(scale * W))
        new_H = max(1, int(scale * H))

        if new_W == W and new_H == H:
            return image, list(button_labels), list(hole_labels)

        fill_grey_scale = random.randint(0, 80)
        fill = _gray_fill_for_image(image, fill_grey_scale)

        resized = image.resize((new_W, new_H), Image.BILINEAR)
        canvas = Image.new(image.mode, (W, H), fill)

        offset_x = random.randint(0, W - new_W)
        offset_y = random.randint(0, H - new_H)

        canvas.paste(resized, (offset_x, offset_y))

        def transform_point(point: Label) -> Label:
            x, y = point
            px = x * new_W + offset_x
            py = y * new_H + offset_y
            return px / W, py / H

        new_button_labels, new_hole_labels = _transform_pairs_keep_visible_buttons(
            button_labels,
            hole_labels,
            transform_point,
        )

        return canvas, new_button_labels, new_hole_labels


class RandomProgressiveFoveatedBlur(Transform):
    """
    Blur is strongest far from visible labels and weakest near them.

    The returned labels are unchanged.
    Button/hole pairing is preserved.
    Hole labels outside [0, 1] are ignored for the blur map, but still kept
    if their paired button is visible.
    """

    def __init__(
        self,
        p: float = 0.3,
        current_epoch: int = 0,
        initial_max_blur_radius: float = 15.0,
        final_max_blur_radius: float = 0.0,
        blur_levels: int = 6,
        keep_radius: float = 0.015,
        fade_radius: float = 0.12,
        center_jitter: float = 0.0,
        total_decay_epochs: int = 20,
        decay_every: int = 1,
        schedule: str = "cosine",
        blur_mode: str = "box",
        smoothstep: bool = True,
    ):
        super().__init__()
        self.p = p
        self.initial_max_blur_radius = initial_max_blur_radius
        self.final_max_blur_radius = final_max_blur_radius
        self.blur_levels = max(2, blur_levels)
        self.keep_radius = keep_radius
        self.fade_radius = fade_radius
        self.center_jitter = center_jitter
        self.total_decay_epochs = max(1, total_decay_epochs)
        self.decay_every = max(1, decay_every)
        self.schedule = schedule
        self.blur_mode = blur_mode
        self.smoothstep = smoothstep
        self.current_epoch = max(0, current_epoch)

        if self.fade_radius < self.keep_radius:
            raise ValueError("fade_radius must be >= keep_radius")

    def _get_progress(self):
        effective_epoch = (self.current_epoch // self.decay_every) * self.decay_every
        t = effective_epoch / float(self.total_decay_epochs)
        return min(1.0, max(0.0, t))

    def _get_current_max_blur_radius(self):
        t = self._get_progress()

        if self.schedule == "linear":
            factor = 1.0 - t
        elif self.schedule == "cosine":
            factor = 0.5 * (1.0 + math.cos(math.pi * t))
        else:
            raise ValueError(f"Unknown schedule: {self.schedule}")

        return (
            self.final_max_blur_radius
            + (self.initial_max_blur_radius - self.final_max_blur_radius) * factor
        )

    def _blur(self, image, radius):
        if radius <= 0.0:
            return image.copy()
        if self.blur_mode == "box":
            return image.filter(ImageFilter.BoxBlur(radius))
        if self.blur_mode == "gaussian":
            return image.filter(ImageFilter.GaussianBlur(radius))
        raise ValueError(f"Unknown blur_mode: {self.blur_mode}")

    def _build_blur_stack(self, image, current_max_blur_radius):
        radii = []
        blurred_arrays = []

        for i in range(self.blur_levels):
            t = i / float(self.blur_levels - 1)
            radius = current_max_blur_radius * t

            radius_key = round(radius, 3)
            if len(radii) > 0 and abs(radius_key - radii[-1]) < 1e-6:
                blurred_arrays.append(blurred_arrays[-1])
                continue

            blurred = self._blur(image, radius)
            arr = np.asarray(blurred).astype(np.float32)
            radii.append(radius_key)
            blurred_arrays.append(arr)

        return np.stack(blurred_arrays, axis=0)

    def _compute_distance_map(self, W, H, labels):
        xs = np.arange(W, dtype=np.float32)[None, :]
        ys = np.arange(H, dtype=np.float32)[:, None]

        min_dist_sq = np.full((H, W), np.inf, dtype=np.float32)

        for x, y in labels:
            jitter_x = random.uniform(-self.center_jitter, self.center_jitter)
            jitter_y = random.uniform(-self.center_jitter, self.center_jitter)

            x = min(1.0, max(0.0, x + jitter_x))
            y = min(1.0, max(0.0, y + jitter_y))

            px = x * (W - 1)
            py = y * (H - 1)

            dist_sq = (xs - px) ** 2 + (ys - py) ** 2
            min_dist_sq = np.minimum(min_dist_sq, dist_sq)

        return np.sqrt(min_dist_sq)

    def _compute_blend_position_map(self, distance_map, W, H):
        base = float(min(W, H))
        keep_px = self.keep_radius * base
        fade_px = self.fade_radius * base

        if fade_px <= keep_px:
            fade_px = keep_px + 1.0

        t = (distance_map - keep_px) / (fade_px - keep_px)
        t = np.clip(t, 0.0, 1.0)

        if self.smoothstep:
            t = t * t * (3.0 - 2.0 * t)

        return t * float(self.blur_levels - 1)

    def __call__(self, image, button_labels, hole_labels):
        _check_paired_labels(button_labels, hole_labels)

        if random.random() > self.p:
            return image, list(button_labels), list(hole_labels)

        visible_focus_labels = list(button_labels)
        visible_focus_labels.extend(
            hole for hole in hole_labels if _is_normalized_point_visible(hole)
        )

        if not visible_focus_labels:
            return image, list(button_labels), list(hole_labels)

        W, H = image.size
        current_max_blur_radius = self._get_current_max_blur_radius()

        if current_max_blur_radius <= 0.0:
            return image, list(button_labels), list(hole_labels)

        blur_stack = self._build_blur_stack(image, current_max_blur_radius)
        distance_map = self._compute_distance_map(W, H, visible_focus_labels)
        level_map = self._compute_blend_position_map(distance_map, W, H)

        level0 = np.floor(level_map).astype(np.int32)
        level1 = np.clip(level0 + 1, 0, self.blur_levels - 1)
        alpha = (level_map - level0).astype(np.float32)

        yy, xx = np.indices((H, W))

        if blur_stack.ndim == 4:
            pix0 = blur_stack[level0, yy, xx]
            pix1 = blur_stack[level1, yy, xx]
            out = pix0 * (1.0 - alpha[..., None]) + pix1 * alpha[..., None]
        else:
            pix0 = blur_stack[level0, yy, xx]
            pix1 = blur_stack[level1, yy, xx]
            out = pix0 * (1.0 - alpha) + pix1 * alpha

        out = np.clip(out, 0, 255).astype(np.uint8)

        return Image.fromarray(out), list(button_labels), list(hole_labels)


class RandomResize(Transform):
    def __init__(self, sizes, resample=Image.BILINEAR):
        super().__init__()
        self.sizes = list(sizes)
        self.resample = resample

    def __call__(self, image, button_labels, hole_labels):
        _check_paired_labels(button_labels, hole_labels)

        size = random.choice(self.sizes)
        image = image.resize((size, size), self.resample)

        return image, list(button_labels), list(hole_labels)