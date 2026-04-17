from abc import ABC, abstractmethod
import random
import math
from PIL import Image, ImageDraw, ImageFilter
import numpy as np


class Transform(ABC):
    def __init__(self):
        self.current_epoch = 0

    def set_epoch(self, epoch: int):
        self.current_epoch = max(0, epoch)

    @abstractmethod
    def __call__(self, input, labels):
        return input, labels


class RandomSafeErasing(Transform):
    def __init__(
            self,
            p: float = 0.2,
            min_width: float = 0.02, max_width: float = 0.2,
            min_height: float = 0.02, max_height: float = 0.2,
            safety_radius: float = 0.04,
            max_trials: int = 20
    ):
        self.p = p
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height
        self.safety_radius = safety_radius
        self.max_trials = max_trials

    def _is_correct(self, to_avoid, x0, y0, x1, y1, W, H):
        r_x = self.safety_radius * W
        r_y = self.safety_radius * H
        for bx, by in to_avoid:
            px = bx * W
            py = by * H
            safe_x0 = px - r_x
            safe_y0 = py - r_y
            safe_x1 = px + r_x
            safe_y1 = py + r_y

            overlaps = not (
                x1 < safe_x0 or
                x0 > safe_x1 or
                y1 < safe_y0 or
                y0 > safe_y1
            )
            if overlaps:
                return False
        return True

    
    def __call__(self, image, to_avoid):
        if random.random() > self.p:
            return image, to_avoid
        W, H = image.size
        out = image.copy()
        draw = ImageDraw.Draw(out)
        how_much = random.randrange(1, 3)
        done = 0
        for _ in range(self.max_trials):
            if done >= how_much:
                break
            width = random.uniform(self.min_width, self.max_width) * W
            height = random.uniform(self.min_height, self.max_height) * H
            random_x = random.randrange(0, W - int(width))
            random_y = random.randrange(0, H - int(height))
            # coordinates
            x0 = random_x
            y0 = random_y
            x1 = random_x + width
            y1 = random_y + height
            if not self._is_correct(to_avoid, x0, y0, x1, y1, W, H):
                continue
            # else, remove the part
            draw.rectangle([x0, y0, x1, y1], fill=(0, 0, 0))
            done += 1
        return out, to_avoid


class RandomButtonErasing(Transform):
    def __init__(
        self,
        p: float = 0.25,
        min_size: float = 0.05,
        max_size: float = 0.2
    ):
        self.p = p
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, image, labels):
        if random.random() > self.p or not labels:
            return image, labels
        W, H = image.size
        randomly_choosen = random.choice(labels)
        x, y = randomly_choosen
        # then define the rectangle around that area
        width = random.uniform(self.min_size, self.max_size) * W
        height = random.uniform(self.min_size, self.max_size) * H
        center_x = x * W
        center_y = y * H
        rectangle = [center_x - width // 2, center_y - height // 2, center_x + width // 2, center_y + height // 2]
        # define the fill color
        v = random.randint(0, 160)
        fill = (v, v, v)
        # then draw the recangle
        out = image.copy()
        draw = ImageDraw.Draw(out)
        draw.rectangle(rectangle, fill=fill)
        # then update the labels
        new_labels = []
        for (x, y) in labels:
            point = (x * W, y * H)
            if rect_contains_point(point, rectangle):
                continue
            new_labels.append((x, y))
        return out, new_labels


def rect_contains_point(point, rect):
    px, py = point
    x1, y1, x2, y2 = rect
    
    if x1 > x2:
        x1, x2 = x2, x1
    if y1 > y2:
        y1, y2 = y2, y1
    if (x1 <= px <= x2) and (y1 <= py <= y2):
        return True
    else:
        return False


class ComposeWithLabels(Transform):
    def __init__(self, transformations):
        self.transformations = transformations

    def __call__(self, image, labels):
        for transformation in self.transformations:
            image, labels = transformation(image, labels)
        return image, labels
    

class ComposeWrapper(Transform):
    def __init__(self, transformation):
        self.transformation = transformation

    def __call__(self, image, labels):
        image = self.transformation(image)
        return image, labels


class RandomHorizontalFlip(Transform):
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, input, labels):
        if random.random() >= self.p:
            return input, labels
        image = input.transpose(Image.FLIP_LEFT_RIGHT)
        new_labels = []
        for (x, y) in labels:
            new_labels.append((1.0 - x, y))
        return image, new_labels
    

class RandomHorizontalTranslation(Transform):
    def __init__(self, p: float = 0.5, min: float = -0.3, max: float = 0.3):
        self.p = p
        self.min = min
        self.max = max

    def __call__(self, image, labels):
        if random.random() >= self.p:
            return image, labels
        W, H = image.size
        shift = random.uniform(self.min, self.max)
        translated = Image.new(image.mode, (W, H))
        translated.paste(image, (int(shift * W), 0))
        new_labels = []
        for (x, y) in labels:
            new_x = x + shift
            if new_x < 0.0 or new_x > 1.0:
                continue
            new_labels.append((new_x, y))
        return translated, new_labels


class RandomVerticalTranslation(Transform):
    def __init__(self, p: float = 0.5, min: float = -0.3, max: float = 0.3):
        self.p = p
        self.min = min
        self.max = max

    def __call__(self, image, labels):
        if random.random() >= self.p:
            return image, labels
        W, H = image.size
        shift = random.uniform(self.min, self.max)
        translated = Image.new(image.mode, (W, H))
        translated.paste(image, (0, int(shift * H)))
        new_labels = []
        for (x, y) in labels:
            new_y = y + shift
            if new_y < 0.0 or new_y > 1.0:
                continue
            new_labels.append((x, new_y))
        return translated, new_labels


class SaveImage(Transform):
    def __call__(self, input, labels):
        image, labels = super().__call__(input, labels)
        out = image.copy()
        draw = ImageDraw.Draw(out)
        W, H = out.size
        radius = 4
        for i, (x, y) in enumerate(labels):
            px = x * W
            py = y * H
            # draw a small circle
            draw.ellipse(
                (px - radius, py - radius, px + radius, py + radius),
                fill=(255, 0, 0),
                outline=(255, 255, 255)
            )
            # optional: draw the label index next to it
            draw.text((px + 6, py - 6), str(i), fill=(255, 0, 0))
        out.save(f"tests/{random.randrange(500)}.png")
        return image, labels
    
class RandomRotation(Transform):
    def __init__(
        self,
        p: float = 0.5,
        min_angle: float = -90.0,
        max_angle: float = 90.0,
        fill=(0, 0, 0)
    ):
        self.p = p
        self.min_angle = min_angle
        self.max_angle = max_angle
        self.fill = fill

    def __call__(self, image, labels):
        if random.random() >= self.p:
            return image, labels
        W, H = image.size
        angle = random.uniform(self.min_angle, self.max_angle)
        # rotate image around center, keep same canvas size
        rotated = image.rotate(
            angle,
            resample=Image.BILINEAR,
            expand=False,
            fillcolor=self.fill
        )
        # rotate labels around image center
        cx = W / 2.0
        cy = H / 2.0
        theta = -angle * 3.141592653589793 / 180.0
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        new_labels = []
        for x, y in labels:
            px = x * W
            py = y * H
            dx = px - cx
            dy = py - cy
            new_dx = dx * cos_t - dy * sin_t
            new_dy = dx * sin_t + dy * cos_t
            new_px = cx + new_dx
            new_py = cy + new_dy
            new_x = new_px / W
            new_y = new_py / H
            if 0.0 <= new_x <= 1.0 and 0.0 <= new_y <= 1.0:
                new_labels.append((new_x, new_y))
        return rotated, new_labels


class RandomSafeCrop(Transform):
    def __init__(
            self,
            p: float = 0.4,
            min_width: float = 0.7,
            max_width: float = 1.0,
            min_height: float = 0.7,
            max_height: float = 1.0
    ):
        self.p = p
        self.min_width = min_width
        self.max_width = max_width
        self.min_height = min_height
        self.max_height = max_height

    def __call__(self, image, labels):
        if random.random() >= self.p:
            return image, labels
        W, H = image.size
        crop_w = int(random.uniform(self.min_width, self.max_width) * W)
        crop_h = int(random.uniform(self.min_height, self.max_height) * H)
        crop_w = max(1, min(crop_w, W))
        crop_h = max(1, min(crop_h, H))
        if crop_w == W and crop_h == H:
            return image, labels
        left = random.randint(0, W - crop_w)
        top = random.randint(0, H - crop_h)
        right = left + crop_w
        bottom = top + crop_h
        cropped = image.crop((left, top, right, bottom))
        new_labels = []
        for (x, y) in labels:
            px = x * W
            py = y * H
            new_x = (px - left) / crop_w
            new_y = (py - top) / crop_h
            if new_x < 0.0 or new_x > 1.0 or new_y < 0.0 or new_y > 1.0:
                continue
            new_labels.append((new_x, new_y))
        return cropped, new_labels
    

class RandomZoomOut(Transform):
    def __init__(
            self,
            p: float = 0.5,
            min_scale: float = 0.7,
            max_scale: float = 1.0,
    ):
        self.p = p
        self.min_scale = min_scale
        self.max_scale = max_scale

    def __call__(self, image, labels):
        if random.random() >= self.p:
            return image, labels
        W, H = image.size
        scale = random.uniform(self.min_scale, self.max_scale)
        new_W = max(1, int(scale * W))
        new_H = max(1, int(scale * H))
        if new_W == W and new_H == H:
            return image, labels
        fill_grey_scale = random.randint(0, 80)
        fill = (fill_grey_scale, fill_grey_scale, fill_grey_scale)
        resized = image.resize((new_W, new_H), Image.BILINEAR)
        canvas = Image.new(image.mode, (W, H), fill)
        offset_x = random.randint(0, W - new_W)
        offset_y = random.randint(0, H - new_H)
        canvas.paste(resized, (offset_x, offset_y))
        new_labels = []
        for (x, y) in labels:
            px = x * new_W + offset_x
            py = y * new_H + offset_y
            new_x = px / W
            new_y = py / H
            new_labels.append((new_x, new_y))
        return canvas, new_labels


class RandomProgressiveFoveatedBlur(Transform):
    """
    Blur is strongest far from the labels and weakest near them.

    Labels are expected to be:
        [(x1, y1), (x2, y2), ...]
    with x, y normalized in [0, 1].

    Progressive behavior:
    - call set_epoch(epoch) from your training loop
    - the maximum blur radius will decrease over time
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
        schedule: str = "cosine",   # "linear" or "cosine"
        blur_mode: str = "box",     # "box" or "gaussian"
        smoothstep: bool = True
    ):
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

        self.current_epoch = current_epoch

        if self.fade_radius < self.keep_radius:
            raise ValueError("fade_radius must be >= keep_radius")

    def _get_progress(self):
        effective_epoch = (self.current_epoch // self.decay_every) * self.decay_every
        t = effective_epoch / float(self.total_decay_epochs)
        if t < 0.0:
            t = 0.0
        if t > 1.0:
            t = 1.0
        return t

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
        elif self.blur_mode == "gaussian":
            return image.filter(ImageFilter.GaussianBlur(radius))
        else:
            raise ValueError(f"Unknown blur_mode: {self.blur_mode}")

    def _build_blur_stack(self, image, current_max_blur_radius):
        radii = []
        blurred_arrays = []

        for i in range(self.blur_levels):
            if self.blur_levels == 1:
                t = 0.0
            else:
                t = i / float(self.blur_levels - 1)
            radius = current_max_blur_radius * t

            # avoid recomputing nearly identical blur values
            radius_key = round(radius, 3)
            if len(radii) > 0 and abs(radius_key - radii[-1]) < 1e-6:
                blurred_arrays.append(blurred_arrays[-1])
                continue

            blurred = self._blur(image, radius)
            arr = np.asarray(blurred).astype(np.float32)
            radii.append(radius_key)
            blurred_arrays.append(arr)

        stack = np.stack(blurred_arrays, axis=0)  # [L, H, W, C] or [L, H, W]
        return stack

    def _compute_distance_map(self, W, H, labels):
        xs = np.arange(W, dtype=np.float32)[None, :]
        ys = np.arange(H, dtype=np.float32)[:, None]

        min_dist_sq = np.full((H, W), np.inf, dtype=np.float32)

        for (x, y) in labels:
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

    def __call__(self, image, labels):
        if random.random() > self.p:
            return image, labels
        if not labels:
            return image, labels

        W, H = image.size
        current_max_blur_radius = self._get_current_max_blur_radius()

        if current_max_blur_radius <= 0.0:
            return image, labels

        blur_stack = self._build_blur_stack(image, current_max_blur_radius)
        distance_map = self._compute_distance_map(W, H, labels)
        level_map = self._compute_blend_position_map(distance_map, W, H)

        level0 = np.floor(level_map).astype(np.int32)
        level1 = np.clip(level0 + 1, 0, self.blur_levels - 1)
        alpha = (level_map - level0).astype(np.float32)

        yy, xx = np.indices((H, W))

        if blur_stack.ndim == 4:
            # RGB / multi-channel
            pix0 = blur_stack[level0, yy, xx]   # [H, W, C]
            pix1 = blur_stack[level1, yy, xx]   # [H, W, C]
            out = pix0 * (1.0 - alpha[..., None]) + pix1 * alpha[..., None]
            out = np.clip(out, 0, 255).astype(np.uint8)
        else:
            # grayscale
            pix0 = blur_stack[level0, yy, xx]   # [H, W]
            pix1 = blur_stack[level1, yy, xx]   # [H, W]
            out = pix0 * (1.0 - alpha) + pix1 * alpha
            out = np.clip(out, 0, 255).astype(np.uint8)

        return Image.fromarray(out), labels
    


class RandomResize(Transform):
    def __init__(self, sizes, resample=Image.BILINEAR):
        self.sizes = list(sizes)
        self.resample = resample

    def __call__(self, image, labels):
        size = random.choice(self.sizes)
        image = image.resize((size, size), self.resample)
        # labels stay unchanged because they are normalized in [0,1]
        return image, labels