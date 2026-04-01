from abc import ABC, abstractmethod
import random
import math
from PIL import Image, ImageDraw


class Transform(ABC):
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