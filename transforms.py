from abc import ABC, abstractmethod
import random
from PIL import Image, ImageDraw


class Transform(ABC):
    @abstractmethod
    def __call__(self, input, labels):
        return input, labels


class RandomSafeErasing(Transform):
    def __init__(
            self,
            p: float = 0.3,
            min_width: float = 0.02, max_width: float = 0.5,
            min_height: float = 0.02, max_height: float = 0.5,
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
        how_much = random.randrange(2, 5)
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
            if x < 0.0 or x > 1.0:
                continue
            new_labels.append((new_x, y))
        return translated, new_labels


class SaveImage(Transform):
    def __call__(self, input, labels):
        result = super().__call__(input, labels)
        input.save("out.png")
        return result