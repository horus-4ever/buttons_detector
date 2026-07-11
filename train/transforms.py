from dataformat.dataset import Transform
from augmentations.augmentations import *
from torchvision import transforms as T
from dataformat.dataformat import Annotation
import random


class TrainingTransform(Transform):
    def __init__(self, sizes=(512,)):
        self.sizes = sizes

    def __call__(self, image: Image.Image, annotations: Annotation):
        size = random.choice(self.sizes)
        transform = ComposeWithLabels([
            ComposeWrapper(T.Resize((size, size))),
            RandomSafeErasing(p=0.6),
            RandomFastenerPartMasking(
                p=0.3,
                target_fastener_types=("velcro",),
                min_width=0.05,
                max_width=0.20,
                min_height=0.05,
                max_height=0.2,
                max_masks=1,
                button_remove_mode="overlap",
            ),
            # RandomButtonErasing(p=0.2),
            RandomHorizontalFlip(),
            RandomHorizontalTranslation(p=0.5, min=-0.3, max=0.3),
            RandomVerticalTranslation(p=0.5, min=-0.3, max=0.3),
            # RandomRotation(p=0.5, min_angle=-45, max_angle=45),
            ComposeWrapper(T.RandomGrayscale(p=0.1)),
            ComposeWrapper(T.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
            )),
            ComposeWrapper(T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))),
            # RandomProgressiveFoveatedBlur(p=0.5, current_epoch=self.epoch),
            ComposeWrapper(T.ToTensor()),
            ComposeWrapper(T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )),
        ])
        return transform(image, annotations)


class ValidationTransform(Transform):
    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: Image.Image, annotations: Annotation):
        size = self.size
        transform = ComposeWithLabels([
            ComposeWrapper(T.Resize((size, size))),
            ComposeWrapper(T.ToTensor()),
            ComposeWrapper(T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )),
        ])
        return transform(image, annotations)
