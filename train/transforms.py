from dataformat.dataset import Transform
from augmentations import Transform, ComposeWithLabels, ComposeWrapper, get_transforms
from torchvision import transforms as T
from dataformat.dataformat import Annotation
from PIL import Image



class TrainingTransform(Transform):
    def __init__(self, transforms: list[Transform], img_size: int):
        super().__init__()
        self.transforms = transforms
        self.img_size = img_size

    def __call__(self, image: Image.Image, annotations: Annotation):
        transform = ComposeWithLabels([
            ComposeWrapper(T.Resize((self.img_size, self.img_size))),
            *self.transforms,
            ComposeWrapper(T.ToTensor()),
            ComposeWrapper(T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )),
        ])
        return transform(image, annotations)


class ValidationTransform(Transform):
    def __init__(self, img_size: int):
        self.img_size = img_size

    def __call__(self, image: Image.Image, annotations: Annotation):
        transform = ComposeWithLabels([
            ComposeWrapper(T.Resize((self.img_size, self.img_size))),
            ComposeWrapper(T.ToTensor()),
            ComposeWrapper(T.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )),
        ])
        return transform(image, annotations)
