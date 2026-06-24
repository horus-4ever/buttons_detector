from ultralytics import YOLO
import argparse
from pathlib import Path


def finetune(model, dataset: Path, epochs: int, image_size: int):
    """
    Finetune the model, and freeze the YOLO backbone.
    The domain may not be that far away, as proven by Deformabl DETR experiments.
    The backbone, to have similar results, is then freezed in YOLO too.

    From the YOLO documentation:
        The freeze parameter accepts either an integer or a list.
        An integer freeze=10 freezes the first 10 layers (0 through 9,
        which corresponds to the backbone in YOLO26).
    """
    freeze = 10
    results = model.train(data=dataset, epochs=epochs, imgsz=image_size, freeze=freeze)
    return results

def load_model(model_path: Path):
    model = YOLO(model_path)
    return model

def init_parser():
    parser = argparse.ArgumentParser("Fine tune a given YOLO model.")
    parser.add_argument("--model", type=str, required=True, help="Path to the YOLO model weights to finetune.")
    parser.add_argument("--dataset", type=str, required=True, help="Path to the .yaml file of the dataset.")
    parser.add_argument("--epochs", type=int, required=False, default=50, help="Number of epochs.")
    parser.add_argument("--img-size", type=int, required=False, default=512, help="Image size for training.")
    return parser


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    # now load the model and finetune
    model_path = Path(args.model)
    model = load_model(model_path)
    # now finetune it
    dataset_path = Path(args.dataset)
    epochs = args.epochs
    image_size = args.img_size
    results = finetune(model, dataset_path, epochs, image_size)
