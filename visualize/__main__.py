from .gui import MainWindow
from .model import Application, Inference
from model.prtr import load_model_from_weights
from PIL import Image
import tkinter as tk
import torch
from argparse import ArgumentParser
from pathlib import Path
import time

WHICH = None

def do_cmd(args):
    global WHICH
    WHICH = "cmd"


def init_parser():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(required=True)
    gui_group = subparsers.add_parser("gui", help="GUI arguments")
    gui_group.set_defaults(func=lambda _: 0)
    cmd_group = subparsers.add_parser("cmd", help="CMD arguments")
    cmd_group.set_defaults(func=do_cmd)
    cmd_group.add_argument("--model", type=str, required=True)
    cmd_group.add_argument("--imgs", type=str, required=True)
    cmd_group.add_argument("--out", type=str, required=True)
    return parser


def iter_image_files(imgs_path: Path) -> list[Path]:
    paths = []
    for extension in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.avif"):
        paths.extend(imgs_path.glob(extension))
    return paths


def cmd_app(args):
    weights_path = Path(args.model)
    imgs_path = Path(args.imgs)
    out_path = Path(args.out)
    # load the model
    device = torch.device("cpu")
    model = load_model_from_weights(weights_path, device)
    model.eval()
    # run the inference
    inference = Inference(model, device)
    all_images = iter_image_files(imgs_path)
    start_time = time.time_ns()
    for i, image_path in enumerate(all_images):
        print(f"Image [{i:<4} / {len(all_images)}]", end="\r")
        image_name = image_path.stem
        image = Image.open(image_path)
        result = inference.run_one(image)
        result.prediction_image.save(f"{out_path / image_name}.png")
    print()
    duration = (time.time_ns() - start_time) / 1e9
    print(f"Inference ended in {duration} seconds (mean time per image: {duration / len(all_images)} second)")


if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()
    args.func(args)
    match WHICH:
        case "cmd": cmd_app(args)
        case _: exit(0)
    # create the application
    root = tk.Tk()
    root.title("Dataset Tool")
    root.geometry("800x600")
    application = Application(device=device)
    main_window = MainWindow(application, master=root)
    main_window.pack(fill="both", expand=True)
    main_window.mainloop()