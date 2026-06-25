from ultralytics import YOLO
import argparse
from pathlib import Path

# Optional: Save the annotated image to disk
# results[0].save(filename="result.jpg")

def init_parser():
    parser = argparse.ArgumentParser("Display one YOLO prediction.")
    parser.add_argument("--model", type=str, required=False, default="runs/detect/train/weights/best.pt", help="Path to the model weights.")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder to visualize.")
    parser.add_argument("--out", type=str, required=False, default="viz_outputs", help="Path to the output folder.")
    return parser

def visualize_directory(model, directory: Path, out_dir: Path):
    for image in directory.glob("*.png"):
        results = model(image)
        # save the predictions
        filename = f"{image.stem}__yolo.jpg"
        out_path = out_dir / filename
        results[0].save(filename=out_path)

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    model_path = Path(args.model)
    folder = Path(args.folder)
    out_folder = Path(args.out)
    model = YOLO(model_path)
    visualize_directory(model, folder, out_folder)