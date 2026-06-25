from ultralytics import YOLO
import argparse
from pathlib import Path

# Optional: Save the annotated image to disk
# results[0].save(filename="result.jpg")

def init_parser():
    parser = argparse.ArgumentParser("Display one YOLO prediction.")
    parser.add_argument("--model", type=str, required=False, default="runs/detect/train/weights/best.pt", help="Path to the model weights.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image to visualize.")
    return parser

if __name__ == "__main__":
    parser = init_parser()
    args = parser.parse_args()

    model_path = Path(args.model)
    image_path = Path(args.image)
    model = YOLO(model_path)
    results = model(image_path)
    results[0].show()