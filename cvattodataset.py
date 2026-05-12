import xml.etree.ElementTree
import argparse
from pathlib import Path
import json


def parse_one_image_annotations(image_element):
    buttons = []
    for button in image_element.findall("points"):
        points = button.get("points").split(",")
        points = [float(points[0]), float(points[1])]
        buttons.append(points)
    return buttons


def normalize_buttons(buttons, image_width, image_height):
    result = []
    for i, button in enumerate(buttons):
        button_name = f"button_{i}"
        normalized_buttons = [button[0] / image_width, button[1] / image_height]
        result.append({
            "name": button_name,
            "x_ndc": normalized_buttons[0],
            "y_ndc": normalized_buttons[1],
            "x_px": button[0],
            "y_px": button[1]
        })
    return result


def convert_cvat_xml_to_dataset(cvat_xml_path, output_dir, image_dir):
    cvat_xml_path = Path(cvat_xml_path)
    output_dir = Path(output_dir)
    image_dir = Path(image_dir)
    # create the output directories (images/ and annotations/) if they don't exist
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)
    # parse the tree
    parser = xml.etree.ElementTree.parse(cvat_xml_path)
    root = parser.getroot()
    for image in root.findall("image"):
        image_name = image.get("name", "")
        image_width = int(image.get("width", "0"))
        image_height = int(image.get("height", "0"))
        buttons = parse_one_image_annotations(image)
        buttons = normalize_buttons(buttons, image_width, image_height)
        # result
        result = {
            "name": Path(image_name).stem,
            "width": image_width,
            "height": image_height,
            "buttons": buttons
        }
        # save the image to the output directory
        image_path = image_dir / image_name
        output_image_path = output_dir / "images" / image_name
        output_image_path.write_bytes(image_path.read_bytes())
        # save the annotations to the output directory
        annotation_path = output_dir / "annotations" / f"{Path(image_name).stem}.json"
        with open(annotation_path, "w") as f:
            json.dump(result, f)


if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Convert CVAT XML annotations to a dataset format.")
    args.add_argument("--cvat_xml", type=str, required=True, help="Path to the CVAT XML annotation file.")
    args.add_argument("--output_dir", type=str, required=True, help="Directory to save the converted dataset.")
    args.add_argument("--image_dir", type=str, required=True, help="Directory containing the images referenced in the CVAT XML.")
    args = args.parse_args()

    convert_cvat_xml_to_dataset(args.cvat_xml, args.output_dir, args.image_dir)