"""
This tool converts annotations in XML "CVAT for images" format into the model expected format.
"""

import xml.etree.ElementTree as xml
import argparse
from pathlib import Path
import json
import dataformat as df


class CVATParser:
    def __init__(self, xml: xml.Element):
        self.xml = xml
        self._current_width = 0
        self._current_height = 0

    @classmethod
    def open(cls, xml_path: Path):
        parser = xml.parse(xml_path)
        root = parser.getroot()
        return cls(root)

    def _normalize_two(self, x1, y1, x2, y2):
        """
        Normalize two points from the current parser width and height image information.
        """
        return (
            float(x1) / self._current_width,
            float(y1) / self._current_height,
            float(x2) / self._current_width,
            float(y2) / self._current_height
        )

    def _get_image_info(self, image: xml.Element) -> df.ImageInfo:
        image_name = image.get("name")
        image_width = image.get("width")
        image_height = image.get("height")
        if image_name is None or image_width is None or image_height is None:
            raise ValueError("Aborting. One element should not be None (`image_name`, `image_width` or `image_height`)")
        return df.ImageInfo(
            url=image_name,
            width=int(image_width),
            height=int(image_height)
        )
    
    def _parse_bbox(self, cvat_box: xml.Element) -> df.BoundingBox:
        xtl, ytl, xbr, ybr = cvat_box.get("xtl"), cvat_box.get("ytl"), cvat_box.get("xbr"), cvat_box.get("ybr")
        assert xtl and ytl and xbr and ybr
        # normalize the positions
        x1, y1, x2, y2 = self._normalize_two(xtl, ytl, xbr, ybr)
        return df.BoundingBox.from_x1y1x2y2(
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2
        )
    
    def _parse_button(self, button: xml.Element) -> tuple[str, df.Button]:
        cvat_box = button
        pair_id = button.find("attribute").text
        if cvat_box is None or pair_id is None:
            raise ValueError("There should be a bounding box.")
        button_bbox = self._parse_bbox(cvat_box)
        return pair_id, df.Button(button_bbox, visible=True)
    
    def _parse_fastener(self, fastener: xml.Element) -> tuple[str, df.Fastener]:
        cvat_box = fastener
        pair_id = fastener.find("attribute").text
        if cvat_box is None or pair_id is None:
            raise ValueError("There should be a bounding box.")
        bbox = self._parse_bbox(cvat_box)
        return pair_id, df.Fastener(bbox, visible=True, type="fastener")
    
    def _parse_pairs(self, image: xml.Element) -> list[df.Pair]:
        buttons = {}
        fasteners = {}
        boxes = image.findall("box")
        buttons_boxes = list(filter(lambda box: box.get("label") == "button", boxes))
        fasteners_boxes = list(filter(lambda box: box.get("label") == "fastener", boxes))
        for button in buttons_boxes:
            pair_id, button = self._parse_button(button)
            buttons[pair_id] = button
        for fastener in fasteners_boxes:
            pair_id, fastener = self._parse_fastener(fastener)
            fasteners[pair_id] = fastener
        # since we build pairs, there should be an equal number of each
        assert len(buttons) == len(fasteners), f"There should be an equal number of buttons and fasteners.\n{buttons}\n{fasteners}"
        # now build each pair, the order doesn't matter
        pairs = []
        for pair_id in buttons:
            button, fastener = buttons[pair_id], fasteners[pair_id]
            pair = df.Pair(button, fastener)
            pairs.append(pair)
        return pairs
    
    def _parse_cloth(self, image: xml.Element) -> df.Cloth:
        pairs = self._parse_pairs(image)
        return df.Cloth(
            type="real",
            segmentation="",
            pairs=pairs
        )

    def _parse_one_image(self, image: xml.Element) -> df.Annotation:
        image_info = self._get_image_info(image)
        self._current_width = image_info.width
        self._current_height = image_info.height
        cloth = self._parse_cloth(image)
        return df.Annotation(image_info, cloth)
        
    
    def parse(self) -> tuple[bool, list[df.Annotation]]:
        annotations = []
        has_error = False
        for i, image in enumerate(self.xml.findall("image")):
            try:
                annotation = self._parse_one_image(image)
                annotations.append(annotation)
            except Exception:
                print(f"Error in image: {i}")
                has_error = True
        return has_error, annotations


def convert_cvat_xml_to_dataset(cvat_xml_path, output_dir, image_dir):
    cvat_xml_path = Path(cvat_xml_path)
    output_dir = Path(output_dir)
    image_dir = Path(image_dir)
    # create the output directories (images/ and annotations/) if they don't exist
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "annotations").mkdir(parents=True, exist_ok=True)
    # parse the tree
    parser = CVATParser.open(cvat_xml_path)
    error, annotations = parser.parse()
    if error:
        print(f"Errors detected. Abort.")
        return
    print(f"Parsed {len(annotations)}.")
    for i, annotation in enumerate(annotations):
        print(f"[{i / len(annotations):2f}%] copying files...", end="\r")
        image_name = annotation.image.url
        image_path = image_dir / image_name
        output_image_path = output_dir / "images" / image_name
        output_image_path.write_bytes(image_path.read_bytes())
        annotation_path = output_dir / "annotations" / f"{Path(image_name).stem}.json"
        with open(annotation_path, "w") as file:
            json.dump(annotation.to_json(), file)
    print("\nSuccessfully converted the annotations.")

if __name__ == "__main__":
    args = argparse.ArgumentParser(description="Convert CVAT XML annotations to a dataset format.")
    args.add_argument("--cvat_xml", type=str, required=True, help="Path to the CVAT XML annotation file.")
    args.add_argument("--output_dir", type=str, required=True, help="Directory to save the converted dataset.")
    args.add_argument("--image_dir", type=str, required=True, help="Directory containing the images referenced in the CVAT XML.")
    args = args.parse_args()

    convert_cvat_xml_to_dataset(args.cvat_xml, args.output_dir, args.image_dir)