from ultralytics.data.utils import visualize_image_annotations

label_map = {  # Define the label map with all annotated class labels.
    0: "button",
}

# Visualize
visualize_image_annotations(
    "/home/tomtom/Documents/DATASET_5_YOLO_DETECTION/images/train/cloth_5_buttons_00000252.png",  # Input image path.
    "/home/tomtom/Documents/DATASET_5_YOLO_DETECTION/labels/train/cloth_5_buttons_00000252.txt",  # Annotation file path for the image.
    label_map,
)