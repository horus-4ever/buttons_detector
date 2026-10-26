from dataformat.dataformat import Annotation


def annotations_to_tensor(annotations: list[Annotation], device):
    targets = []
    for annotation in annotations:
        classes, coord_buttons, coord_counterparts, *_ = annotation.to_tensor(device=device)
        targets.append({
            "labels": classes,
            "buttons": coord_buttons,
            "counterparts": coord_counterparts
        })
    return targets


def run_one(model, images, padding_mask, annotations, device):
    # annotations: b * Annotation
    images = images.to(device, non_blocking=True)
    padding_mask = padding_mask.to(device, non_blocking=True)
    # now transform the targets into tensors
    targets = annotations_to_tensor(annotations, device=device)
    # inference of the batch
    outputs = model(images, padding_mask)
    return outputs, targets