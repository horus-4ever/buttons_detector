import torch

def collate_fn(batch):
    """
    Padds and computes each image mask for the given batch.
    """
    images, targets = zip(*batch)
    # get the maximum image size of the batch
    max_h = max(img.shape[1] for img in images)
    max_w = max(img.shape[2] for img in images)
    # targets: list of [Annotation]

    batch_size = len(images)
    channels = images[0].shape[0]
    dtype = images[0].dtype
    # initialize the tensors that will contain the batch
    # the mask is: 0 means image data ; 1 means padded data (to be ignored)
    # [B, C, max H, max W]
    padded_images = torch.zeros((batch_size, channels, max_h, max_w), dtype=dtype)
    padding_mask = torch.ones((batch_size, max_h, max_w), dtype=torch.bool)

    new_targets = []
    for i, (image, target) in enumerate(zip(images, targets)):
        _, h, w = image.shape
        padded_images[i, :, :h, :w] = image
        padding_mask[i, :h, :w] = False
        new_targets.append(target)
    # take the new common size
    common_size = (max_w, max_h)
    return padded_images, padding_mask, new_targets, common_size