import numpy as np
import cv2


def image_transform(
    image_name,
    input_shape=(304, 304),
    mean=None,
    std=None,
):
    image = cv2.imread(image_name).astype(np.float32)  # uint8 to float32
    # image = cv2.resize(image, input_shape, interpolation=cv2.INTER_CUBIC)
    # Normalization
    if mean is not None:
        mean = np.array(mean, dtype=np.float32)  # BGR
        image -= mean[None, None, :]
    if std is not None:
        std = np.array(std, dtype=np.float32)  # BGR
        image /= std[None, None, :]
    image = image[:, :, ::-1].transpose([2, 0, 1])  # BGR -> RGB, change to C x H x W.

    return image


def vis_seg(img, seg, palette, alpha=0.5):
    """
    Visualize segmentation as an overlay on the image.
    Takes:
        img: H x W x 3 image in [0, 255]
        seg: H x W segmentation image of class IDs
        palette: K x 3 colormap for all classes
        alpha: opacity of the segmentation in [0, 1]
    Gives:
        H x W x 3 image with overlaid segmentation
    """
    vis = np.array(img, dtype=np.float32)
    mask = seg > 0
    vis[mask] *= 1. - alpha
    vis[mask] += alpha * palette[seg[mask].flat]
    vis = vis.astype(np.uint8)
    return vis


def make_palette(num_classes):
    """
    Maps classes to colors in the style of PASCAL VOC.
    Close values are mapped to far colors for segmentation visualization.
    See http://host.robots.ox.ac.uk/pascal/VOC/voc2012/index.html#devkit
    Takes:
        num_classes: the number of classes
    Gives:
        palette: the colormap as a k x 3 array of RGB colors
    """
    palette = np.zeros((num_classes, 3), dtype=np.uint8)
    for k in range(0, num_classes):
        label = k
        i = 0
        while label:
            palette[k, 0] |= (((label >> 0) & 1) << (7 - i))
            palette[k, 1] |= (((label >> 1) & 1) << (7 - i))
            palette[k, 2] |= (((label >> 2) & 1) << (7 - i))
            label >>= 3
            i += 1
    return palette
