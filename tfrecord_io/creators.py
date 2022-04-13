""" Create examples for TFRecord files """

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tfrecord_io.features import bytes_feature, float_feature, int64_feature
from tfrecord_io.image import encode_image


def add_image_features(image: np.ndarray, encoding_format: str = "png", **params):
    """Create a string: feature dict, containing image (meta)data

    Args:
        image: image to write into the example
        encoding_format: compression format to apply to the image (eg jpeg or png)
        params: optional named parameters passed to PIL. See for reference:
            https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
    Returns:
        feature dict
    """
    H, W, C = image.shape
    encoded_image = encode_image(image, format=encoding_format, **params)
    return {
        "image/encoded": bytes_feature(encoded_image),
        "image/format": bytes_feature(encoding_format.encode("utf8")),
        "image/height": int64_feature(H),
        "image/width": int64_feature(W),
        "image/channels": int64_feature(C),
    }


def add_classification_features(labelid: int):
    """Create a string: feature dict, containing class labels

    Args:
        labelid: the class index to which the image belongs
    Returns:
        feature dict
    """
    return {"image/class/label": int64_feature(labelid)}


def add_probability_features(probabilities: List[float]):
    """Create a string: feature dict, containing probability data

    Args:
        probabilities: list of floats or np.ndarray
            NumClasses long list with per-class probabilities in each entry
    Returns:
        feature dict
    """
    if isinstance(probabilities, np.ndarray):
        probabilities = probabilities.flatten().tolist()
    return {"image/class/prob": float_feature(probabilities)}


def add_detection_features(
    boxes: np.ndarray, labelids: List[int], class_names: List[bytes]
):
    """Create a string: feature dict, containing detection boxes (meta)data

    Args:
        probabilities: list of floats or np.ndarray
            NumClasses long list with per-class probabilities in each entry
    Returns:
        feature dict
    """
    if not isinstance(boxes, np.ndarray):
        boxes = np.array(boxes)
    if len(boxes):
        left, top, right, bottom = boxes.T
    else:
        left, top, right, bottom = [], [], [], []

    return {
        "image/object/bbox/xmin": float_feature(left),
        "image/object/bbox/ymin": float_feature(top),
        "image/object/bbox/xmax": float_feature(right),
        "image/object/bbox/ymax": float_feature(bottom),
        "image/object/class/label": int64_feature(labelids),
        "image/object/class/text": bytes_feature(class_names),
    }


def add_instance_segmentation_features(instance_masks: List[np.ndarray]):
    """Create a string: feature dict, containing instance segmentations

    Args:
        instance_masks: list of np.ndarrays
            binary masks each of size equal to the corresponding image
    Returns:
        feature dict
    """
    png_encoded_masks = [encode_image(mask) for mask in instance_masks]
    return {"image/object/mask": bytes_feature(png_encoded_masks)}


def add_segmentation_features(segmentation_mask: np.ndarray):
    """Create a string: feature dict, containing segmentations

    Args:
        segmentation_mask: np.ndarray
    Returns:
        feature dict
    """
    png_encoded_mask = encode_image(segmentation_mask)
    return {
        "image/segmentation/class/encoded": bytes_feature(png_encoded_mask),
        "image/segmentation/class/format": bytes_feature("png".encode("utf8")),
    }


def create_image_example(
    image: np.ndarray,
    encoding_format: str = "png",
    **params,
) -> tf.train.Example:
    """Create a tensorflow example containing a bare (unannotated) image

    Args:
        image: image to write into the example
        encoding_format: compression format to apply to the image (eg jpeg or png)
        params: optional named parameters passed to PIL. See for reference:
            https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
    Returns:
        example: tf.train.Example
    """
    image_features = add_image_features(image, encoding_format, **params)
    return tf.train.Example(features=tf.train.Features(feature=image_features))


def create_classification_example(
    image: np.ndarray,
    labelid: int,
    encoding_format: str = "png",
    **params,
) -> tf.train.Example:
    """Create a tensorflow 'classification' Example with image and class label

    Args:
        image: image to write into the examle
        labelid: the class index to which the image belongs
        encoding_format: compression format of the image (eg jpeg or png)
        params: optional named parameters passed to PIL. See for reference:
            https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
    Returns:
        example: tf.train.Example
    """
    image_features = add_image_features(image, encoding_format, **params)
    label_features = add_classification_features(labelid)
    features = {**image_features, **label_features}
    return tf.train.Example(features=tf.train.Features(feature=features))


def create_probability_example(
    image: np.ndarray,
    labelids: int,
    probabilities: List[float],
    encoding_format: str = "png",
    **params,
) -> tf.train.Example:
    """Create a probability Example with only probability targets

    Args:
        image: image to write into the examle
        labelids: the class index to which the image belongs
        probabilities: list of floats or np.ndarray
            NumClasses long list with per-class probabilities in each entry
        encoding_format: compression format of the image (eg jpeg or png)
        params: optional named parameters passed to PIL. See for reference:
            https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
    Returns:
        example: tf.train.Example
    """
    image_features = add_image_features(image, encoding_format, **params)
    label_features = add_classification_features(labelids)
    prob_features = add_probability_features(probabilities)
    features = {**image_features, **label_features, **prob_features}
    return tf.train.Example(features=tf.train.Features(feature=features))


def create_detection_example(
    image: np.ndarray,
    boxes: np.ndarray,
    labelids: List[int],
    class_names: List[bytes],
    encoding_format: str = "png",
    **params,
) -> tf.train.Example:
    """
    Create an example compatible with tensorflow's object detection api.

    Args:
        image: image to write into the example
        boxes: bounding boxes with shape [NumBoxes, 4] in relative coordinates
                and in [left, top, right, bottom] order
        classes: class id's according to labelmap
        classes_text: binary encoded text representation of the class
        encoding_format: compression format of the image (eg jpeg or png)
        params: optional named parameters passed to PIL. See for reference:
            https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
    Returns:
        example: tf.train.Example
    """
    image_features = add_image_features(image, encoding_format, **params)
    detection_features = add_detection_features(boxes, labelids, class_names)
    features = {**image_features, **detection_features}
    return tf.train.Example(features=tf.train.Features(feature=features))


def create_instance_segmentation_example(
    image: np.ndarray,
    boxes: np.ndarray,
    labelids: List[int],
    class_names: List[bytes],
    masks: List[np.ndarray],
    encoding_format: str = "png",
    **params,
) -> tf.train.Example:
    """
    Create an example compatible with tensorflow's object detection api.

    Args:
        image: image to write into the example
        boxes: bounding boxes with shape [NumBoxes, 4] in relative coordinates
                and in [left, top, right, bottom] order
        labelids: class id's according to labelmap
        classes_names: binary encoded text representation of the class
        masks: np.ndarrays, list of binary masks each of size equal to image
        encoding_format: compression format of the image (eg jpeg or png)
        params: optional named parameters passed to PIL. See for reference:
            https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
    Returns:
        example: tf.train.Example
    """
    image_features = add_image_features(image, encoding_format, **params)
    detection_features = add_detection_features(boxes, labelids, class_names)
    mask_features = add_instance_segmentation_features(masks)
    features = {**image_features, **detection_features, **mask_features}
    return tf.train.Example(features=tf.train.Features(feature=features))


def create_segmentation_example(
    image: np.ndarray,
    segmentation: np.ndarray,
    encoding_format: str = "png",
    **params,
) -> tf.train.Example:
    """
    Create an example compatible with tensorflow's object detection api.

    Args:
        image: image to write into the example
        segmentation: of the image assigning each pixel a labelid
        encoding_format: compression format of the image (eg jpeg or png)
        params: optional named parameters passed to PIL. See for reference:
            https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html

    Returns:
        example: tf.train.Example
    """
    image_features = add_image_features(image, encoding_format, **params)
    segmentation_features = add_segmentation_features(segmentation)
    features = {**image_features, **segmentation_features}
    return tf.train.Example(features=tf.train.Features(feature=features))


def create_classification_and_segmentation_example(
    image: np.ndarray,
    labelid: int,
    segmentation: np.ndarray,
    encoding_format: str = "png",
    **params,
) -> tf.train.Example:
    """
    Create an example compatible with tensorflow's object detection api.

    Args:
        image: image to write into the example
        labelid: the class index to which the image belongs
        segmentation: of the image assigning each pixel a labelid
        encoding_format: compression format of the image (eg jpeg or png)
        params: optional named parameters passed to PIL. See for reference:
            https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html
    Returns:
        example: tf.train.Example
    """
    image_features = add_image_features(image, encoding_format, **params)
    label_features = add_classification_features(labelid)
    segmentation_features = add_segmentation_features(segmentation)
    features = {**image_features, **label_features, **segmentation_features}
    return tf.train.Example(features=tf.train.Features(feature=features))
