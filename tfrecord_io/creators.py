""" Create examples for TFRecord files """

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tfrecord_io.features import bytes_feature, float_feature, int64_feature
from tfrecord_io.image import encode_image


def add_image_features(image: np.ndarray, format="jpeg"):
    H, W, C = image.shape
    encoded_image = encode_image(image, format=format)
    return {
        "image/encoded": bytes_feature(encoded_image),
        "image/format": bytes_feature(format.encode("utf8")),
        "image/height": int64_feature(H),
        "image/width": int64_feature(W),
        "image/channels": int64_feature(C),
    }


def add_classification_features(labelid: int):
    return {"image/class/label": int64_feature(labelid)}


def add_probability_features(probabilities):
    if isinstance(probabilities, np.ndarray):
        probabilities = probabilities.flatten().tolist()
        return {"image/class/prob": float_feature(probabilities)}


def add_detection_features(
    boxes: np.ndarray, labelids: List[int], class_names: List[bytes]
):
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
    png_encoded_masks = [encode_image(mask) for mask in instance_masks]
    return {"image/object/mask": bytes_feature(png_encoded_masks)}


def add_segmentation_features(segmentation_mask: np.ndarray):
    png_encoded_mask = encode_image(segmentation_mask)
    return {
        "image/segmentation/class/encoded": bytes_feature(png_encoded_mask),
        "image/segmentation/class/format": bytes_feature("png".encode("utf8")),
    }


def create_image_example(
    image: np.ndarray, encoding_format: str = "jpeg"
) -> tf.train.Example:
    """Create a tensorflow example containing a bare (unannotated) image

    Args:
      image: image to write into the example
      encoding_format: compression format to apply to the image (eg jpeg or png)

    Returns:
      example: tf.train.Example
    """
    image_features = add_image_features(image, format=encoding_format)
    return tf.train.Example(features=tf.train.Features(feature=image_features))


def create_classification_example(
    image: np.ndarray, labelids: int, encoding_format: str = "jpeg"
) -> tf.train.Example:
    """Create a tensorflow 'classification' Example with image and class label

    Args:
      image: image to write into the examle
      labelids: the class index to which the image belongs
      encoding_format: compression format of the image (eg jpeg or png)

    Returns:
      example: tf.train.Example
    """
    image_features = add_image_features(image, format=encoding_format)
    label_features = add_classification_features(labelids)
    features = {**image_features, **label_features}
    return tf.train.Example(features=tf.train.Features(feature=features))


def create_probability_example(
    image: np.ndarray,
    labelids: int,
    probabilities: List[float],
    encoding_format: str = "jpeg",
) -> tf.train.Example:
    """Create a probability Example with only probability targets

    Args:
     image: image to write into the examle
     labelids: the class index to which the image belongs
     probabilities: list of np.ndarray
       NumClasses long list with per-class probabilities in each entry
     encoding_format: compression format of the image (eg jpeg or png)

    Returns:
      example: tf.train.Example
    """
    image_features = add_image_features(image, format=encoding_format)
    label_features = add_classification_features(labelids)
    prob_features = add_probability_features(probabilities)
    features = {**image_features, **label_features, **prob_features}
    return tf.train.Example(features=tf.train.Features(feature=features))


def create_detection_example(
    image: np.ndarray,
    boxes: np.ndarray,
    labelids: List[int],
    class_names: List[bytes],
    encoding_format: str = "jpeg",
) -> tf.train.Example:
    """
    Create an example compatible with tensorflow's object detection api.

    Args:
      image: image to write into the example
      boxes: bounding boxes with shape [NumBoxes, 4] in relative coordinates
              and in [left, top, right, bottom] order
      classes:
        class id's according to labelmap
      classes_text:
        binary encoded text representation of the class

    Returns:
      example: tf.train.Example
    """
    image_features = add_image_features(image, format=encoding_format)
    detection_features = add_detection_features(boxes, labelids, class_names)
    features = {**image_features, **detection_features}
    return tf.train.Example(features=tf.train.Features(feature=features))


def create_instance_segmentation_example(
    image: np.ndarray,
    boxes: np.ndarray,
    labelids: List[int],
    class_names: List[bytes],
    masks: List[np.ndarray],
    encoding_format: str = "jpeg",
) -> tf.train.Example:
    """
    Create an example compatible with tensorflow's object detection api.

    Args:
      image: image to write into the example
      boxes: bounding boxes with shape [NumBoxes, 4] in relative coordinates
              and in [left, top, right, bottom] order
      classes:
        class id's according to labelmap
      classes_text:
        binary encoded text representation of the class

    Returns:
      example: tf.train.Example
    """
    image_features = add_image_features(image, format=encoding_format)
    detection_features = add_detection_features(boxes, labelids, class_names)
    mask_features = add_instance_segmentation_features(masks)
    features = {**image_features, **detection_features, **mask_features}
    return tf.train.Example(features=tf.train.Features(feature=features))


def create_segmentation_example(
    image: np.ndarray, segmentation: np.ndarray, encoding_format: str = "jpeg"
) -> tf.train.Example:
    """
    Create an example compatible with tensorflow's object detection api.

    Args:
      image: image to write into the example
      segmentation: of the image assigning each pixel a labelid

    Returns:
      example: tf.train.Example
    """
    image_features = add_image_features(image, format=encoding_format)
    segmentation_features = add_segmentation_features(segmentation)
    features = {**image_features, **segmentation_features}
    return tf.train.Example(features=tf.train.Features(feature=features))


def create_classification_and_segmentation_example(
    image: np.ndarray,
    labelid: int,
    segmentation: np.ndarray,
    encoding_format: str = "jpeg",
) -> tf.train.Example:
    """
    Create an example compatible with tensorflow's object detection api.

    Args:
      image: image to write into the example
      labelid: the class index to which the image belongs
      segmentation: of the image assigning each pixel a labelid
      encoding_format: compression format of the image (eg jpeg or png)

    Returns:
      example: tf.train.Example
    """
    image_features = add_image_features(image, format=encoding_format)
    label_features = add_classification_features(labelid)
    segmentation_features = add_segmentation_features(segmentation)
    features = {**image_features, **label_features, **segmentation_features}
    return tf.train.Example(features=tf.train.Features(feature=features))
