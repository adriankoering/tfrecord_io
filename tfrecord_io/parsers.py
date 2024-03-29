""" TFRecords contain serialized examples. Parsers read in these parsed
      examples and return a decoded tf.Tensor.
"""

from typing import Tuple

import tensorflow as tf


def parse_image(serialized: tf.Tensor, channels: int = 3) -> tf.Tensor:
    """Parse an image contained in 'serialized' under the key 'image/encoded'.

    Args:
        serialized: 0-d tf.Tensor with dtype=tf.string containing the serialized example
        channels: int, number of color channels in the returned image

    Returns:
        image: tf.Tensor with dtype tf.uint8 and shape [height, width, channels]
    """
    parsed_example = tf.io.parse_single_example(
        serialized, features={"image/encoded": tf.io.FixedLenFeature((), tf.string)}
    )

    image = tf.image.decode_image(parsed_example["image/encoded"], channels=channels)
    return image


def parse_detection(serialized: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
    """Parse bounding boxes and classes contained in 'serialized' under the
        key 'image/object/bbox/{xmin, xmax, ymin, ymax}' and
        'image/object/class/{text, labels}' and returns a boundingbox and
        class Tensor.

    Args:
        serialized: 0-d tf.Tensor with dtype=tf.string containing the serialized example

    Returns:
        boundingboxes: tf.Tensor with dtype tf.int32 and shape [num_boxes, 4]
        classes: tf.Tensor with dtype tf.int32 and shape [num_boxes]
        texts: tf.Tensor with dtype tf.string and shape [num_boxes]
    """
    parsed_example = tf.io.parse_single_example(
        serialized,
        features={
            "image/object/bbox/ymin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/xmin": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/ymax": tf.io.VarLenFeature(tf.float32),
            "image/object/bbox/xmax": tf.io.VarLenFeature(tf.float32),
            "image/object/class/text": tf.io.VarLenFeature(tf.string),
            "image/object/class/label": tf.io.VarLenFeature(tf.int64),
        },
    )

    ymin = tf.sparse.to_dense(parsed_example["image/object/bbox/ymin"])
    xmin = tf.sparse.to_dense(parsed_example["image/object/bbox/xmin"])
    ymax = tf.sparse.to_dense(parsed_example["image/object/bbox/ymax"])
    xmax = tf.sparse.to_dense(parsed_example["image/object/bbox/xmax"])
    bboxes = tf.stack((ymin, xmin, ymax, xmax), axis=-1)

    text = tf.sparse.to_dense(
        parsed_example["image/object/class/text"], default_value=""
    )
    label = tf.sparse.to_dense(parsed_example["image/object/class/label"])
    return bboxes, label, text


def parse_classification(serialized: tf.Tensor) -> tf.Tensor:
    """Parse a classlabel contained in 'serialized' under the key
        'image/class/label' and return it.

    Args:
        serialized: 0-d tf.Tensor with dtype=tf.string containing the serialized example
    Returns:
        class: tf.Tensor with dtype tf.int64 and shape []
    """
    parsed_example = tf.io.parse_single_example(
        serialized, features={"image/class/label": tf.io.FixedLenFeature((), tf.int64)}
    )

    label = parsed_example["image/class/label"]
    return label


def parse_probability(serialized: tf.Tensor) -> tf.Tensor:
    """Parse probabilities contained in 'serialized' under the key
        'image/class/prob' and returns them.

    Args:
        serialized: 0-d tf.Tensor with dtype=tf.string containing the serialized example
    Returns:
        probabilities: tf.Tensor with dtype tf.float and shape [num_classes]
    """
    parsed_example = tf.io.parse_single_example(
        serialized,
        features={
            "image/class/prob": tf.io.VarLenFeature(tf.float32),
        },
    )

    probabilities = tf.sparse.to_dense(parsed_example["image/class/prob"])
    return probabilities


def parse_segmentation(serialized: tf.Tensor) -> tf.Tensor:
    """Parse segmentation contained in 'serialized' under the key
         'image/segmentation/class/encoded' and returns it.

    Args:
        serialized: 0-d tf.Tensor with dtype=tf.string containing the serialized example
    Returns:
        segmentation: tf.Tensor with dtype tf.int64 and shape [H, W, 1]
    """
    key = "image/segmentation/class/encoded"
    parsed_example = tf.io.parse_single_example(
        serialized, features={key: tf.io.FixedLenFeature((), tf.string)}
    )

    segmentation = tf.image.decode_png(parsed_example[key])
    return segmentation


def parse_instance_segmentation(serialized: tf.Tensor) -> tf.Tensor:
    """Parse segmentation contained in 'serialized' under the key
         'image/object/mask' and returns it.

    Args:
        serialized: 0-d tf.Tensor with dtype=tf.string containing the serialized example
    Returns:
        instance_segmentations: tf.Tensor with dtype tf.int64 and shape [N, H, W, 1]
    """
    key = "image/object/mask"
    parsed_example = tf.io.parse_single_example(
        serialized,
        features={
            key: tf.io.VarLenFeature(tf.string),
        },
    )

    instance_segmentations = tf.map_fn(
        tf.image.decode_png, parsed_example[key], dtype=tf.uint8
    )
    return instance_segmentations
