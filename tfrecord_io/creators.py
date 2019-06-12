""" Create examples for TFRecord files """

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import tensorflow as tf
from tfrecord_io.features import bytes_feature, float_feature, int64_feature


def create_detection_example(
        jpeg_encoded_image: bytes, image_shape: Tuple[int, int, int],
        boxes: np.ndarray, classes: List[int], classes_text: List[bytes],
        filename: str = "") -> tf.train.Example:
  """
  Create an example compatible with tensorflow's object detection api.

  Args:
    jpeg_encoded_image:
      jpeg encoded / compressed image
    image_shape:
      [height, width, channels] dimension of the image
    boxes:
      bounding boxes with shape [NumBoxes, 4] in relative coordinates and in
      [left, top, right, bottom] order
    classes:
      class id's according to labelmap
    classes_text:
      binary encoded text representation of the class
    filename:
      name of the image file on disk
  """
  h, w, c = image_shape
  boxes = np.array(boxes)
  if len(boxes):
    left, top, right, bottom = boxes.T
  else:
    left, top, right, bottom = [], [], [], []
  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'image/height': int64_feature(h),
              'image/width': int64_feature(w),
              'image/channels': int64_feature(c),
              'image/filename': bytes_feature(filename.encode("utf8")),
              'image/encoded': bytes_feature(jpeg_encoded_image),
              'image/format': bytes_feature('jpeg'.encode('utf8')),
              'image/object/bbox/xmin': float_feature(left),
              'image/object/bbox/ymin': float_feature(top),
              'image/object/bbox/xmax': float_feature(right),
              'image/object/bbox/ymax': float_feature(bottom),
              'image/object/class/text': bytes_feature(classes_text),
              'image/object/class/label': int64_feature(classes)
          }))
  return example


def create_classification_example(
        encoded_image: bytes, image_shape: Tuple[int, int, int],
        image_format: str, class_label: int,
        probabilities: List[float]) -> tf.train.Example:
  """ Create a tensorflow 'classification' Example with image and class label

  Args:
    encoded_image: encoded image file (aka compressed image (eg jpg or png))
    image_shape: [height, width, num_color_channels] of the image
    image_format: compression format of the image (eg jpeg or png)
    class_label: the class index to which the image blongs
    probabilities: probability distribution (usually one_hot encoded class)

  Returns:
    Example containing all information, ready to be serialized into a tfrecord
  """
  h, w, c = image_shape
  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              "image/height": int64_feature(h),
              "image/width": int64_feature(w),
              "image/channels": int64_feature(c),
              "image/encoded": bytes_feature(encoded_image),
              "image/format": bytes_feature(image_format.encode("utf8")),
              "image/class/label": int64_feature(class_label),
              "image/class/prob": float_feature(probabilities),
          }))
  return example


def create_image_example(encoded_image: bytes,
                         image_shape: Tuple[int, int, int],
                         image_format: str) -> tf.train.Example:
  """ Create a tensorflow 'image' Example with image only

  Args:
    encoded_image: encoded image file (aka compressed image (eg jpg or png))
    image_shape: [height, width, num_color_channels] of the image
    image_format: compression format of the image (eg jpeg or png)

  Returns:
    Example containing all information, ready to be serialized into a tfrecord
  """
  h, w, c = image_shape
  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              "image/height": int64_feature(h),
              "image/width": int64_feature(w),
              "image/channels": int64_feature(c),
              "image/encoded": bytes_feature(encoded_image),
              "image/format": bytes_feature(image_format.encode("utf8")),
          }))
  return example


def create_probability_example(classes: int,
                               probabilities: List[float]) -> tf.train.Example:
  """ Create a probability Example with only probability targets

  Args:
   classes: class index
   probabilities: list of np.ndarray
     NumClasses long list with per-class probabilities in each entry

  Returns:
     Example containing all information, ready to be serialized into a tfrecord
  """
  if isinstance(probabilities, np.ndarray):
    probabilities = probabilities.flatten().tolist()
  example = tf.train.Example(
      features=tf.train.Features(
          feature={"image/class/prob": float_feature(probabilities),
                   "image/class/label": int64_feature(classes)}))
  return example
