""" Create examples for TFRecord files """

from pathlib import Path
from typing import Any, Dict, List, Tuple

import tensorflow as tf
from tfrecord_io.features import bytes_feature, float_feature, int64_feature


def create_detection_example(
        jpeg_encoded_image: bytes, image_shape: Tuple[int, int, int],
        boxes: Tuple[List[float], List[float], List[float], List[float]],
        classes: List[int], classes_text: List[bytes]) -> tf.train.Example:
  h, w, c = image_shape
  xmin, ymin, xmax, ymax = boxes
  example = tf.train.Example(
      features=tf.train.Features(
          feature={
              'image/height': int64_feature(h),
              'image/width': int64_feature(w),
              'image/filename': bytes_feature(fname.encode("utf8")),
              'image/encoded': bytes_feature(jpeg_encoded_image),
              'image/format': bytes_feature('jpeg'.encode('utf8')),
              'image/object/bbox/xmin': float_feature(xmin),
              'image/object/bbox/xmax': float_feature(xmax),
              'image/object/bbox/ymin': float_feature(ymin),
              'image/object/bbox/ymax': float_feature(ymax),
              'image/object/class/text': bytes_feature(classes_text),
              'image/object/class/label': int64_feature(classes)
          }))
  return example
