""" TFRecords contain examples and examples consist of features. """
import tensorflow as tf


def int64_feature(value: int) -> tf.train.Feature:
  # create a int64 feature containing 'value'
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def float_feature(value: float) -> tf.train.Feature:
  # create a float feature containing 'value'
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def bytes_feature(value: bytes) -> tf.train.Feature:
  # create a bytes feature containing 'value'
  if not isinstance(value, list):
    value = [value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
