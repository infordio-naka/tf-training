"""
ref: http://arkouji.cocolog-nifty.com/blog/2016/08/tensorflow-76e9.html
"""

import os
import sys
import glob
import tensorflow as tf
import numpy      as np

NUM_CLASSES = 2
IMAGE_SIZE  = 28
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000

"""
input_file : csv
"imagepath","label"
ex.
/home/user/img.jpeg,0
...
/home/user/img200.jpeg,5
"""

def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.
  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.
  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 16
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])

def read_dataset(path, batch_size):
    """
    ref: http://qiita.com/knok/items/2dd15189cbca5f9890c5
    """

    height = IMAGE_SIZE
    width  = IMAGE_SIZE

    fname_queue    = tf.train.string_input_producer([path])
    reader         = tf.TextLineReader()
    key, val       = reader.read(fname_queue)
    fname, label   = tf.decode_csv(val, [["aa"], [1]])
    jpeg_r         = tf.read_file(fname)
    raw_image      = tf.image.decode_jpeg(jpeg_r, channels=3)
    #float_image    = tf.image.resize_images(raw_image, [28,28])
    reshaped_image   = tf.image.resize_images(raw_image, [28,28])

    """
    ref: https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py
    """
    distored_image = tf.random_crop(reshaped_image, [height, width, 3])
    distored_image = tf.image.random_flip_left_right(distored_image)
    distored_image = tf.image.random_brightness(distored_image, max_delta=63)
    distored_image = tf.image.random_contrast(distored_image, lower=0.2, upper=1.8)

    float_image    = tf.image.per_image_standardization(distored_image)
    
    float_image.set_shape([height, width, 3])
    float_image = tf.reshape(float_image, [IMAGE_SIZE*IMAGE_SIZE*3])

    label = tf.cast(label, tf.int64)
    min_fraction_of_examples_in_queue = 0.4
    min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                             min_fraction_of_examples_in_queue)
    
    return (_generate_image_and_label_batch(float_image, label,
                                         min_queue_examples, batch_size,
                                         shuffle=True))

if( __name__ == "__main__"):
    path = os.path.join("season.csv")
    read_dataset(path)
