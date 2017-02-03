"""
ref: http://arkouji.cocolog-nifty.com/blog/2016/08/tensorflow-76e9.html
"""

import os
import sys
import glob
import tensorflow as tf
import numpy      as np
import cv2

NUM_CLASSES = 2
IMAGE_SIZE  = 24

"""
input_file : csv
"imagepath","label"
ex.
/home/user/img.jpeg,0
...
/home/user/img200.jpeg,5
"""

def getdirs(path):
    dirs = []
    for item in os.listdir(path):
        imgpath = os.path.join(path, item)
        if (os.path.isdir(imgpath)):
            dirs.append(imgpath)
    return (dirs)

def getfile(path):
    file = os.path.join(path,'image.label')
    return (file)

def read_dataset(path):
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
    resape_image   = tf.image.resize_images(images, [28,28])

    """
    ref: https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_input.py
    """
    distored_image = tf.random_crop(reshaped_image, [height, width, 3])
    distored_image = tf.image.random_flip_left_right(distored_image)
    distored_image = tf.image.random_brightness(distored_image, max_delta=63)
    distored_image = tf.image.random_contrast(distored_image, lower=0.2, upper=1.8)

    float_image    = tf.image.per_image_standardization(distored_image)
    
    float_image.set_shape([height, width, 3])
    
    return(float_image, label)

if( __name__ == "__main__"):
    path = os.path.join("season.csv")
    read_dataset(path)
