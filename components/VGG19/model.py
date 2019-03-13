from __future__ import print_function

import os
import numpy as np
import tensorflow as tf

from tensorpack.models.conv2d import *
from tensorpack.models.pool import *
from tensorpack.tfutils.argscope import *
from tensorpack.tfutils.sessinit import *
from tensorpack.tfutils.symbolic_functions import *

from components.path import WEIGHTS_DIR

"""
Subset of the VGG19 model with all convolution layers, trained on ImageNet
"""

VGG_MEAN = [103.939, 116.779, 123.68]


def preprocess(image):
    return image[:, :, :, ::-1] - VGG_MEAN


def postprocess(image):
    return np.round((image + VGG_MEAN)[:, :, :, ::-1], decimals=0)


def load_weights():
    param_dict = np.load(os.path.join(WEIGHTS_DIR, 'VGG19/vgg19.npz'))
    return DictRestore(dict(param_dict))


class VGG19ConvSub:
    def __init__(self, image):
        with argscope(Conv2D, kernel_shape=3, nl=tf.identity):
            self.conv1_1 = Conv2D("conv1_1", image, 64)
            self.relu1_1 = tf.nn.relu(self.conv1_1, "relu1_1")
            self.conv1_2 = Conv2D("conv1_2", self.relu1_1, 64)
            self.relu1_2 = tf.nn.relu(self.conv1_2, "relu1_2")
            self.pool1 = MaxPooling("pool1", self.relu1_2, 2)

            self.conv2_1 = Conv2D("conv2_1", self.pool1, 128)
            self.relu2_1 = tf.nn.relu(self.conv2_1, "relu2_1")
            self.conv2_2 = Conv2D("conv2_2", self.relu2_1, 128)
            self.relu2_2 = tf.nn.relu(self.conv2_2, "relu2_2")
            self.pool2 = MaxPooling("pool2", self.relu2_2, 2)

            self.conv3_1 = Conv2D("conv3_1", self.pool2, 256)
            self.relu3_1 = tf.nn.relu(self.conv3_1, "relu3_1")
            self.conv3_2 = Conv2D("conv3_2", self.relu3_1, 256)
            self.relu3_2 = tf.nn.relu(self.conv3_2, "relu3_2")
            self.conv3_3 = Conv2D("conv3_3", self.relu3_2, 256)
            self.relu3_3 = tf.nn.relu(self.conv3_3, "relu3_3")
            self.conv3_4 = Conv2D("conv3_4", self.relu3_3, 256)
            self.relu3_4 = tf.nn.relu(self.conv3_4, "relu3_4")
            self.pool3 = MaxPooling("pool3", self.relu3_4, 2)

            self.conv4_1 = Conv2D("conv4_1", self.pool3, 512)
            self.relu4_1 = tf.nn.relu(self.conv4_1, "relu4_1")
            self.conv4_2 = Conv2D("conv4_2", self.relu4_1, 512)
            self.relu4_2 = tf.nn.relu(self.conv4_2, "relu4_2")
            self.conv4_3 = Conv2D("conv4_3", self.relu4_2, 512)
            self.relu4_3 = tf.nn.relu(self.conv4_3, "relu4_3")
            self.conv4_4 = Conv2D("conv4_4", self.relu4_3, 512)
            self.relu4_4 = tf.nn.relu(self.conv4_4, "relu4_4")
            self.pool4 = MaxPooling("pool4", self.relu4_4, 2)

            self.conv5_1 = Conv2D("conv5_1", self.pool4, 512)
            self.relu5_1 = tf.nn.relu(self.conv5_1, "relu5_1")
