from __future__ import print_function

import os

from six.moves import urllib
from tensorpack.models.conv2d import *
from tensorpack.models.pool import *
from tensorpack.tfutils.argscope import *
from tensorpack.tfutils.sessinit import *
from tensorpack.tfutils.symbolic_functions import *

"""
Subset of the VGG19 model with all convolution layers, trained on ImageNet
"""


def preprocess(img):
    """
    Preprocess an numpy image of shape (height, width, 3), type np.float32 and values between 0 and 255
    to the required input shape
    """
    img = img[None, :, :, :]
    VGG_MEAN = [103.939, 116.779, 123.68]
    img[:, :, :, 0] -= VGG_MEAN[2]
    img[:, :, :, 1] -= VGG_MEAN[1]
    img[:, :, :, 2] -= VGG_MEAN[0]
    return img


def load_weights(weights_filename):
    """Load VGG19 weights for a given session (download if necessary)"""
    if not os.path.isfile(weights_filename):
        urllib.request.urlretrieve("http://models.tensorpack.com/caffe/vgg19.npy", weights_filename)

    param_dict = np.load(weights_filename, encoding='latin1').item()
    return DictRestore(dict(param_dict))


class VGG19ConvSub:
    def __init__(self, name, image):
        with argscope(Conv2D, kernel_shape=3, nl=tf.identity):
            self.conv1_1 = Conv2D('conv1_1' + name, image, 64)
            self.relu1_1 = tf.nn.relu(self.conv1_1, 'relu1_1')
            self.conv1_2 = Conv2D('conv1_2' + name, self.relu1_1, 64)
            self.relu1_2 = tf.nn.relu(self.conv1_2, 'relu1_2')
            self.pool1 = MaxPooling('pool1' + name, self.relu1_2, 2)

            self.conv2_1 = Conv2D('conv2_1' + name, self.pool1, 128)
            self.relu2_1 = tf.nn.relu(self.conv2_1, 'relu2_1')
            self.conv2_2 = Conv2D('conv2_2' + name, self.relu2_1, 128)
            self.relu2_2 = tf.nn.relu(self.conv2_2, 'relu2_2')
            self.pool2 = MaxPooling('pool2' + name, self.relu2_2, 2)

            self.conv3_1 = Conv2D('conv3_1' + name, self.pool2, 256)
            self.relu3_1 = tf.nn.relu(self.conv3_1, 'relu3_1')
            self.conv3_2 = Conv2D('conv3_2' + name, self.relu3_1, 256)
            self.relu3_2 = tf.nn.relu(self.conv3_2, 'relu3_2')
            self.conv3_3 = Conv2D('conv3_3' + name, self.relu3_2, 256)
            self.relu3_3 = tf.nn.relu(self.conv3_3, 'relu3_3')
            self.conv3_4 = Conv2D('conv3_4' + name, self.relu3_3, 256)
            self.relu3_4 = tf.nn.relu(self.conv3_4, 'relu3_4')
            self.pool3 = MaxPooling('pool3' + name, self.relu3_4, 2)

            self.conv4_1 = Conv2D('conv4_1' + name, self.pool3, 512)
            self.relu4_1 = tf.nn.relu(self.conv4_1, 'relu4_1')
            self.conv4_2 = Conv2D('conv4_2' + name, self.relu4_1, 512)
            self.relu4_2 = tf.nn.relu(self.conv4_2, 'relu4_2')
            self.conv4_3 = Conv2D('conv4_3' + name, self.relu4_2, 512)
            self.relu4_3 = tf.nn.relu(self.conv4_3, 'relu4_3')
            self.conv4_4 = Conv2D('conv4_4' + name, self.relu4_3, 512)
            self.relu4_4 = tf.nn.relu(self.conv4_4, 'relu4_4')
            self.pool4 = MaxPooling('pool4' + name, self.relu4_4, 2)

            self.conv5_1 = Conv2D('conv5_1' + name, self.pool4, 512)
            self.relu5_1 = tf.nn.relu(self.conv5_1, 'relu5_1')
            self.conv5_2 = Conv2D('conv5_2' + name, self.relu5_1, 512)
            self.relu5_2 = tf.nn.relu(self.conv5_2, 'relu5_2')
            self.conv5_3 = Conv2D('conv5_3' + name, self.relu5_2, 512)
            self.relu5_3 = tf.nn.relu(self.conv5_3, 'relu5_3')
            self.conv5_4 = Conv2D('conv5_4' + name, self.relu5_3, 512)
