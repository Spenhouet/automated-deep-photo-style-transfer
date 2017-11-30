"""
Created by sanzenba on 11/30/17
"""

from __future__ import print_function

from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *

"""
Subset of the VGG19 model with all convolutional layers, trained on ImageNet
"""


class VGG19ConvSub:
    def __init__(self, image_plhdr):
        with argscope(Conv2D, kernel_shape=3, nl=tf.identity):
            self.conv1_1 = Conv2D('conv1_1', image_plhdr, 64)
            self.relu1_1 = tf.nn.relu(self.conv1_1, 'relu1_1')
            self.conv1_2 = Conv2D('conv1_2', self.relu1_1, 64)
            self.relu1_2 = tf.nn.relu(self.conv1_2, 'relu1_2')
            self.pool1 = MaxPooling('pool1', self.relu1_2, 2)

            self.conv2_1 = Conv2D('conv2_1', self.pool1, 128)
            self.relu2_1 = tf.nn.relu(self.conv2_1, 'relu2_1')
            self.conv2_2 = Conv2D('conv2_2', self.relu2_1, 128)
            self.relu2_2 = tf.nn.relu(self.conv2_2, 'relu2_2')
            self.pool2 = MaxPooling('pool2', self.relu2_2, 2)

            self.conv3_1 = Conv2D('conv3_1', self.pool2, 256)
            self.relu3_1 = tf.nn.relu(self.conv3_1, 'relu3_1')
            self.conv3_2 = Conv2D('conv3_2', self.relu3_1, 256)
            self.relu3_2 = tf.nn.relu(self.conv3_2, 'relu3_2')
            self.conv3_3 = Conv2D('conv3_3', self.relu3_2, 256)
            self.relu3_3 = tf.nn.relu(self.conv3_3, 'relu3_3')
            self.conv3_4 = Conv2D('conv3_4', self.relu3_3, 256)
            self.relu3_4 = tf.nn.relu(self.conv3_4, 'relu3_4')
            self.pool3 = MaxPooling('pool3', self.relu3_4, 2)

            self.conv4_1 = Conv2D('conv4_1', self.pool3, 512)
            self.relu4_1 = tf.nn.relu(self.conv4_1, 'relu4_1')
            self.conv4_2 = Conv2D('conv4_2', self.relu4_1, 512)
            self.relu4_2 = tf.nn.relu(self.conv4_2, 'relu4_2')
            self.conv4_3 = Conv2D('conv4_3', self.relu4_2, 512)
            self.relu4_3 = tf.nn.relu(self.conv4_3, 'relu4_3')
            self.conv4_4 = Conv2D('conv4_4', self.relu4_3, 512)
            self.relu4_4 = tf.nn.relu(self.conv4_4, 'relu4_4')
            self.pool4 = MaxPooling('pool4', self.relu4_4, 2)

            self.conv5_1 = Conv2D('conv5_1', self.pool4, 512)
            self.relu5_1 = tf.nn.relu(self.conv5_1, 'relu5_1')
            self.conv5_2 = Conv2D('conv5_2', self.relu5_1, 512)
            self.relu5_2 = tf.nn.relu(self.conv5_2, 'relu5_2')
            self.conv5_3 = Conv2D('conv5_3', self.relu5_2, 512)
            self.relu5_3 = tf.nn.relu(self.conv5_3, 'relu5_3')
            self.conv5_4 = Conv2D('conv5_4', self.relu5_3, 512)

    # load VGGG19 weights for a given session
    def load_weights(self, weights_filename, session):
        param_dict = np.load(weights_filename, encoding='latin1').item()
        restorer = DictRestore(param_dict)
        restorer._run_init(session)

    # preprocess an numpy image of shape (height, width, 3), type np.float32 and values between 0 and 255
    # to the required input shape
    def preprocess_input(self, img):
        img = img[None, :, :, :]
        VGG_MEAN = [103.939, 116.779, 123.68]
        img[:, :, :, 0] -= VGG_MEAN[2]
        img[:, :, :, 1] -= VGG_MEAN[1]
        img[:, :, :, 2] -= VGG_MEAN[0]
        return img
