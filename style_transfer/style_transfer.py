import argparse
import os

import cv2
import numpy as np
import tensorflow as tf

import utils
from vgg19 import VGG19ConvSub


def style_transfer(content_image, style_image, weights_path):
    input_shape = content_image.shape

    # input placeholder
    image_plhdr = tf.placeholder(tf.float32, shape=(1,) + input_shape)

    # create the VGG19 subset network
    net = VGG19ConvSub(image_plhdr)

    # normalize and reshape to input shape
    content_input = net.preprocess_input(content_image)

    # example: run a session where the output for layer conv2_1 is predicted and the feature maps are saved
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        net.load_weights(weights_path, sess)
        activations = sess.run(net.conv2_1, {image_plhdr: content_input})
        utils.dump_layer_activation(activations, "features/conv2_1_%i.png")


def load_image(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    return image

if __name__ == '__main__':

    # parse program arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_image_path", type=str, help="content image path")
    parser.add_argument("--style_image_path", type=str, help="style image path")
    parser.add_argument("--weights_data", type=str,
                        help="path to weights data (vgg19.npy). Download if file does not exist.", default="vgg19.npy")
    parser.add_argument("--output_image_path", type=str, help="output image path, default: result.jpg",
                        default="result.jpg")
    args = parser.parse_args()

    # check if file exists for all parsed filenames
    for path in [args.content_image_path, args.style_image_path]:
        if not os.path.isfile(path):
            print("File %s does not exist." % path)
            exit(0)

    if not os.path.isfile(args.weights_data):
        # todo: download weights data (and do not exit)
        exit(0)

    content_image = load_image(args.content_image_path)
    style_image = load_image(args.style_image_path)
    result = style_transfer(content_image, style_image, args.weights_data)

    # todo: write image
