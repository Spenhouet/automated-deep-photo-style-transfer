import argparse
import os

import cv2
import numpy as np
import tensorflow as tf

from utils import save_layer_activations
from vgg19 import VGG19ConvSub, load_weights, preprocess


def style_transfer(content_image, style_image, init_image, weights_path):
    """Create the VGG19 subset network"""
    content_vgg19 = VGG19ConvSub('content_vgg19', tf.constant(content_image))
    style_vgg19 = VGG19ConvSub('style_vgg19', tf.constant(style_image))
    transfer_vgg19 = VGG19ConvSub('transfer_vgg19', tf.Variable(init_image))

    restorer = load_weights(weights_path)

    """Example: run a session where multiple conv layers are calculated and the feature maps are saved"""
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restorer.init(sess)

        content_conv4_2 = sess.run(content_vgg19.conv4_2)
        style_conv1_1 = sess.run(style_vgg19.conv1_1)
        style_conv2_1 = sess.run(style_vgg19.conv2_1)
        style_conv3_1 = sess.run(style_vgg19.conv3_1)
        style_conv4_1 = sess.run(style_vgg19.conv4_1)
        style_conv5_1 = sess.run(style_vgg19.conv5_1)
        transfer_conv4_2 = sess.run(transfer_vgg19.conv4_2)

        save_layer_activations(content_conv4_2, "features/content/conv4_2_%i.png")
        save_layer_activations(style_conv1_1, "features/style/conv1_1_%i.png")
        save_layer_activations(style_conv2_1, "features/style/conv2_1_%i.png")
        save_layer_activations(style_conv3_1, "features/style/conv3_1_%i.png")
        save_layer_activations(style_conv4_1, "features/style/conv4_1_%i.png")
        save_layer_activations(style_conv5_1, "features/style/conv5_1_%i.png")
        save_layer_activations(transfer_conv4_2, "features/transfer/conv4_2_%i.png")


def load_image(filename):
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    image = preprocess(image)
    return image


if __name__ == '__main__':
    """Parse program arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_image_path", type=str, help="content image path", default="")
    parser.add_argument("--style_image_path", type=str, help="style image path", default="")
    parser.add_argument("--weights_data", type=str,
                        help="path to weights data (vgg19.npy). Download if file does not exist.", default="vgg19.npy")
    parser.add_argument("--output_image_path", type=str, help="output image path, default: result.jpg",
                        default="result.jpg")
    args = parser.parse_args()

    """Check if image files exist"""
    for path in [args.content_image_path, args.style_image_path]:
        if path is None or not os.path.isfile(path):
            print("Image file %s does not exist." % path)
            exit(0)

    content_image = load_image(args.content_image_path)
    style_image = load_image(args.style_image_path)
    init_image = np.random.randn(*content_image.shape).astype(np.float32) * 0.0001
    result = style_transfer(content_image, style_image, init_image, args.weights_data)

    # todo: write image
