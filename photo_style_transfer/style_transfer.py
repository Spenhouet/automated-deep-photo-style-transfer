import argparse
import os

import numpy as np
import tensorflow as tf
from PIL import Image

from photo_style_transfer.vgg19 import VGG19ConvSub, load_weights, VGG_MEAN


def style_transfer(content_image, style_image, init_image, weights_path):
    """Create the VGG19 subset network"""
    content_vgg19 = VGG19ConvSub('content_vgg19', content_image)
    style_vgg19 = VGG19ConvSub('style_vgg19', style_image)
    transfer_vgg19 = VGG19ConvSub('transfer_vgg19', init_image)

    restorer = load_weights(weights_path)

    content_loss = calculate_layer_content_loss(content_vgg19.conv4_2, transfer_vgg19.conv4_2)

    style_loss = (1. / 5.) * calculate_layer_style_loss(style_vgg19.conv1_1, transfer_vgg19.conv1_1)
    style_loss += (1. / 5.) * calculate_layer_style_loss(style_vgg19.conv2_1, transfer_vgg19.conv2_1)
    style_loss += (1. / 5.) * calculate_layer_style_loss(style_vgg19.conv3_1, transfer_vgg19.conv3_1)
    style_loss += (1. / 5.) * calculate_layer_style_loss(style_vgg19.conv4_1, transfer_vgg19.conv4_1)
    style_loss += (1. / 5.) * calculate_layer_style_loss(style_vgg19.conv5_1, transfer_vgg19.conv5_1)

    content_loss = 1e-3 * content_loss
    style_loss = 1 * style_loss
    total_loss = content_loss + style_loss

    optimizer = tf.train.AdamOptimizer(learning_rate=1.0, beta1=0.9, beta2=0.999, epsilon=1e-08)
    gradient = optimizer.compute_gradients(total_loss, [init_image])
    train_op = optimizer.apply_gradients(gradient)

    """Example: run a session where multiple conv layers are calculated and the feature maps are saved"""
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        restorer.init(sess)
        min_loss, best_image = float("inf"), None
        for i in range(1, 1000):
            _, result_image, loss, c_loss, s_loss = sess.run(
                [train_op, init_image, total_loss, content_loss, style_loss])

            print("Iteration {0}, Loss {1}, Content loss {2}, Style loss {3}".format(i, loss, c_loss, s_loss))

            if loss < min_loss:
                min_loss, best_image = loss, result_image

            if i % 10 == 0:
                save_image(best_image, "transfer/res_{}.png".format(i))

        """
        content_conv4_2 = sess.run(content_vgg19.conv4_2)
        transfer_conv4_2 = sess.run(transfer_vgg19.conv4_2)

        style_conv1_1 = sess.run(style_vgg19.conv1_1)
        style_conv2_1 = sess.run(style_vgg19.conv2_1)
        style_conv3_1 = sess.run(style_vgg19.conv3_1)
        style_conv4_1 = sess.run(style_vgg19.conv4_1)
        style_conv5_1 = sess.run(style_vgg19.conv5_1)
        transfer_conv1_1 = sess.run(transfer_vgg19.conv1_1)
        transfer_conv2_1 = sess.run(transfer_vgg19.conv2_1)
        transfer_conv3_1 = sess.run(transfer_vgg19.conv3_1)
        transfer_conv4_1 = sess.run(transfer_vgg19.conv4_1)
        transfer_conv5_1 = sess.run(transfer_vgg19.conv5_1)

        save_layer_activations(content_conv4_2, "features/content/conv4_2_%i.png")
        save_layer_activations(style_conv1_1, "features/style/conv1_1_%i.png")
        save_layer_activations(style_conv2_1, "features/style/conv2_1_%i.png")
        save_layer_activations(style_conv3_1, "features/style/conv3_1_%i.png")
        save_layer_activations(style_conv4_1, "features/style/conv4_1_%i.png")
        save_layer_activations(style_conv5_1, "features/style/conv5_1_%i.png")
        save_layer_activations(transfer_conv4_2, "features/transfer/conv4_2_%i.png")
        """

        return best_image


def calculate_layer_content_loss(content_layer, transfer_layer):
    return tf.reduce_mean(tf.squared_difference(content_layer, transfer_layer))


def calculate_layer_style_loss(style_layer, transfer_layer):
    feature_map_count = np.float32(transfer_layer.shape[3].value)
    feature_map_size = np.float32(transfer_layer.shape[1].value * transfer_layer.shape[2].value)

    style_gram_matrix = calculate_gram_matrix(style_layer)
    transfer_gram_matrix = calculate_gram_matrix(transfer_layer)

    mean_square_error = tf.reduce_mean(tf.squared_difference(style_gram_matrix, transfer_gram_matrix))
    return mean_square_error / (4 * tf.square(feature_map_count) * tf.square(feature_map_size))


def calculate_gram_matrix(convolution_layer):
    matrix = tf.reshape(convolution_layer, shape=[-1, convolution_layer.shape[3]])
    return tf.matmul(tf.transpose(matrix), matrix)


def load_image(filename):
    image = np.array(Image.open(filename).convert("RGB"), dtype=np.float32)
    image = image[:, :, ::-1] - VGG_MEAN
    image = image.reshape((1, image.shape[0], image.shape[1], 3)).astype(np.float32)
    return image


def save_image(image, filename):
    image = image[0, :, :, :] + VGG_MEAN
    image = image[:, :, ::-1]
    image = np.clip(image, 0, 255.0)
    image = np.uint8(image)

    result = Image.fromarray(image)
    result.save(filename)


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

    content_image = tf.constant(load_image(args.content_image_path))
    style_image = tf.constant(load_image(args.style_image_path))
    init_image = tf.Variable(np.random.randn(*content_image.shape).astype(np.float32) * 0.0001)
    result = style_transfer(content_image, style_image, init_image, args.weights_data)
    save_image(result, "final_transfer_image.png")
