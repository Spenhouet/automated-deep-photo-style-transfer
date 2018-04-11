import os

import cv2
import numpy as np
import scipy.io as sio
import tensorflow as tf

from components.PSPNet.model import PSPNet50, load_color_label_dict
from components.path import WEIGHTS_DIR

CROP_SIZE = [473, 473]


def compute_segmentation(content_image_path, style_image_path):
    print("Compute segmentation started")

    with tf.Session(graph=tf.Graph()) as sess:
        content_image = load_img(content_image_path)
        style_image = load_img(style_image_path)

        placeholder = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        net = PSPNet50({'data': placeholder}, is_training=False, num_classes=150)

        content_pred, content_pre = segmentation_pred(net, content_image)
        style_pred, style_pre = segmentation_pred(net, style_image)

        # Init tf Session
        init = tf.global_variables_initializer()

        sess.run(init)

        restore_var = tf.global_variables()

        checkpoint = tf.train.get_checkpoint_state(os.path.join(WEIGHTS_DIR, 'PSPNet/checkpoint'))
        if checkpoint and checkpoint.model_checkpoint_path:
            loader = tf.train.Saver(var_list=restore_var)
            load(loader, sess, checkpoint.model_checkpoint_path)
        else:
            print('No checkpoint file found.')

        content_pre_image = sess.run(content_pre)
        style_pre_image = sess.run(style_pre)

        content_segmentation_bgr = sess.run(content_pred, feed_dict={placeholder: content_pre_image})
        style_segmentation_bgr = sess.run(style_pred, feed_dict={placeholder: style_pre_image})

        content_segmentation = cv2.cvtColor(content_segmentation_bgr[0], cv2.COLOR_BGR2RGB)
        style_segmentation = cv2.cvtColor(style_segmentation_bgr[0], cv2.COLOR_BGR2RGB)

        return content_segmentation, style_segmentation


def segmentation_pred(net, image):
    num_classes = 150

    image_shape = tf.shape(image)
    height = tf.maximum(CROP_SIZE[0], image_shape[0])
    width = tf.maximum(CROP_SIZE[1], image_shape[1])

    raw_output = net.layers['conv6']

    # Predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, size=[height, width], align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, image_shape[0], image_shape[1])
    raw_output_up = tf.argmax(raw_output_up, axis=3)

    color_table = list(load_color_label_dict().keys())
    color_mat = tf.constant(color_table, dtype=tf.float32)
    onehot_output = tf.one_hot(raw_output_up, depth=num_classes)
    onehot_output = tf.reshape(onehot_output, (-1, num_classes))
    pred = tf.matmul(onehot_output, color_mat)
    pred = tf.reshape(pred, (1, image_shape[0], image_shape[1], 3))

    pre = preprocess(image, height, width)

    return pred, pre


IMG_MEAN = np.array((103.939, 116.779, 123.68), dtype=np.float32)


def load_img(img_path):
    if not os.path.isfile(img_path):
        print('Not found file: {0}'.format(img_path))
        exit(0)

    filename = img_path.split('/')[-1]
    ext = filename.split('.')[-1]

    if ext.lower() == 'png':
        return tf.image.decode_png(tf.read_file(img_path), channels=3)
    elif ext.lower() == 'jpg':
        return tf.image.decode_jpeg(tf.read_file(img_path), channels=3)
    else:
        print('cannot process {0} file.'.format(filename))
        exit(0)


def preprocess(image, height, width):
    # Convert RGB to BGR
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=image)
    image = tf.cast(tf.concat(axis=2, values=[img_b, img_g, img_r]), dtype=tf.float32)
    # Extract mean.
    image -= IMG_MEAN

    pad_img = tf.image.pad_to_bounding_box(image, 0, 0, height, width)
    pad_img = tf.expand_dims(pad_img, dim=0)

    return pad_img


def read_label_colors(matfn):
    mat = sio.loadmat(matfn)
    color_table = mat['colors']
    shape = color_table.shape
    color_list = [tuple(color_table[i]) for i in range(shape[0])]

    return color_list


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))


if __name__ == '__main__':
    import argparse
    from style_transfer import change_filename

    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="image path", default="image.png")
    args = parser.parse_args()

    segmentation, _ = compute_segmentation(args.image, args.image)

    result_dir = 'raw_seg'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    cv2.imwrite(change_filename(result_dir, args.image, '_seg_raw', '.png'), segmentation)
