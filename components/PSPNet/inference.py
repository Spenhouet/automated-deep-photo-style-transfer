from __future__ import print_function

from model import PSPNet50
from tools import *
from ..path import WEIGHTS_DIR


def create_segmentation_ade20k(img_path, net, sess, placeholder):
    import tensorflow as tf

    param = {'crop_size': [473, 473],
             'num_classes': 150,
             'model': PSPNet50}

    crop_size = param['crop_size']
    num_classes = param['num_classes']

    # preprocess images
    img, filename = load_img(img_path)
    img_shape = tf.shape(img)
    h, w = (tf.maximum(crop_size[0], img_shape[0]), tf.maximum(crop_size[1], img_shape[1]))

    raw_output = net.layers['conv6']

    # Predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1])
    raw_output_up = tf.argmax(raw_output_up, axis=3)
    pred = decode_labels(raw_output_up, img_shape, num_classes)

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

    img_tf = preprocess(img, h, w)
    img = sess.run(img_tf)
    preds = sess.run(pred, feed_dict={placeholder: img})

    save_dir = "./"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    return preds[0]


def load(saver, sess, ckpt_path):
    saver.restore(sess, ckpt_path)
    print("Restored model parameters from {}".format(ckpt_path))
