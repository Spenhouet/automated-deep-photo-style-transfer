import argparse

from scipy import misc

from main.PSPNet.inference import load
from main.PSPNet.model import PSPNet50
from main.PSPNet.tools import *
from main.path import WEIGHTS_DIR


def get_arguments():
    parser = argparse.ArgumentParser(description="Reproduced PSPNet")
    parser.add_argument("--img-path", type=str, default='',
                        help="Path to the RGB image file.")
    parser.add_argument("--checkpoints", type=str, default=os.path.join(WEIGHTS_DIR, 'PSPNet'),
                        help="Path to restore weights.")
    parser.add_argument("--flipped-eval", action="store_true",
                        help="whether to evaluate with flipped img.")
    parser.add_argument("--dataset", type=str, default='',
                        choices=['ade20k', 'cityscapes'],
                        required=True)

    return parser.parse_args()


def main():
    args = get_arguments()

    param = {'crop_size': [473, 473],
             'num_classes': 150,
             'model': PSPNet50}

    crop_size = param['crop_size']
    num_classes = param['num_classes']
    PSPNet = param['model']

    # preprocess images
    img, filename = load_img(args.img_path)
    img_shape = tf.shape(img)
    h, w = (tf.maximum(crop_size[0], img_shape[0]), tf.maximum(crop_size[1], img_shape[1]))

    img = preprocess(img, h, w)

    # Create network.
    net = PSPNet({'data': img}, is_training=False, num_classes=num_classes)
    with tf.variable_scope('', reuse=True):
        flipped_img = tf.image.flip_left_right(tf.squeeze(img))
        flipped_img = tf.expand_dims(flipped_img, dim=0)
        net2 = PSPNet({'data': flipped_img}, is_training=False, num_classes=num_classes)

    raw_output = net.layers['conv6']

    # Do flipped eval or not
    if args.flipped_eval:
        flipped_output = tf.image.flip_left_right(tf.squeeze(net2.layers['conv6']))
        flipped_output = tf.expand_dims(flipped_output, dim=0)
        raw_output = tf.add_n([raw_output, flipped_output])

    # Predictions.
    raw_output_up = tf.image.resize_bilinear(raw_output, size=[h, w], align_corners=True)
    raw_output_up = tf.image.crop_to_bounding_box(raw_output_up, 0, 0, img_shape[0], img_shape[1])
    raw_output_up = tf.argmax(raw_output_up, dimension=3)
    pred = decode_labels(raw_output_up, img_shape, num_classes)

    # Init tf Session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    init = tf.global_variables_initializer()

    sess.run(init)

    restore_var = tf.global_variables()

    checkpoint = tf.train.get_checkpoint_state(args.checkpoints)
    if checkpoint and checkpoint.model_checkpoint_path:
        loader = tf.train.Saver(var_list=restore_var)
        load(loader, sess, checkpoint.model_checkpoint_path)
    else:
        print('No checkpoint file found.')

    preds = sess.run(pred)

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    misc.imsave(args.save_dir + filename, preds[0])


if __name__ == '__main__':
    main()
