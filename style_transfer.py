import argparse
import json
import os
from datetime import datetime

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

import components.NIMA.model as nima
import components.VGG19.model as vgg
from components.matting import compute_matting_laplacian
from components.segmentation import compute_segmentation
from components.semantic_merge import merge_segments, reduce_dict, mask_for_tf, extract_segmentation_masks


def style_transfer(content_image, style_image, content_masks, style_masks, init_image, result_dir, timestamp, args):
    print("Style transfer started")

    content_image = vgg.preprocess(content_image)
    style_image = vgg.preprocess(style_image)

    weight_restorer = vgg.load_weights()

    image_placeholder = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    vgg19 = vgg.VGG19ConvSub(image_placeholder)

    with tf.Session() as sess:
        transfer_image = tf.Variable(init_image)
        transfer_image_vgg = vgg.preprocess(transfer_image)
        transfer_image_nima = nima.preprocess(transfer_image)

        sess.run(tf.global_variables_initializer())
        weight_restorer.init(sess)
        content_conv4_2 = sess.run(fetches=vgg19.conv4_2, feed_dict={image_placeholder: content_image})
        style_conv1_1, style_conv2_1, style_conv3_1, style_conv4_1, style_conv5_1 = sess.run(
            fetches=[vgg19.conv1_1, vgg19.conv2_1, vgg19.conv3_1, vgg19.conv4_1, vgg19.conv5_1],
            feed_dict={image_placeholder: style_image})

        with tf.variable_scope("", reuse=True):
            vgg19 = vgg.VGG19ConvSub(transfer_image_vgg)

        content_loss = calculate_layer_content_loss(content_conv4_2, vgg19.conv4_2)

        style_loss = (1. / 5.) * calculate_layer_style_loss(style_conv1_1, vgg19.conv1_1, content_masks, style_masks)
        style_loss += (1. / 5.) * calculate_layer_style_loss(style_conv2_1, vgg19.conv2_1, content_masks, style_masks)
        style_loss += (1. / 5.) * calculate_layer_style_loss(style_conv3_1, vgg19.conv3_1, content_masks, style_masks)
        style_loss += (1. / 5.) * calculate_layer_style_loss(style_conv4_1, vgg19.conv4_1, content_masks, style_masks)
        style_loss += (1. / 5.) * calculate_layer_style_loss(style_conv5_1, vgg19.conv5_1, content_masks, style_masks)

        photorealism_regularization = calculate_photorealism_regularization(transfer_image_vgg, content_image)

        nima_loss = compute_nima_loss(transfer_image_nima)

        content_loss = args.content_weight * content_loss
        style_loss = args.style_weight * style_loss
        photorealism_regularization = args.regularization_weight * photorealism_regularization
        nima_loss = args.nima_weight * nima_loss

        total_loss = content_loss + style_loss + photorealism_regularization + nima_loss

        tf.summary.scalar('Content loss', content_loss)
        tf.summary.scalar('Style loss', style_loss)
        tf.summary.scalar('Photorealism Regularization', photorealism_regularization)
        tf.summary.scalar('NIMA loss', nima_loss)
        tf.summary.scalar('Total loss', total_loss)

        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(os.path.join(os.path.dirname(__file__), 'logs/{}'.format(timestamp)),
                                               sess.graph)

        iterations_dir = os.path.join(result_dir, "iterations")
        os.mkdir(iterations_dir)

        optimizer = tf.train.AdamOptimizer(learning_rate=args.adam_learning_rate, beta1=args.adam_beta1,
                                           beta2=args.adam_beta2, epsilon=args.adam_epsilon)

        train_op = optimizer.minimize(total_loss, var_list=[transfer_image])
        sess.run(adam_variables_initializer(optimizer, [transfer_image]))

        min_loss, best_image = float("inf"), None
        for i in range(args.iterations + 1):
            _, result_image, loss, c_loss, s_loss, p_loss, n_loss, summary = sess.run(
                fetches=[train_op, transfer_image, total_loss, content_loss, style_loss, photorealism_regularization,
                         nima_loss, summary_op])

            summary_writer.add_summary(summary, i)

            if i % args.print_loss_interval == 0:
                print(
                    "Iteration: {0:5} \t "
                    "Total loss: {1:15.2f} \t "
                    "Content loss: {2:15.2f} \t "
                    "Style loss: {3:15.2f} \t "
                    "Photorealism Regularization: {4:15.2f} \t "
                    "NIMA loss: {5:15.2f} \t".format(i, loss, c_loss, s_loss, p_loss, n_loss))

            if loss < min_loss:
                min_loss, best_image = loss, result_image

            if i % args.intermediate_result_interval == 0:
                save_image(best_image, os.path.join(iterations_dir, "iter_{}.png".format(i)))

        return best_image


def adam_variables_initializer(adam_opt, var_list):
    adam_vars = [adam_opt.get_slot(var, name)
                 for name in adam_opt.get_slot_names()
                 for var in var_list if var is not None]
    adam_vars.extend(list(adam_opt._get_beta_accumulators()))
    return tf.variables_initializer(adam_vars)


def compute_nima_loss(image):
    model = nima.get_nima_model(image)

    def mean_score(scores):
        scores = tf.squeeze(scores)
        si = tf.range(1, 11, dtype=tf.float32)
        return tf.reduce_sum(tf.multiply(si, scores), name='nima_score')

    nima_score = mean_score(model.output)

    nima_loss = tf.identity(10.0 - nima_score, name='nima_loss')
    return nima_loss


def calculate_layer_content_loss(content_layer, transfer_layer):
    return tf.reduce_mean(tf.squared_difference(content_layer, transfer_layer))


def calculate_layer_style_loss(style_layer, transfer_layer, content_masks, style_masks):
    # scale masks to current layer
    content_size = tf.TensorShape(transfer_layer.shape[1:3])
    style_size = tf.TensorShape(style_layer.shape[1:3])

    def resize_masks(masks, size):
        return [tf.image.resize_bilinear(mask, size) for mask in masks]

    style_masks = resize_masks(style_masks, style_size)
    content_masks = resize_masks(content_masks, content_size)

    feature_map_count = np.float32(transfer_layer.shape[3].value)
    feature_map_size = np.float32(transfer_layer.shape[1].value) * np.float32(transfer_layer.shape[2].value)

    means_per_channel = []
    for content_mask, style_mask in zip(content_masks, style_masks):
        transfer_gram_matrix = calculate_gram_matrix(transfer_layer, content_mask)
        style_gram_matrix = calculate_gram_matrix(style_layer, style_mask)

        mean = tf.reduce_mean(tf.squared_difference(style_gram_matrix, transfer_gram_matrix))
        means_per_channel.append(mean / (2 * tf.square(feature_map_count) * tf.square(feature_map_size)))

    style_loss = tf.reduce_sum(means_per_channel)

    return style_loss


def calculate_photorealism_regularization(output, content_image):
    # normalize content image and out for matting and regularization computation
    content_image = content_image / 255.0
    output = output / 255.0

    # compute matting laplacian
    matting = compute_matting_laplacian(content_image[0, ...])

    # compute photorealism regularization loss
    regularization_channels = []
    for output_channel in tf.unstack(output, axis=-1):
        channel_vector = tf.reshape(tf.transpose(output_channel), shape=[-1])
        matmul_right = tf.sparse_tensor_dense_matmul(matting, tf.expand_dims(channel_vector, -1))
        matmul_left = tf.matmul(tf.expand_dims(channel_vector, 0), matmul_right)
        regularization_channels.append(matmul_left)

    regularization = tf.reduce_sum(regularization_channels)
    return regularization


def calculate_gram_matrix(convolution_layer, mask):
    matrix = tf.reshape(convolution_layer, shape=[-1, convolution_layer.shape[3]])
    mask_reshaped = tf.reshape(mask, shape=[matrix.shape[0], 1])
    matrix_masked = matrix * mask_reshaped
    return tf.matmul(matrix_masked, matrix_masked, transpose_a=True)


def load_image(filename):
    image = np.array(Image.open(filename).convert("RGB"), dtype=np.float32)
    image = np.expand_dims(image, axis=0)
    return image


def save_image(image, filename):
    image = image[0, :, :, :]
    image = np.clip(image, 0, 255.0)
    image = np.uint8(image)

    result = Image.fromarray(image)
    result.save(filename)


def change_filename(dir_name, filename, suffix, extension=None):
    path, ext = os.path.splitext(filename)
    if extension is None:
        extension = ext
    return os.path.join(dir_name, path + suffix + extension)


def write_metadata(dir, args, load_segmentation):
    # collect metadata and write to transfer dir
    meta = {
        "init": args.init,
        "iterations": args.iterations,
        "content": args.content_image,
        "style": args.style_image,
        "content_weight": args.content_weight,
        "style_weight": args.style_weight,
        "regularization_weight": args.regularization_weight,
        "nima_weight": args.nima_weight,
        "semantic_thresh": args.semantic_thresh,
        "similarity_metric": args.similarity_metric,
        "load_segmentation": load_segmentation,
        "adam": {
            "learning_rate": args.adam_learning_rate,
            "beta1": args.adam_beta1,
            "beta2": args.adam_beta2,
            "epsilon": args.adam_epsilon
        }
    }
    filename = os.path.join(dir, "meta.json")
    with open(filename, "w+") as file:
        file.write(json.dumps(meta, indent=4))


if __name__ == "__main__":
    """Parse program arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_image", type=str, help="content image path", default="content.png")
    parser.add_argument("--style_image", type=str, help="style image path", default="style.png")
    parser.add_argument("--output_image", type=str, help="Output image path, default: result.jpg",
                        default="result.jpg")
    parser.add_argument("--iterations", type=int, help="Number of iterations, default: 4000",
                        default=4000)
    parser.add_argument("--intermediate_result_interval", type=int,
                        help="Interval of iterations until a intermediate result is saved., default: 100",
                        default=100)
    parser.add_argument("--print_loss_interval", type=int,
                        help="Interval of iterations until the current loss is printed to console., default: 1",
                        default=1)
    parser.add_argument("--content_weight", type=float,
                        help="Weight of the content loss., default: 1",
                        default=1)
    parser.add_argument("--style_weight", type=float,
                        help="Weight of the style loss., default: 100",
                        default=100)
    parser.add_argument("--regularization_weight", type=float,
                        help="Weight of the photorealism regularization.",
                        default=10 ** 4)
    parser.add_argument("--nima_weight", type=float,
                        help="Weight for nima loss.",
                        default=10 ** 5)
    parser.add_argument("--adam_learning_rate", type=float,
                        help="Learning rate for the adam optimizer., default: 1.0",
                        default=1.0)
    parser.add_argument("--adam_beta1", type=float,
                        help="Beta1 for the adam optimizer., default: 0.9",
                        default=0.9)
    parser.add_argument("--adam_beta2", type=float,
                        help="Beta2 for the adam optimizer., default: 0.999",
                        default=0.999)
    parser.add_argument("--adam_epsilon", type=float,
                        help="Epsilon for the adam optimizer., default: 1e-08",
                        default=1e-08)
    parser.add_argument("--semantic_thresh", type=float, help="Semantic threshold for label grouping., default: 0.8",
                        default=0.8)
    parser.add_argument("--similarity_metric", type=str, help="Semantic similarity metric for label grouping., default: li",
                        default="li")
    init_image_options = ["noise", "content", "style"]
    similarity_metric_options = ["li", "wpath", "jcn", "lin", "wup", "res"]
    parser.add_argument("--init", type=str, help="Initialization image (%s).", default="content")
    parser.add_argument("--gpu", help="comma separated list of GPU(s) to use.", default="0")

    args = parser.parse_args()
    assert (args.init in init_image_options)
    # For more information on the similarity metrics: http://gsi-upm.github.io/sematch/similarity/#word-similarity
    assert (args.similarity_metric in similarity_metric_options)

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    timestamp = datetime.now().strftime('%Y_%m_%d_%H_%M')

    result_dir = 'result_' + timestamp
    os.mkdir(result_dir)

    # check if manual segmentation masks are available
    content_segmentation_filename = change_filename('', args.content_image, '_seg', '.png')
    style_segmentation_filename = change_filename('', args.style_image, '_seg', '.png')
    load_segmentation = os.path.exists(content_segmentation_filename) and os.path.exists(style_segmentation_filename)

    write_metadata(result_dir, args, load_segmentation)

    """Check if image files exist"""
    for path in [args.content_image, args.style_image]:
        if path is None or not os.path.isfile(path):
            print("Image file {} does not exist.".format(path))
            exit(0)

    content_image = load_image(args.content_image)
    style_image = load_image(args.style_image)

    # use existing if available
    if (load_segmentation):
        print("Load segmentation from files.")
        content_segmentation_image = cv2.imread(content_segmentation_filename)
        style_segmentation_image = cv2.imread(style_segmentation_filename)
        content_segmentation_masks = extract_segmentation_masks(content_segmentation_image)
        style_segmentation_masks = extract_segmentation_masks(style_segmentation_image)
    else:
        print("Create segmentation.")
        content_segmentation, style_segmentation = compute_segmentation(args.content_image, args.style_image)

        cv2.imwrite(change_filename(result_dir, args.content_image, '_seg_raw', '.png'), content_segmentation)
        cv2.imwrite(change_filename(result_dir, args.style_image, '_seg_raw', '.png'), style_segmentation)

        content_segmentation_masks, style_segmentation_masks = merge_segments(content_segmentation, style_segmentation,
                                                                              args.semantic_thresh, args.similarity_metric)

    cv2.imwrite(change_filename(result_dir, args.content_image, '_seg', '.png'),
                reduce_dict(content_segmentation_masks, content_image))
    cv2.imwrite(change_filename(result_dir, args.style_image, '_seg', '.png'),
                reduce_dict(style_segmentation_masks, style_image))

    if args.init == "noise":
        random_noise_scaling_factor = 0.0001
        random_noise = np.random.randn(*content_image.shape).astype(np.float32)
        init_image = vgg.postprocess(random_noise * random_noise_scaling_factor).astype(np.float32)
    elif args.init == "content":
        init_image = content_image
    elif args.init == "style":
        init_image = style_image
    else:
        print("Init image parameter {} unknown.".format(args.init))
        exit(0)

    result = style_transfer(content_image, style_image, mask_for_tf(content_segmentation_masks),
                            mask_for_tf(style_segmentation_masks), init_image, result_dir, timestamp, args)
    save_image(result, os.path.join(result_dir, "final_transfer_image.png"))
