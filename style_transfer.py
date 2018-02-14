from NIMA.model import get_nima_model
from PSPNet.model import *
from matting import *
from segmentation import *
from vgg19 import VGG19ConvSub, load_weights, VGG_MEAN


def style_transfer(content_image, style_image, content_masks, style_masks, init_image, args):
    print("Style transfer started")

    weight_restorer = load_weights()

    image_placeholder = tf.placeholder(tf.float32, shape=[1, None, None, 3])
    vgg19 = VGG19ConvSub(image_placeholder)

    with tf.Session() as sess:
        transfer_image = tf.Variable(init_image)

        sess.run(tf.global_variables_initializer())
        weight_restorer.init(sess)
        content_conv4_2 = sess.run(fetches=vgg19.conv4_2, feed_dict={image_placeholder: content_image})
        style_conv1_1, style_conv2_1, style_conv3_1, style_conv4_1, style_conv5_1 = sess.run(
            fetches=[vgg19.conv1_1, vgg19.conv2_1, vgg19.conv3_1, vgg19.conv4_1, vgg19.conv5_1],
            feed_dict={image_placeholder: style_image})

        with tf.variable_scope("", reuse=True):
            vgg19 = VGG19ConvSub(transfer_image)

        content_loss = calculate_layer_content_loss(content_conv4_2, vgg19.conv4_2)

        style_loss = (1. / 5.) * calculate_layer_style_loss(style_conv1_1, vgg19.conv1_1, content_masks, style_masks)
        style_loss += (1. / 5.) * calculate_layer_style_loss(style_conv2_1, vgg19.conv2_1, content_masks, style_masks)
        style_loss += (1. / 5.) * calculate_layer_style_loss(style_conv3_1, vgg19.conv3_1, content_masks, style_masks)
        style_loss += (1. / 5.) * calculate_layer_style_loss(style_conv4_1, vgg19.conv4_1, content_masks, style_masks)
        style_loss += (1. / 5.) * calculate_layer_style_loss(style_conv5_1, vgg19.conv5_1, content_masks, style_masks)

        photorealism_regularization = calculate_photorealism_regularization(transfer_image, content_image)

        nima_loss = compute_nima_loss(transfer_image)

        content_loss = args.content_weight * content_loss
        style_loss = args.style_weight * style_loss
        photorealism_regularization = args.regularization_weight * photorealism_regularization
        nima_loss = args.nima_weight * nima_loss

        total_loss = content_loss + style_loss + photorealism_regularization + nima_loss

        optimizer = tf.train.AdamOptimizer(learning_rate=args.adam_learning_rate, beta1=args.adam_beta1,
                                           beta2=args.adam_beta2, epsilon=args.adam_epsilon)

        train_op = optimizer.minimize(total_loss, var_list=[transfer_image])
        sess.run(adam_variables_initializer(optimizer, [transfer_image]))

        min_loss, best_image = float("inf"), None
        for i in range(args.iterations + 1):
            _, result_image, loss, c_loss, s_loss, n_loss = sess.run(
                fetches=[train_op, transfer_image, total_loss, content_loss, style_loss, nima_loss])

            if i % args.print_loss_interval == 0:
                print(
                    "Iteration: {0:5} \t Total loss: {1:15.2f} \t "
                    "Content loss: {2:15.2f} \t Style loss: {3:15.2f} \t"
                    "NIMA loss: {4:15.2f} \t".format(i, loss, c_loss, s_loss, n_loss))

            if loss < min_loss:
                min_loss, best_image = loss, result_image

            if i % args.intermediate_result_interval == 0:
                save_image(best_image, "transfer/res_{}.png".format(i))

        print("Style transfer finished")
        return best_image


def adam_variables_initializer(adam_opt, var_list):
    adam_vars = [adam_opt.get_slot(var, name)
                 for name in adam_opt.get_slot_names()
                 for var in var_list if var is not None]
    adam_vars.extend(list(adam_opt._get_beta_accumulators()))
    return tf.variables_initializer(adam_vars)


def compute_nima_loss(transfer_image):
    input = (transfer_image / 127.5) - 1.0
    model = get_nima_model(input)

    def mean_score(scores):
        scores = tf.squeeze(scores)
        si = tf.range(1, 11, 1, dtype=tf.float32)
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


# load image and preprocess as VGG19 input
def load_input_image(filename, normalize=False):
    image = np.array(Image.open(filename).convert("RGB"), dtype=np.float32)
    image = image[:, :, ::-1] - VGG_MEAN
    image = image.reshape((1, image.shape[0], image.shape[1], 3)).astype(np.float32)
    if normalize:
        image = image / 255.0
    return image


def save_image(image, filename):
    image = image[0, :, :, :] + VGG_MEAN
    image = image[:, :, ::-1]
    image = np.clip(image, 0, 255.0)
    image = np.uint8(image)

    result = Image.fromarray(image)
    result.save(filename)


# if images are of different shape, resize image1 to match shape of image0
def match_shape(image0, image1):
    if image0.shape == image1.shape:
        return image1
    else:
        shape = (image0.shape[1], image0.shape[0])
        return cv2.resize(image1, shape)


if __name__ == "__main__":
    import argparse

    """Parse program arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--content_image", type=str, help="content image path", default="content.png")
    parser.add_argument("--style_image", type=str, help="style image path", default="style.png")
    parser.add_argument("--output_image", type=str, help="Output image path, default: result.jpg",
                        default="result.jpg")
    parser.add_argument("--iterations", type=int, help="Number of iterations, default: 2000",
                        default=4000)
    parser.add_argument("--intermediate_result_interval", type=int,
                        help="Interval of iterations until a intermediate result is saved., default: 100",
                        default=100)
    parser.add_argument("--print_loss_interval", type=int,
                        help="Interval of iterations until the current loss is printed to console., default: 1",
                        default=1)
    parser.add_argument("--content_weight", type=float,
                        help="Weight of the content loss.",
                        default=1)
    parser.add_argument("--style_weight", type=float,
                        help="Weight of the style loss.",
                        default=100)
    parser.add_argument("--regularization_weight", type=float,
                        help="Weight of the photorealism regularization.",
                        default=10 ** 4)
    parser.add_argument("--nima_weight", type=float,
                        help="Weight for nima loss.",
                        default=10 ** 5)
    parser.add_argument("--init_image_scaling", type=float,
                        help="Scaling factor for the init image (random noise)., default: 0.0001",
                        default=0.0001)
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
    parser.add_argument("--semantic_thresh", type=float, help="Smantic threshold for label grouping., default: 0.4",
                        default=0.8)
    init_image_options = ["noise", "content", "style"]
    parser.add_argument("--init", type=str, help="Initialization image (%s).", default="content")
    parser.add_argument("--gpu", help="comma separated list of GPU(s) to use.", default="0")
    parser.add_argument("--cache", type=bool, nargs="?", help="If specified, reuse segmentation images if they exist.",
                        const=True, default=False)

    args = parser.parse_args()
    assert (args.init in init_image_options)

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    args = parser.parse_args()

    """Check if image files exist"""
    for path in [args.content_image, args.style_image]:
        if path is None or not os.path.isfile(path):
            print("Image file %s does not exist." % path)
            exit(0)

    # create directory transfer if it does not exist
    if not os.path.exists("transfer"):
        os.makedirs("transfer")

    content_image = load_input_image(args.content_image)
    style_image = load_input_image(args.style_image)

    with tf.Session(graph=tf.Graph()) as sess:
        placeholder = tf.placeholder(tf.float32, shape=[1, None, None, 3])
        net = PSPNet50({'data': placeholder}, is_training=False, num_classes=150)

        content_labels, content_seg = compute_segmentation(args.content_image, net, sess, placeholder,
                                                           args.semantic_thresh, args.cache)
        style_labels, style_seg = compute_segmentation(args.style_image, net, sess, placeholder, args.semantic_thresh,
                                                       args.cache)

    # enforce image shapes on segmentation shapes
    content_seg = match_shape(content_image, content_seg)
    style_seg = match_shape(style_image, style_seg)

    # compute all segmentation labels as union of style labels and content labels
    labels = tuple(content_labels | style_labels)

    # create binary masks for each label and both segmentation images
    content_segmentation_masks = [extract_mask_for_label(content_seg, label) for label in labels]
    style_segmentation_masks = [extract_mask_for_label(style_seg, label) for label in labels]

    if args.init == "noise":
        init_image = np.random.randn(*content_image.shape).astype(np.float32) * args.init_image_scaling
    elif args.init == "content":
        init_image = content_image
    elif args.init == "style":
        init_image = style_image

    result = style_transfer(content_image, style_image, content_segmentation_masks, style_segmentation_masks,
                            init_image, args)
    save_image(result, "final_transfer_image.png")
