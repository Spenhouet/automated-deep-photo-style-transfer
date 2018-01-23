from ast import literal_eval
import tensorflow as tf
import numpy as np

SEGMENTATION_MAX_LABELS = 20


def read_segmentation_labels(filename):
    labels = dict()
    with open(filename, 'r+') as file:
        lines = file.read().split('\n')

        for line in lines[:10]:
            color_str, classes_str = line.split('\t')
            color = literal_eval(color_str)
            classes = [class_name.strip() for class_name in classes_str.split(',')]

            if color in labels:
                # if color appears twice, do not overwrite it, but append it to the existing set of class names
                # unlikely to happen, but might due to wrong annotation in PSPNet-ADE20k model
                labels[color].extend(classes)
            else:
                labels[color] = classes

    return labels

def load_segmentation(filename):
    image = np.array(Image.open(filename).convert("RGB"), dtype=np.uint8)
    image.reshape((1, image.shape[0], image.shape[1], 3))

    def iterate_pixels(image):
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                yield tuple(image[y, x])

    unique_colors = set([tuple(color) for color in iterate_pixels(image)])

    if len(unique_colors) > SEGMENTATION_MAX_LABELS:
        raise ValueError("Found %i colors in segmentation, %i allowed." % (len(unique_colors), SEGMENTATION_MAX_LABELS))

    return unique_colors, image


def extract_mask_for_label(segmentation, label):

    # mask in numpy representation
    mask = np.all(segmentation == label, axis=-1).astype(np.float32)

    # mask as tensor
    mask_tensor = tf.expand_dims(tf.expand_dims(tf.constant(mask), 0), -1)

    return mask_tensor


if __name__ == '__main__':

    # only for testing
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', type=str)
    args = parser.parse_args()

    labels = read_segmentation_labels(args.filename)

    for color in labels:
        print(color, labels[color])