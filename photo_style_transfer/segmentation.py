import os
from ast import literal_eval

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image

from photo_style_transfer.PSPNet.inference import create_segmentation_ade20k
from photo_style_transfer.semantic_merge import merge_classes

SEGMENTATION_MAX_LABELS = 20


def read_segmentation_labels(filename, color_filter=None):
    segmentation_labels = dict()
    with open(filename, 'r+') as file:
        lines = file.read().split('\n')

        for line in lines:
            color_str, classes_str = line.split('\t')
            segmentation_color = literal_eval(color_str)
            classes = [class_name.strip() for class_name in classes_str.split(',')]

            if color_filter and segmentation_color in color_filter:
                if segmentation_color in segmentation_labels:
                    # if color appears twice, do not overwrite it, but append it to the existing set of class names
                    # unlikely to happen, but might due to wrong annotation in PSPNet-ADE20k model
                    segmentation_labels[segmentation_color].extend(classes)
                else:
                    segmentation_labels[segmentation_color] = classes

    return segmentation_labels


def extract_unique_colors(input_image):
    def iterate_pixels(image):
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                yield tuple(image[y, x])

    unique_colors = set([tuple(current_color) for current_color in iterate_pixels(input_image)])

    if len(unique_colors) > SEGMENTATION_MAX_LABELS:
        raise ValueError("Found %i colors in segmentation, %i allowed." % (len(unique_colors), SEGMENTATION_MAX_LABELS))

    return unique_colors


def load_segmentation(filename):
    image = np.array(Image.open(filename).convert("RGB"), dtype=np.uint8)
    image.reshape((1, image.shape[0], image.shape[1], 3))

    unique_colors = extract_unique_colors(image)

    return unique_colors, image


def bgr2rgb(bgr_image):
    return bgr_image[..., ::-1]


def change_filename(filename, suffix, extension=None):
    path, ext = os.path.splitext(filename)
    if extension is None:
        extension = ext
    return path + suffix + extension


def image_group_colors(image, groups, match_dominant_colors=True):
    """
    image: image containing only colors that are also in groups
    groups: list of list of colors
    return: an image where each color is mapped to the first color in its group
    """
    print("Reducing color segments to one segmentation started")

    grouped = image.copy()

    if match_dominant_colors:
        # keep the dominant colors in the segmentation consistent
        # this can be used to compare different semantic thresholds but does not effect the final result
        histogram = dict()
        for group in groups:
            for current_color in group:
                histogram[current_color] = 0
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                histogram[tuple(image[y, x])] += 1
        dominant_colors = []
        for group in groups:
            group_histogram = dict((current_color, histogram[current_color]) for current_color in group)
            dominant_colors.append(max(group_histogram.keys(), key=(lambda k: group_histogram[k])))
    else:
        dominant_colors = [group[0] for group in groups]

    for i, group in enumerate(groups):
        target = dominant_colors[i]
        for y in range(image.shape[0]):
            for x in range(image.shape[1]):
                if tuple(image[y, x]) in group:
                    grouped[y, x] = target

    print("Reducing color segments to one segmentation finished")

    return grouped


"""
filename: image path
semantic_threshold: threshold between 0 and 1 for reducing the number of labels in the segmentation by grouping
semantically similar labels. (0: all classes are merged, 1: classes remain distinct)
dump_results: write intermediate segmentation results to file system
return: list of labels, segmentation
"""


def compute_segmentation(filename, net, sess, placeholder, semantic_threshold=1, dump_results=True):
    print("Segmentation started for '{}'".format(filename))

    segmentation_filename = change_filename(filename, '_seg', '.png')

    # only create segmentation mask if it does not exist yet
    if os.path.exists(segmentation_filename):
        print("Segmentation file '{}' already exists, use existing.".format(segmentation_filename))
        return load_segmentation(segmentation_filename)

    # create PSPNet segmentation
    segmentation = create_segmentation_ade20k(filename, net, sess, placeholder)

    if dump_results:
        cv2.imwrite(change_filename(filename, '_seg_raw', '.png'), bgr2rgb(segmentation))

    if semantic_threshold <= 1:
        unique_colors = extract_unique_colors(segmentation)
        print("Found {} segmentation classes in image '{}'.".format(len(unique_colors), filename))

        # read labels for semantic grouping
        labels_filename = os.path.join(os.path.dirname(__file__), 'PSPNet/utils/ade20k_labels.txt')
        labels = read_segmentation_labels(labels_filename, color_filter=unique_colors)

        # merge colors based on the semantic distance of class labels
        merged = merge_classes(labels, semantic_threshold)

        # group the segmentation colors accordingly
        color_groups = [pair[0] for pair in merged]
        segmentation = image_group_colors(segmentation, color_groups)

    if dump_results:
        cv2.imwrite(segmentation_filename, bgr2rgb(segmentation))

    unique_colors = extract_unique_colors(segmentation)

    print("Segmentation finished for '{}'".format(filename))
    return unique_colors, segmentation


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
