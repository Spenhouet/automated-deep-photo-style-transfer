import itertools as it
from operator import itemgetter

import networkx as nx
import nltk
import numpy as np
import tensorflow as tf
from os.path import join

from components.PSPNet.model import load_color_label_dict
from components.path import WEIGHTS_DIR

nltk.data.path.append(join(WEIGHTS_DIR, 'WordNet'))

from sematch.semantic.similarity import WordNetSimilarity

wns = WordNetSimilarity()


def color_tuples_to_label_list_tuples(color_tuples, color_label_dict):
    return it.chain.from_iterable(
        it.product(color_label_dict[first], color_label_dict[second]) for (first, second) in color_tuples)


# Replace colors in dictionary of color -> mask.
def replace_colors_in_dict(color_mask_dict, replacement_colors):
    new_color_mask_dict = {}
    for color, mask in color_mask_dict.items():
        new_color = replacement_colors[color] if color in replacement_colors else color
        # Merge masks if color already exists
        new_color_mask_dict[new_color] = np.logical_or(mask, new_color_mask_dict[
            new_color]) if new_color in new_color_mask_dict else mask

    return new_color_mask_dict


def merge_difference(first_masks, first_colors, second_colors, color_label_dict, label_color_dict, similarity_metric):
    print("Semantic merge of different segments started")

    # Get all colors that are only contained in one of the segmentation images
    difference = list(set(first_colors).difference(second_colors))

    # Combine minimal set of colors to compare via semantic similarity
    difference_colors_to_compare = [list(it.product([dif_color], second_colors)) for dif_color in difference]

    # Transform colors to labels
    difference_labels_to_compare_list = [color_tuples_to_label_list_tuples(colors_to_compare, color_label_dict)
                                         for colors_to_compare in difference_colors_to_compare]

    # Add similarity score to label tuples
    annotated_difference_labels = [annotate_label_similarity(dif_labels, similarity_metric) for dif_labels in
                                   difference_labels_to_compare_list]

    # For labels that are only contained in one segmentation image get the highest matching label that is contained in
    # both images each
    highest_difference_matches = [max(annotated_tuples, key=itemgetter(0)) for annotated_tuples in
                                  annotated_difference_labels]

    # Drop similarity score
    edge_list_labels = [label_tuple for similarity, label_tuple in highest_difference_matches]

    # Turn labels back to colors and map as replacement color
    replacement_colors = {label_color_dict[l1]: label_color_dict[l2] for l1, l2 in edge_list_labels}

    new_first_segmentation = replace_colors_in_dict(first_masks, replacement_colors)

    return new_first_segmentation


def merge_segments(content_segmentation, style_segmentation, semantic_threshold, similarity_metric):
    print("Semantic merge of segments started")

    # load color - label mapping
    color_label_dict = load_color_label_dict()
    label_color_dict = {label: color for color, labels in color_label_dict.items() for label in labels}
    colors = color_label_dict.keys()

    # Extract the boolean mask for every color
    content_masks = extract_segmentation_masks(content_segmentation, colors)
    style_masks = extract_segmentation_masks(style_segmentation, colors)

    content_colors = content_masks.keys()
    style_colors = style_masks.keys()

    # Merge all colors that only occur in the style segmentation with the most similar in the content segmentation
    style_masks = merge_difference(style_masks, style_colors, content_colors, color_label_dict, label_color_dict, similarity_metric)
    style_colors = style_masks.keys()

    # Merge all colors that only occur in the content segmentation with the most similar in the style segmentation
    content_masks = merge_difference(content_masks, content_colors, style_colors, color_label_dict, label_color_dict, similarity_metric)
    content_colors = content_masks.keys()

    assert(frozenset(style_colors) == frozenset(content_colors))

    # Get all colors that are contained in both segmentation images
    intersection = list(set(content_colors).intersection(style_colors))

    # Combine minimal set of colors to compare via semantic similarity
    intersection_colors_to_compare = it.combinations(intersection, 2)

    # Transform colors to labels
    intersection_labels_to_compare = color_tuples_to_label_list_tuples(intersection_colors_to_compare, color_label_dict)

    # Add similarity score to label tuples
    annotated_intersection_labels = annotate_label_similarity(intersection_labels_to_compare, similarity_metric)

    # For labels that are contained in both segmentation images merge only these with a similarity over the threshold
    above_threshold_intersection = [(similarity, label_tuple) for (similarity, label_tuple) in
                                    annotated_intersection_labels if similarity >= semantic_threshold]

    # Drop similarity score
    edge_list_labels = [label_tuple for similarity, label_tuple in above_threshold_intersection]

    # Turn labels back to colors
    edge_list_colors = [(label_color_dict[l1], label_color_dict[l2]) for l1, l2 in edge_list_labels]

    # Find all sub graphs
    color_sub_graphs = list(nx.connected_components(nx.from_edgelist(edge_list_colors)))

    # Create a dictionary with all necessary color replacements
    replacement_colors = {color: list(color_graph)[0] for color_graph in color_sub_graphs for color in color_graph}

    new_content_segmentation = replace_colors_in_dict(content_masks, replacement_colors)
    new_style_segmentation = replace_colors_in_dict(style_masks, replacement_colors)

    assert new_content_segmentation.keys() == new_style_segmentation.keys()

    return new_content_segmentation, new_style_segmentation


def reduce_dict(dict, image):
    _, h, w, _ = image.shape
    arr = np.zeros((h, w, 3), int)
    for k, v in dict.items():
        I, J = np.where(v)
        arr[I, J] = k[::-1]
    return arr


def annotate_label_similarity(labels_to_compare, similarity_metric):
    return [(wns.word_similarity(l1, l2, similarity_metric), (l1, l2)) for (l1, l2) in labels_to_compare]


def get_labels_to_compare(label_lists_to_compare):
    return it.chain.from_iterable(it.product(l1, l2) for (l1, l2) in label_lists_to_compare)


def get_unique_colors_from_image(image):
    h, w, c = image.shape
    assert (c == 3)
    vec = np.reshape(image, (h * w, c))
    unique_colors = np.unique(vec, axis=0)
    return [tuple(color) for color in unique_colors]


def extract_segmentation_masks(segmentation, colors=None):
    if colors is None:
        # extract distinct colors from segmentation image
        colors = get_unique_colors_from_image(segmentation)
        colors = [color[::-1] for color in colors]

    return {color: mask for (color, mask) in
            ((color, np.all(segmentation.astype(np.int32) == color[::-1], axis=-1)) for color in colors) if
            mask.max()}


def mask_for_tf(segmentation_mask):
    return [tf.expand_dims(tf.expand_dims(tf.constant(segmentation_mask[key].astype(np.float32)), 0), -1) for key
            in sorted(segmentation_mask)]


if __name__ == '__main__':
    import argparse
    import os
    import cv2
    from style_transfer import load_image, change_filename

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_segmentation", type=str, help="raw segmentation image path", default="raw_seg.png")
    parser.add_argument("--semantic_thresh", type=float, help="Smantic threshold for label grouping", default=0.5)
    parser.add_argument("--similarity_metric", type=str, help="Smantic similarity metric for label grouping., default: li",
                        default="li")
    similarity_metric_options = ["li", "wpath", "jcn", "lin", "wup", "res"]
    args = parser.parse_args()

    # For more information on the similarity metrics: http://gsi-upm.github.io/sematch/similarity/#word-similarity
    assert (args.similarity_metric in similarity_metric_options)

    image = load_image(args.raw_segmentation)

    segmentation_image = cv2.imread(args.raw_segmentation)

    segmentation_masks, _ = merge_segments(segmentation_image, segmentation_image,
                                           args.semantic_thresh, args.similarity_metric)

    result_dir = 'semantic_merge'
    if not os.path.exists(result_dir):
        os.mkdir(result_dir)
    cv2.imwrite(change_filename(result_dir, args.raw_segmentation, '_{}'.format(args.semantic_thresh), '.png'),
                reduce_dict(segmentation_masks, image))
