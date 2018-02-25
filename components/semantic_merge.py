import itertools as it
from operator import itemgetter
from os.path import join

import networkx as nx
import nltk
import numpy as np
import tensorflow as tf

from components.PSPNet.model import load_color_label_dict

from components.path import WEIGHTS_DIR

nltk.data.path.append(join(WEIGHTS_DIR, 'WordNet'))

from sematch.semantic.similarity import WordNetSimilarity

wns = WordNetSimilarity()


def merge_segments(content_segmentation, style_segmentation, semantic_threshold):
    print("Semantic merge of segments started")

    color_label_dict = load_color_label_dict()
    label_color_dict = {label: color for color, labels in color_label_dict.items() for label in labels}
    colors = color_label_dict.keys()

    content_mask = extract_segmentation_mask(content_segmentation, colors)
    style_mask = extract_segmentation_mask(style_segmentation, colors)

    content_colors = content_mask.keys()
    style_colors = style_mask.keys()

    difference = list(set(content_colors).symmetric_difference(style_colors))
    intersection = list(set(content_colors).intersection(style_colors))

    difference_colors_to_compare = [it.product(intersection, [dif_color]) for dif_color in difference]
    intersection_colors_to_compare = it.combinations(intersection, 2)

    def color_tuples_to_label_list_tuples(color_tuples):
        return it.chain.from_iterable(
            it.product(color_label_dict[first], color_label_dict[second]) for (first, second) in color_tuples)

    difference_labels_to_compare_list = [color_tuples_to_label_list_tuples(colors_to_compare) for colors_to_compare in
                                         difference_colors_to_compare]
    intersection_labels_to_compare = color_tuples_to_label_list_tuples(intersection_colors_to_compare)

    annotated_difference_labels = [annotate_label_similarity(dif_labels) for dif_labels in
                                   difference_labels_to_compare_list]
    annotated_intersection_labels = annotate_label_similarity(intersection_labels_to_compare)

    highest_difference_matches = [max(annotated_tuples, key=itemgetter(0)) for annotated_tuples in
                                  annotated_difference_labels]

    above_threshold_intersection = filter(lambda t: t[0] > semantic_threshold,
                                          annotated_intersection_labels)

    labels_to_merge = it.chain(highest_difference_matches, above_threshold_intersection)

    edge_list = [label_tuple for similarity, label_tuple in labels_to_merge]

    merged_labels = list(nx.connected_components(nx.from_edgelist(edge_list)))

    def labels_to_colors(labels):
        return [label_color_dict[label] for label in labels]

    merged_colors = [labels_to_colors(labels) for labels in merged_labels]

    replacement_colors = {color: colors[0] for colors in merged_colors for color in colors}

    new_content_segmentation = {replacement_colors[color] if color in replacement_colors else color: mask for
                                color, mask in content_mask.items()}
    new_style_segmentation = {replacement_colors[color] if color in replacement_colors else color: mask for
                              color, mask in style_mask.items()}

    return new_content_segmentation, new_style_segmentation


def reduce_dict(dict, image):
    _, h, w, _ = image.shape
    arr = np.zeros((h, w, 3), int)
    for k, v in dict.items():
        I, J = np.where(v)
        arr[I, J] = k[::-1]
    return arr


def annotate_label_similarity(labels_to_compare):
    return [(wns.word_similarity(l1, l2, 'li'), (l1, l2)) for (l1, l2) in labels_to_compare]


def get_labels_to_compare(label_lists_to_compare):
    return it.chain.from_iterable(it.product(l1, l2) for (l1, l2) in label_lists_to_compare)


def word_similarity(word_a, word_b):
    return wns.word_similarity(word_a, word_b, 'li')


def extract_segmentation_mask(segmentation, colors):
    return {color: mask for (color, mask) in
            ((color, np.all(segmentation.astype(np.int32) == color[::-1], axis=-1)) for color in colors) if
            mask.max()}


def mask_for_tf(segmentation_mask):
    return [tf.expand_dims(tf.expand_dims(tf.constant(mask.astype(np.float32)), 0), -1) for mask
            in segmentation_mask.values()]
