from os.path import join

import nltk

from .path import WEIGHTS_DIR

nltk.data.path.append(join(WEIGHTS_DIR, 'WordNet'))

from sematch.semantic.similarity import WordNetSimilarity

wns = WordNetSimilarity()


def merge_classes(labels, semantic_threshold):
    """
    labels: dict (color -> list of class names)
    return: list of tuples (list of colors, list of class names)
    """
    print("Merge classes by semantic similarity started")

    matched_color_pairs = list()

    for color_a in labels:
        for color_b in labels:
            if color_a != color_b and match_labels(labels[color_a], labels[color_b], semantic_threshold):
                matched_color_pairs.append((color_a, color_b))

    sub_graphs = split_subgraphs(labels.keys(), matched_color_pairs)

    merged_classes = list()
    for merged_colors in sub_graphs:
        merged_classes.append((
            merged_colors,
            [item for sublist in [labels[merged_color] for merged_color in merged_colors] for item in sublist]
        ))

    print("Merge classes by semantic similarity finished")

    return merged_classes


def match_labels(labels_a, labels_b, thresh):
    for label_a in labels_a:
        for label_b in labels_b:
            if word_similarity(label_a, label_b) >= thresh:
                return True
    return False


def word_similarity(word_a, word_b):
    return wns.word_similarity(word_a, word_b, 'li')


def split_subgraphs(vertices, edges):
    """
    Split a graph into independent subgraphs
    vertices: list of vertices
    edges: list of edges (tuples)
    return: list of list of vertices
    """
    groups = list()

    vertices = set(vertices)
    edges = set([frozenset(edge) for edge in edges])

    def get_connected_vertices(a):
        return [b for b in vertices if frozenset((a, b)) in edges]

    while len(vertices) > 0:
        group = list()
        seed = vertices.pop()

        buffer = [seed]
        while len(buffer) > 0:
            v = buffer.pop()
            group.append(v)
            conn = get_connected_vertices(v)

            for w in conn:
                vertices.remove(w)
                edges.remove(frozenset((v, w)))
            buffer.extend(conn)

        groups.append(group)

    return groups
