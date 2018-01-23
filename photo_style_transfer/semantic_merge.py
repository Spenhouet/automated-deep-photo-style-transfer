import nltk
nltk.download('wordnet')
nltk.download('wordnet_ic')
from sematch.semantic.similarity import WordNetSimilarity
from segmentation import *
import argparse

wns = WordNetSimilarity()

def merge_lists(*lists):
    merged = list()
    for l in lists:
        merged.extend(l)
    return merged

"""
labels: dict (color -> list of class names)
return: list of tuples (list of colors, list of class names)
"""
def merge_classes(labels, semantic_threshold):

    matched_color_pairs = list()

    for color_a in labels:
        for color_b in labels:
            if color_a != color_b:
                if match_labels(labels[color_a], labels[color_b], semantic_threshold):
                    matched_color_pairs.append((color_a, color_b))

    subgraphs = split_subgraphs(labels.keys(), matched_color_pairs)

    """
    merged = [
        ( [color1, color2, ...], [class1, class2, class3] ),
        ...
    ]
    """
    merged = list()
    for merged_colors in subgraphs:

        merged.append((
            merged_colors,
            merge_lists([labels[color] for color in merged_colors])
        ))

    return merged



def match_labels(labels_a, labels_b, thresh):
    for label_a in labels_a:
        for label_b in labels_b:
            if word_similarity(label_a, label_b) >= thresh:
                return True
    return False


def word_similarity(word_a, word_b):
    return wns.word_similarity(word_a, word_b, 'li')


"""
Split a graph into independent subgraphs
vertices: list of vertices
edges: list of edges (tuples)
return: list of list of vertices
"""
def split_subgraphs(vertices, edges):
    groups = list()

    vertices = set(vertices)
    edges = set([frozenset(edge) for edge in edges])

    def get_connected_vertices(a):
        return [b for b in vertices if frozenset((a, b)) in edges]

    while len(vertices) > 0:
        group = list()
        seed = vertices.pop()

        buffer = [seed]
        while (len(buffer) > 0):
            v = buffer.pop()
            group.append(v)
            conn = get_connected_vertices(v)

            for w in conn:
                vertices.remove(w)
                edges.remove(frozenset((v, w)))
            buffer.extend(conn)

        groups.append(group)

    return groups


if __name__ == '__main__':

    # only for testing
    parser = argparse.ArgumentParser()
    parser.add_argument('--thresh', type=float, help="semantic theshold")
    parser.add_argument('--filename', type=str, default='PSPNet/utils/ade20k_labels.txt')
    args = parser.parse_args()

    assert(args.thresh != None)

    labels = read_segmentation_labels(args.filename)
    merged = merge_classes(labels, args.thresh)

    print('new num labels: %i' % len(merged))

    for el in merged:
        print(el)