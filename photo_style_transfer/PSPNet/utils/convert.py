import argparse

import scipy.io


def extract_colors(mat_filename):
    mat = scipy.io.loadmat(mat_filename)
    colors_mat = mat['colors']
    return colors_mat


def extract_annotations(annotations_filename):
    with open(annotations_filename, 'r+') as annotations_file:
        annotations_content = annotations_file.read()
        lines = annotations_content.split('\n')[1:]
        assert (len(lines) == 150)

        for line in lines:
            cells = line.split('\t')
            class_names = cells[-1]
            yield class_names


if __name__ == '__main__':
    # currently only support for ade20k
    supported_datasets = ['ade20k']

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name (supported: %s)' % ', '.join(supported_datasets),
                        default='ade20k')
    args = parser.parse_args()

    assert (args.dataset in supported_datasets)

    colors_filename = args.dataset + '_colors.mat'
    filename = args.dataset + '_annotations.txt'

    colors = [str(tuple(color)) for color in extract_colors(colors_filename)]
    annotations = extract_annotations(filename)

    labels = zip(colors, annotations)

    with open(args.dataset + '_labels.txt', 'w+') as file:
        content = '\n'.join(['\t'.join(pair) for pair in labels])
        file.write(content)
