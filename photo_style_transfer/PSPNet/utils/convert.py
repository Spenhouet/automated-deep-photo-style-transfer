import scipy.io
import argparse


def extract_colors(mat_filename):
    mat = scipy.io.loadmat(mat_filename)
    colors = mat['colors']
    return colors

def extract_annotations(annotations_file):
    with open(annotations_file, 'r+') as file:
        content = file.read()
        lines = content.split('\n')[1:]
        assert(len(lines) == 150)

        for line in lines:
            cells = line.split('\t')
            class_names = cells[-1]
            yield class_names

if __name__ == '__main__':

    # currently only support for ade20k
    supported_datasets = ['ade20k']

    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset name (supported: %s)' % ', '.join(supported_datasets), default='ade20k')
    args = parser.parse_args()

    assert(args.dataset in supported_datasets)

    colors_filename = args.dataset + '_colors.mat'
    annotations_filename = args.dataset + '_annotations.txt'

    colors = [str(tuple(color)) for color in extract_colors(colors_filename)]
    annotations = extract_annotations(annotations_filename)

    labels = zip(colors, annotations)

    with open(args.dataset + '_labels.txt', 'w+') as file:
        content = '\n'.join(['\t'.join(pair) for pair in labels])
        file.write(content)


