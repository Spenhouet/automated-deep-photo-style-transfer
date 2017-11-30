import cv2


def save_layer_activations(conv_layer, filename_format):
    """Save the layer activations of a given conv layer to the file system"""
    num_feature_maps = conv_layer.shape[3]
    for i in range(num_feature_maps):
        feature_map = conv_layer[0, :, :, i]
        cv2.imwrite(filename_format % i, feature_map)
