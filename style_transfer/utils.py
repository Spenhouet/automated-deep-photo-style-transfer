"""
Created by sanzenba on 11/30/17
"""

import cv2


# save the layer activations of a given conv layer to the file system
def dump_layer_activation(conv_layer, filename_format):
    numFeatureMaps = conv_layer.shape[3]
    for i in range(numFeatureMaps):
        featureMap = conv_layer[0, :, :, i]
        cv2.imwrite(filename_format % i, featureMap)
