import numpy as np
import scipy.ndimage
import scipy.sparse
import scipy.sparse.linalg
import tensorflow as tf


def compute_matting_laplacian(image, consts=None, epsilon=1e-5, window_radius=1):
    print("Compute matting laplacian started")

    num_window_pixels = (window_radius * 2 + 1) ** 2
    height, width, channels = image.shape
    if consts is None:
        consts = np.zeros(shape=(height, width))

    # compute erosion with window square as mask
    consts = scipy.ndimage.morphology.grey_erosion(consts, footprint=np.ones(
        shape=(window_radius * 2 + 1, window_radius * 2 + 1)))

    num_image_pixels = width * height

    # value and index buffers for laplacian in COO format
    laplacian_indices = []
    laplacian_values = []

    # cache pixel indices in a matrix
    pixels_indices = np.reshape(np.array(range(num_image_pixels)), newshape=(height, width), order='F')

    # iterate over image pixels
    for y in range(window_radius, width - window_radius):
        for x in range(window_radius, height - window_radius):
            if consts[x, y]:
                continue

            window_x_start, window_x_end = x - window_radius, x + window_radius + 1
            window_y_start, window_y_end = y - window_radius, y + window_radius + 1
            window_indices = pixels_indices[window_x_start:window_x_end, window_y_start:window_y_end].ravel()
            window_values = image[window_x_start:window_x_end, window_y_start:window_y_end, :]
            window_values = window_values.reshape((num_window_pixels, channels))

            mean = np.mean(window_values, axis=0).reshape(channels, 1)
            cov = np.matmul(window_values.T, window_values) / num_window_pixels - np.matmul(mean, mean.T)

            tmp0 = np.linalg.inv(cov + epsilon / num_window_pixels * np.identity(channels))

            tmp1 = window_values - np.repeat(mean.transpose(), num_window_pixels, 0)
            window_values = (1 + np.matmul(np.matmul(tmp1, tmp0), tmp1.T)) / num_window_pixels

            ind_mat = np.broadcast_to(window_indices, (num_window_pixels, num_window_pixels))

            laplacian_indices.extend(zip(ind_mat.ravel(order='F'), ind_mat.ravel(order='C')))
            laplacian_values.extend(window_values.ravel())

    # create sparse matrix in coo format
    laplacian_coo = scipy.sparse.coo_matrix((laplacian_values, zip(*laplacian_indices)),
                                            shape=(num_image_pixels, num_image_pixels))

    # compute final laplacian
    sum_a = laplacian_coo.sum(axis=1).T.tolist()[0]
    laplacian_coo = (scipy.sparse.diags([sum_a], [0], shape=(num_image_pixels, num_image_pixels)) - laplacian_coo) \
        .tocoo()

    # create a sparse tensor from the coo laplacian
    indices = np.mat([laplacian_coo.row, laplacian_coo.col]).transpose()
    laplacian_tf = tf.to_float(tf.SparseTensor(indices, laplacian_coo.data, laplacian_coo.shape))

    return laplacian_tf
