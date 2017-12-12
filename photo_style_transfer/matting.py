from __future__ import division

import numpy as np
import scipy.sparse
from numpy.lib.stride_tricks import as_strided
import tensorflow as tf


def compute_matting_laplacian(image, consts=None, epsilon=1e-5, win_rad=1):

    neb_size = (win_rad * 2 + 1) ** 2
    h, w, c = image.shape
    if consts is None:
        consts = np.zeros(shape=(h, w))
    img_size = w * h
    consts = scipy.ndimage.morphology.grey_erosion(consts, footprint=np.ones(shape=(win_rad * 2 + 1, win_rad * 2 + 1)))

    indsM = np.reshape(np.array(range(img_size)), newshape=(h, w), order='F')
    tlen = int((-consts[win_rad:-win_rad, win_rad:-win_rad] + 1).sum() * (neb_size ** 2))
    row_inds = np.zeros(tlen)
    col_inds = np.zeros(tlen)
    vals = np.zeros(tlen)
    l = 0
    for j in range(win_rad, w - win_rad):
        for i in range(win_rad, h - win_rad):
            if consts[i, j]:
                continue
            win_inds = indsM[i - win_rad:i + win_rad + 1, j - win_rad: j + win_rad + 1]
            win_inds = win_inds.ravel(order='F')
            win_i = image[i - win_rad:i + win_rad + 1, j - win_rad: j + win_rad + 1, :]
            win_i = win_i.reshape((neb_size, c), order='F')
            win_mu = np.mean(win_i, axis=0).reshape(c, 1)
            win_var = np.linalg.inv(
                np.matmul(win_i.T, win_i) / neb_size - np.matmul(win_mu, win_mu.T) + epsilon / neb_size * np.identity(
                    c))

            win_i2 = win_i - np.repeat(win_mu.transpose(), neb_size, 0)
            tvals = (1 + np.matmul(np.matmul(win_i2, win_var), win_i2.T)) / neb_size

            ind_mat = np.broadcast_to(win_inds, (neb_size, neb_size))
            row_inds[l: (neb_size ** 2 + l)] = ind_mat.ravel(order='C')
            col_inds[l: neb_size ** 2 + l] = ind_mat.ravel(order='F')
            vals[l: neb_size ** 2 + l] = tvals.ravel(order='F')
            l += neb_size ** 2

    vals = vals.ravel(order='F')[0: l]
    row_inds = row_inds.ravel(order='F')[0: l]
    col_inds = col_inds.ravel(order='F')[0: l]
    laplacian_csr = scipy.sparse.csr_matrix((vals, (row_inds, col_inds)), shape=(img_size, img_size))

    sum_a = laplacian_csr.sum(axis=1).T.tolist()[0]
    laplacian_csr = scipy.sparse.diags([sum_a], [0], shape=(img_size, img_size)) - laplacian_csr

    laplacian_coo = laplacian_csr.tocoo()
    indices = np.mat([laplacian_coo.row, laplacian_coo.col]).transpose()
    laplacian_tf = tf.to_float(tf.SparseTensor(indices, laplacian_coo.data, laplacian_coo.shape))
    return laplacian_tf