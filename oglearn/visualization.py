"""Helpers for data visualization"""

from itertools import cycle

import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import atleast2d_or_csr
from sklearn.decomposition import RandomizedPCA


COLORS = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
MARKERS = ['+', 'o', '^', 'v', '<', '>', 'D', 'h', 's']


def scatter(data, labels=None, title=None, name=None):
    """2d PCA scatter plot with optional class info

    Return the pca model to be able to introspect the components or transform
    new data with the same model.
    """
    data = atleast2d_or_csr(data)

    if data.shape[1] == 2:
        # No need for a PCA:
        data_2d = data
    else:
        pca = RandomizedPCA(n_components=2)
        data_2d = pca.fit_transform(data)

    for i, c, m in zip(np.unique(labels), cycle(COLORS), cycle(MARKERS)):
        plt.scatter(data_2d[labels == i, 0], data_2d[labels == i, 1],
                    c=c, marker=m, label=i, alpha=0.5)

    plt.legend(loc='best')
    if title is None:
        title = "2D PCA scatter plot"
        if name is not None:
            title += " for " + name
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.title(title)

    return pca


def gallery(data, shape=None, labels=None, title=None, n_rows=8, n_cols=12,
            interpolation='nearest', figsize=None):
    """Plot a gallery of samples"""
    data = np.atleast_2d(data)
    if shape is not None and len(shape) != 1:
        raise ValueError("Expected a 2d shape, got %r" % (shape,))
    if data.ndim == 2:
        if shape is None:
            raise ValueError("shape is required for 2d data input")
        data = data.reshape((-1,) + tuple(shape))
    elif data.ndim == 3:
        # no need to reshape but enforce gray level coding
        plt.gray()
    elif data.ndim == 4:
        if data.shape[3] == 1:
            plt.gray()
    else:
        raise ValueError('Unsupported data input shape: %r' % data.shape)

    if figsize is None:
        figsize = (1.1 * n_cols, 1.2 * n_rows)
    plt.figure(figsize=figsize)
    plt.title(title)
    for i in range(n_rows):
        for j in range(n_cols):
            idx = i * n_cols + j
            if idx < len(data):
                label = labels[idx] if labels is not None else None
                plt.subplot(n_rows, n_cols, i * n_cols + j, title=label)
                plt.imshow(data[idx], interpolation=interpolation)
                plt.axis('off')


if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.datasets import fetch_olivetti_faces

    digits = load_digits()
    olivetti = fetch_olivetti_faces()

    pca_model = scatter(digits.data, digits.target, name='digits')
    gallery(olivetti.images, labels=olivetti.target,
            title="Olivetti Faces (subsample)")

    plt.show()
