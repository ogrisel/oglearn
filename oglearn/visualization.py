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


if __name__ == "__main__":
    from sklearn.datasets import load_digits

    digits = load_digits()
    pca_model = scatter(digits.data, digits.target, name='digits')
    plt.show()
