"""Tools for model evaluation"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from sklearn.cross_validation import train_test_split


def plot_envelope(x, y_values, label=None, color=None, marker='o'):
    """Plot a series of runs using min/max and stderr envelopes"""
    # Min-Max envelope
    plt.fill_between(x, np.min(y_values, axis=1), np.max(y_values, axis=1),
                     color=color, alpha=.1)

    # More opaque 2 standard error of the mean wide envelope
    y_mean = np.mean(y_values, axis=1)
    confidence = sem(y_values, axis=1) * 2
    plt.fill_between(x, y_mean - confidence, y_mean + confidence,
                     color=color, alpha=.2)

    # Individual values (runs) as dashed lines
    for i in range(y_values.shape[1]):
        plt.plot(x, y_values[:, i], '--', alpha=0.2, color=color)

    # Solid line and dots for the mean
    plt.plot(x, y_mean, marker + '-k', color=color, label=label)


def plot_learning_curves(model, X, y, n_cv_runs=10, n_steps=5, max_score=1.0,
                         test_size=0.1):
    """Compute and plot learning curves.

    Return the pair of arrays (train_scores, test_scores). Each array has
    shape (n_steps, n_cv_runs).

    """
    # TODO: move learning curve computations out in another public function
    # TODO: use joblib parallel
    n_samples = X.shape[0]
    max_train_size = int((1 - test_size) * n_samples)
    min_train_size = int(0.1 * n_samples)
    n_steps = 5

    train_sizes = np.logspace(np.log10(min_train_size),
                              np.log10(max_train_size),
                              n_steps).astype(np.int)

    train_scores = np.zeros((n_steps, n_cv_runs), dtype=np.float)
    test_scores = np.zeros((n_steps, n_cv_runs), dtype=np.float)

    for run_idx in range(n_cv_runs):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, random_state=run_idx, test_size=test_size)

        for ts_idx, train_size in enumerate(train_sizes):
            X_train_sub = X_train[:train_size]
            y_train_sub = y_train[:train_size]
            model.fit(X_train_sub, y_train_sub)
            train_scores[ts_idx, run_idx] = model.score(X_train_sub,
                                                        y_train_sub)
            test_scores[ts_idx, run_idx] = model.score(X_test, y_test)

    plot_envelope(train_sizes, train_scores, label='Train', color='b')
    plot_envelope(train_sizes, test_scores, label='Test', color='g')
    plt.xlabel('Training set size')
    plt.ylabel('Score')
    plt.xlim(train_sizes[0], train_sizes[-1])
    plt.ylim((None, max_score))
    plt.legend(loc='best')
    return train_scores, test_scores


def _display_scores(params, scores, append_star=False):
    """Format the mean score +/- std error for params"""
    params = ", ".join("{0}={1}".format(k, v)
                       for k, v in params.items())
    score_line = "mean: {:.3f} (+/-{:.3f}) stdev: {:.3f}".format(
        params, np.mean(scores), sem(scores), np.std(scores))
    if append_star:
        score_line += " *"
    score_line += "\n" + params
    return score_line


def display_grid_scores(grid_scores, top=None):
    """Helper function to format a report on a grid of scores"""

    grid_scores = sorted(grid_scores, key=lambda x: x[1], reverse=True)
    if top is not None:
        grid_scores = grid_scores[:top]

    # Compute a threshold for staring models with overlapping
    # stderr:
    _, best_mean, best_scores = grid_scores[0]
    threshold = best_mean - 2 * sem(best_scores)

    for params, mean_score, scores in grid_scores:
        append_star = mean_score + 2 * sem(scores) > threshold
        print(_display_scores(params, scores, append_star=append_star))
