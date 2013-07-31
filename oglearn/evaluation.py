"""Tools for model evaluation"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from sklearn.cross_validation import train_test_split
from sklearn.externals.joblib import Parallel, delayed


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


def _compute_scores(model, train_sizes, test_size, X, y, run_idx):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=run_idx, test_size=test_size)
    train_scores = np.empty(len(train_sizes), dtype=np.float)
    test_scores = np.empty(len(train_sizes), dtype=np.float)
    for ts_idx, train_size in enumerate(train_sizes):
        X_train_sub = X_train[:train_size]
        y_train_sub = y_train[:train_size]
        model.fit(X_train_sub, y_train_sub)
        train_scores[ts_idx] = model.score(X_train_sub, y_train_sub)
        test_scores[ts_idx] = model.score(X_test, y_test)
    return train_scores, test_scores


def learning_curves(model, X, y, n_cv_runs=5, steps=5, train_size=None,
                    test_size=0.1, n_jobs=1):
    """Compute train and test learning curves on subsamples of the data

    Return the arrays (train_sizes, train_scores, test_scores).
    Each score array has shape (n_steps, n_cv_runs).

    """
    n_samples = X.shape[0]
    if isinstance(steps, int):
        if train_size is None:
            train_size = (1 - test_size)
            max_train_size = int(train_size * n_samples)
        elif train_size > 1:
            # assume exact number of samples
            max_train_size = int(train_size)
        else:
            if train_size + test_size > 1.:
                raise ValueError(
                    ('The sum of train_size={} and test_size={}'
                     ' should be less than 1.0').format(train_size, test_size))
            max_train_size = int(train_size * n_samples)
        min_train_size = int(0.1 * n_samples)

        train_sizes = np.logspace(np.log10(min_train_size),
                                  np.log10(max_train_size),
                                  steps).astype(np.int)
    else:
        # assume precomputed steps
        train_sizes = np.asarray(steps)

    n_steps = len(train_sizes)
    train_scores = np.zeros((n_steps, n_cv_runs), dtype=np.float)
    test_scores = np.zeros((n_steps, n_cv_runs), dtype=np.float)

    results = Parallel(n_jobs=n_jobs)(
        delayed(_compute_scores)(model, train_sizes, test_size, X, y, run_idx)
        for run_idx in range(n_cv_runs))

    for run_idx, result in enumerate(results):
        train_scores[:, run_idx] = result[0]
        test_scores[:, run_idx] = result[1]

    return train_sizes, train_scores, test_scores


def plot_learning_curves(model, X, y, n_cv_runs=5, steps=5,
                         train_size=None, test_size=0.1, n_jobs=1):
    """Compute and plot learning curves.

    Return the arrays (train_sizes, train_scores, test_scores).
    Each score array has shape (n_steps, n_cv_runs).

    """
    train_sizes, train_scores, test_scores = learning_curves(
        model, X, y, n_cv_runs=n_cv_runs, steps=steps,
        train_size=train_size, test_size=test_size, n_jobs=n_jobs)

    plot_envelope(train_sizes, train_scores, label='Train', color='b')
    plot_envelope(train_sizes, test_scores, label='Test', color='g')
    plt.xlabel('Training set size')
    plt.ylabel('Score')
    plt.xlim(train_sizes[0], train_sizes[-1])
    plt.ylim(None, max(train_scores.max(), test_scores.max()) * 1.05)
    plt.legend(loc='best')
    return train_sizes, train_scores, test_scores


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



if __name__ == "__main__":
    from sklearn.datasets import load_digits
    from sklearn.svm import SVC
    import time

    digits = load_digits()
    tic = time.time()
    plot_learning_curves(SVC(gamma=0.01), digits.data, digits.target, steps=5,
                         n_jobs=-1)
    print("Computed and plotted learning curves in %0.3fs"
          % (time.time() - tic))
    plt.show()
