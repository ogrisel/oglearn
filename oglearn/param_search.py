"""Utilities to perform routine parameter search on some sklearn models

The grid ranges are somewhat arbitrary based on the task I tend to
work with most often.

Note: RandomForest* and ExtraTrees* doe not really need grid search: the most
important parameter is n_estimators and usually the higher the better.

"""

import numpy as np
from scipy.stats import sem


def make_fast_grid(granularity=5):
    return {
        'SVC': {
            'kernel': ['rbf'],
            'C': np.logspace(-3, 3, granularity),
            'gamma': np.logspace(-3, 3, granularity),
        },
        'PassiveAggressiveClassifier': {
            'C': np.logspace(-2, 2, granularity),
        },
        'SGDClassifier': {
            'alpha': np.logspace(-6, 0, granularity),
        },
        'SGDRegressor': {
            'alpha': np.logspace(-6, 0, granularity),
        },
        'MultinomialNB': {
            'alpha': np.logspace(-2, 2, granularity),
        },
    }


FAST_GRIDS = make_fast_grid(5)


# Slow grid are better suited for RandomizedSearchCV
SLOW_GRIDS = make_fast_grid(7)
SLOW_GRIDS.update({
    'TfidfVectorizer': {
        'use_idf': [True, False],
        'min_df': [1, 2, 3],
        'max_df': [0.5, 0.85, 1],
        'ngram_range': [(1, 1), (1, 2), (1, 3)],
        'binary': [True, False],
    },
    'SGDClassifier': {
        'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge'],
        'penalty': ['l1', 'l2', 'elasticnet'],
        'alpha': np.logspace(-6, 0, 7),
        'n_iter': [1, 5, 10, 50],
    },
    'SGDRegressor': {
        'loss': ['squared_loss', 'huber', 'epsilon_insensitive',
                 'squared_epsilon_insensitive'],
        'alpha': np.logspace(-6, 0, 7),
        'n_iter': [1, 5, 10, 50],
    },
    # TODO: add more as we go
})


def _prefix_grid(grid, prefix):
    """Prepend prefix__ to all the parameter name in grid

    Return a copy, the original grid is left untouched
    """
    if not isinstance(grid, dict):
        # Recursively reached a concrete value
        return grid
    new_grid = {}
    for key, value in grid.items():
        new_grid[prefix + '__' + key] = _prefix_grid(value, prefix)
    return new_grid


def get_grid(model, all_grids='fast'):
    grid = {}
    if not isinstance(all_grids, dict):
        all_grids = SLOW_GRIDS if all_grids == 'slow' else FAST_GRIDS

    # Recursive introspection of Pipeline
    # XXX: shall we use get_params here instead to get support for
    # FeatureUnion and possibly ensemble methods?
    if hasattr(model, 'steps'):
        for name, submodel in model.steps:
            subgrid = get_grid(submodel, all_grids=all_grids)
            grid.update(_prefix_grid(subgrid, name))
    else:
        # TODO: add support for list-based grids
        model_name = model.__class__.__name__
        grid.update(all_grids[model_name])
    return grid


def _display_scores(params, scores, append_star=False):
    """Format the mean score +/- std error for params"""
    params = ", ".join("{0}={1}".format(k, v)
                       for k, v in params.items())
    score_line = "mean: {:.3f} (+/-{:.3f}) stdev: {:.3f}".format(
        np.mean(scores), sem(scores), np.std(scores))
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
    from sklearn.pipeline import Pipeline
    from sklearn.svm import SVC
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier
    from pprint import pprint

    print("Fast default grid for SVC")
    pprint(get_grid(SVC()))

    print("Typical grid for text classification")
    p = Pipeline([
        ('vectorizer', TfidfVectorizer()),
        ('classifier', SGDClassifier()),
    ])
    pprint(get_grid(p, all_grids='slow'))
