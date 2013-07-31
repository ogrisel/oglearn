from collections import namedtuple
import numpy as np

from sklearn.utils import check_random_state
from sklearn.metrics.scorer import SCORERS
from sklearn.metrics.scorer import _ProbaScorer
from sklearn.metrics.scorer import _ThresholdScorer


# TODO: write a class with a nice __repr__ and also store the raw scores
score_summary = namedtuple('score_summary',
                           ('mean', 'std', 'min', 'median', 'max'))


class BootstrapScorer(object):

    def __init__(self, scoring, n_bootstraps=100, random_state=None,
                 kwargs=None):
        # Unfortunately the new scorer API in sklearn 0.14 forces us to use
        # introspection of private attributes and types to be able to reverse
        # engineer the scoring properties without code duplication
        scorer = SCORERS[scoring]
        self._score_func = scorer._score_func
        self._sign = scorer._sign
        self.name = scoring
        self.needs_proba = isinstance(scorer, _ProbaScorer)
        self.needs_threshold = isinstance(scorer, _ThresholdScorer)
        self.scoring = scoring
        self.n_bootstraps = n_bootstraps
        self.random_state = random_state
        self.kwargs = kwargs if kwargs is not None else {}

    def __call__(self, model, X, y):
        n_samples = len(y)
        if self.needs_proba:
            y_predicted = model.predict_proba(X)
        elif self.needs_threshold:
            y_predicted = model.decision_function(X)
        else:
            y_predicted = model.predict(X)

        # TODO: move the bootstraping logic and summary in a helper function
        rng = check_random_state(self.random_state)
        scores = []
        for i in range(self.n_bootstraps):
            idx = rng.randint(low=0, high=n_samples, size=n_samples)
            score = self._sign * self._score_func(
                y[idx], y_predicted[idx], **self.kwargs)
            scores.append(score)

        # XXX: use a more efficient way to compute the percentiles by sorting
        # only once instead
        scores = np.array(scores)
        return score_summary(np.mean(scores),
                             np.std(scores),
                             np.min(scores),
                             np.median(scores),
                             np.max(scores))

    def __repr__(self):
        return "boostrap_scorer(%s, %d)" % (self.scoring, self.n_bootstraps)


# Alias function: is it really usefull?

def bootstrap_scorer(name, n_bootstraps=100, random_state=None):
    return BootstrapScorer(name, n_bootstraps=n_bootstraps,
                           random_state=random_state)


if __name__ == "__main__":
    from sklearn.svm import SVC
    from sklearn.datasets import load_digits
    from sklearn.cross_validation import train_test_split

    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data, digits.target, random_state=0)

    clf = SVC(gamma=0.01).fit(X_train, y_train)

    summary = bootstrap_scorer('f1')(clf, X_train, y_train)
    print('f1 train score:')
    print(summary)

    summary = bootstrap_scorer('f1')(clf, X_test, y_test)
    print('f1 test score:')
    print(summary)
