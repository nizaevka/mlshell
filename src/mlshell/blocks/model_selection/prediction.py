"""
The :mod:`mlshells.model_selection.prediction` includes auxiliary
meta-estimators.

:class:`mlshell.model_selection.PredictionTransformer` calls predict_proba on
transform.
:class:`mlshell.model_selection.ThresholdClassifier` applies classification
threshold.
:class:`mlshell.model_selection.MockClassifier` and :class:`mlshell.
model_selection.MockRegressor` always predicts features.

"""

import sklearn
import numpy as np

__all__ = ['MockClassifier', 'MockRegressor', 'PredictionTransformer',
           'ThresholdClassifier']


class MockClassifier(sklearn.base.BaseEstimator, sklearn.base.ClassifierMixin):
    """Estimator always predicts train feature."""
    def __init__(self):
        pass

    def fit(self, x, y, **fit_params):
        return self

    def predict(self, x):
        return x


class MockRegressor(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """Estimator always predicts features."""
    def __init__(self):
        pass

    def fit(self, x, y, **fit_params):
        return self

    def predict(self, x):
        return x


class PredictionTransformer(sklearn.base.BaseEstimator,
                            sklearn.base.TransformerMixin,
                            sklearn.base.MetaEstimatorMixin):

    """Transformer applies predict_proba on features.

    Parameters
    ----------
    classifier : classifier object
        Classifier supported predict_proba.

    """
    def __init__(self, classifier):
        self.classifier = classifier

    def fit(self, x, y, **fit_params):
        self.classifier.fit(x, y, **fit_params)
        return self

    def transform(self, x):
        return self.classifier.predict_proba(x)


class ThresholdClassifier(sklearn.base.BaseEstimator,
                          sklearn.base.ClassifierMixin):
    """Estimator applies classification threshold.

    Classify samples based on whether they are above of below `threshold`.
    Awaits for prediction probabilities in features.

    Parameters
    ----------
    params : dict
        Parameters combined in dictionary to set together. {

        'classes': list of :class:`numpy.ndarray`
            List of sorted unique labels for each target(s) (n_outputs,
            n_classes).
        'pos_labels': list
            List of "positive" label(s) for target(s) (n_outputs,).
        'pos_labels_ind': list
            List of "positive" label(s) index in np.unique(target) for
            target(s) (n_outputs).
        }

    threshold : float [0,1], list of float [0,1], optional(default=None)
        Classification threshold. For multi-output target list of [n_outputs].
        If None, :func:`numpy.argmax` (in binary case equivalent to 0.5).
        If positive class probability P(pos_label) = 1 - P(neg_labels) > th_
        for some sample, classifier predict pos_label, else label in neg_labels
        with max probability.


    Notes
    -----
    Will be replaced with:
        https://github.com/scikit-learn/scikit-learn/pull/16525.

    """
    def __init__(self, params, threshold=None):
        self.params = params
        self.threshold = threshold

    @property
    def classes_(self):
        # Mandatory for classifiers (attribute or property).
        classes = self.params['classes']
        return classes if len(classes) > 1 else classes[0]

    def fit(self, x, y, **fit_params):
        return self

    def predict(self, x):
        # x - predict_proba (n_outputs, n_samples, n_classes).
        classes = self.params['classes']
        pos_labels = self.params['pos_labels']
        pos_labels_ind = self.params['pos_labels_ind']
        if any(not isinstance(i, np.ndarray) for i in classes):
            raise ValueError("Target classes should be numpy.ndarray.")

        if not isinstance(x, list):
            # Compliance to multi-output.
            x = [x]
            threshold = [self.threshold]
        else:
            if len(x) != len(classes):
                # Probably come from MockOptimizer
                # todo: problem if n_samples=n_outputs, assert.
                x = [i.T for i in x]  # list(zip(*x))
                assert len(x) == len(classes), "Multi-output inconsistent."
            threshold = self.threshold if self.threshold is not None \
                else [None] * len(x)

        res = []
        for i in range(len(x)):
            # Check that each train fold contain all classes, otherwise can`t
            # predict_proba, since can`t reconstruct probabilities (add zero),
            # cause don`t know which one class is absent.
            n_classes = x[i].shape[1]
            if n_classes != classes[i].shape[0]:
                raise ValueError("Train folds missed some classes:\n"
                                 "    ThresholdClassifier "
                                 "can`t identify class probabilities.")

            if threshold[i] is None:
                # Take the one with max probability.
                res.append(classes[i].take(np.argmax(x[i], axis=1), axis=0))
            else:
                if pos_labels_ind[i] < 0:
                    raise ValueError("pos_labels_ind values should be "
                                     "non-negative.")
                if n_classes > 2:
                    # Multi-class classification.
                    # Remain label with max prob.
                    mask = np.arange(n_classes) != pos_labels_ind[i]
                    remain_x = x[i][..., mask]
                    remain_classes = classes[i][mask]
                    neg_labels = remain_classes.take(np.argmax(remain_x,
                                                               axis=1), axis=0)
                else:
                    # Binary classification.
                    mask = np.arange(n_classes) != pos_labels_ind[i]
                    neg_labels = classes[i][mask]
                res.append(
                    np.where(x[i][..., pos_labels_ind[i]] > threshold[i],
                             [pos_labels[i]], neg_labels)
                )
        return np.array(res).T if len(res) > 1 else res[0]

    def predict_proba(self, x):
        return x


if __name__ == '__main__':
    pass
