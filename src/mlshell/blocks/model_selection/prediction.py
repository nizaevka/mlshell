"""
The :mod:`mlshell.blocks.model_selection.prediction` includes meta-estimators
to brute force classification threshold as pipeline hyper-parameter.

Notes
-----
Will be replaced with https://github.com/scikit-learn/scikit-learn/pull/16525.

"""

import sklearn
import numpy as np

__all__ = ['MockEstimator', 'PredictionTransformer', 'ThresholdClassifier']


class MockEstimator(sklearn.base.BaseEstimator):
    """Estimator always predict input features."""
    def __init__(self):
        pass

    def fit(self, x, y, **fit_params):
        return self

    def predict(self, x):
        return x


class PredictionTransformer(sklearn.base.BaseEstimator,
                            sklearn.base.TransformerMixin,
                            sklearn.base.MetaEstimatorMixin):
    """Transformer calls predict_proba on features."""

    def __init__(self, classifier):
        """Replaces features with `clf.predict_proba(x)`"""
        self.clf = classifier

    def fit(self, x, y, **fit_params):
        self.clf.fit(x, y, **fit_params)
        return self

    def transform(self, x):
        return self.clf.predict_proba(x)


class ThresholdClassifier(sklearn.base.BaseEstimator,
                          sklearn.base.ClassifierMixin):
    """Estimator to apply classification threshold.

    Classify samples based on whether they are above of below `threshold`.
    Awaits for predict_proba for features.

    Parameters
    ----------
    threshold : float [0,1], list of float [0,1], None, optional(default=None)
        Classification threshold. For multi-output target list of [n_outputs]
        awaited. If None, np.argmax, that is in binary case equivalent to 0.5.
        If positive class probability P(pos_label) = 1 - P(neg_labels) > th_
        for some sample, classifier predict pos_label for this sample, else
        next label in neg_labels with max probability.

    **kwargs : dict
        kwarg-layer need to set multiple params together in resolver/optimizer.
        {
        'classes': list of np.ndarray
            List of sorted unique labels for each target(s) (n_outputs,
            n_classes).
        'pos_labels': list
            List of "positive" label(s) for target(s) (n_outputs,).
        'pos_labels_ind': list
            List of "positive" label(s) index in np.unique(target) for
            target(s) (n_outputs).
        }

    """
    def __init__(self, threshold=None, **kwargs):
        self.classes = kwargs['classes']
        self.pos_labels = kwargs['pos_labels']
        self.pos_labels_ind = kwargs['pos_labels_ind']
        self.threshold = threshold
        if any(not isinstance(i, np.ndarray) for i in self.classes):
            raise ValueError("Each target 'classes' should be numpy.ndarray.")

    def fit(self, x, y, **fit_params):
        return self

    def predict(self, x):
        # x - predict_proba.
        if not isinstance(x, list):
            # Compliance to multi-output.
            x = [x]
            threshold = [self.threshold]
        else:
            threshold = self.threshold
        assert len(x) == len(self.classes), "Multi-output inconsistent."

        res = []
        for i in range(len(x)):
            # Check that each train fold contain all classes, otherwise can`t
            # predict_proba, since can`t reconstruct probabilities (add zero),
            # cause don`t know which one class is absent.
            n_classes = x[i].shape[1]
            if n_classes != self.classes[i].shape[0]:
                raise ValueError("Train folds missed some classes:\n"
                                 "    ThresholdClassifier "
                                 "can`t identify class probabilities.")

            if threshold[i] is None:
                # Take the one with max probability.
                res.append(self.classes[i].take(np.argmax(x[i],
                                                          axis=1), axis=0))
            else:
                if n_classes > 2:
                    # Multi-class classification.
                    # Remain label with max prob.
                    mask = np.arange(n_classes) != self.pos_labels_ind
                    remain_x = x[i][..., mask]
                    remain_classes = self.classes[i][mask]
                    neg_labels = remain_classes.take(np.argmax(remain_x,
                                                               axis=1), axis=0)
                else:
                    # Binary classification.
                    mask = np.arange(n_classes) != self.pos_labels_ind
                    neg_labels = self.classes[i][mask]
                res.append(
                    np.where(x[i][..., self.pos_labels_ind] > threshold[i],
                             [self.pos_labels], neg_labels)
                )
        return res if len(res) > 1 else res[0]

    def predict_proba(self, x):
        return x


if __name__ == '__main__':
    pass
