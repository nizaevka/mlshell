"""
The :mod:`mlshell.producers.metric` contains examples of `Metric` class to make
empty metric object and `MetricProducer` class to fillit.

`Metric` class proposes unified interface to work with underlying scorer.
Intended to be used in `mlshell.Workflow`. Notably, it allow to pass custom
kwargs while grid search. For new metric formats no need to edit `Workflow`
class, only update `Metric` interface logic.

`MetricProducer` class specifies methods to make metric from custom function.
Current implementation inherits sklearn.metric.make_scorer logic.

"""


import pycnfg
import numpy as np
import pandas as pd
import sklearn
import tabulate

__all__ = ['Metric', 'MetricProducer']


class Metric(object):
    def __init__(self, scorer=None, oid=None, score_func=None,
                 score_func_vector=None, greater_is_better=True,
                 needs_proba=False, needs_threshold=False,
                 needs_custom_kwargs=False):
        """Unified pipeline interface.

        Implements interface to access arbitrary scorer.
        Interface: pprint and all underlying scorer methods.

        Attributes
        ----------
        scorer: callable
            Scorer to pass in grid search optimizer.
        oid : str
            Instance identifier.
        score_func: callable
            Scorer`s underlying score function, return scalar value.
        score_func_vector: callable
            Scorer` underlying score function, return vector of values for all
            samples.
        greater_is_better : boolean, default=True
            Whether `score_func` is a score function (default), meaning high
            is good, or a loss function, meaning low is good. In the latter
            case, the scorer object should sign-flip the outcome of the
            `score_func`.
        needs_proba : boolean, default=False
            Whether `score_func` requires predict_proba to get probability
            estimates out of a classifier.
        needs_threshold : boolean, default=False
            Whether `score_func` takes a continuous decision certainty.
            This only works for binary classification using estimators that
            have either a decision_function or predict_proba method.
        needs_custom_kwargs : bool
            If True, allow to pass kwargs while grid search for custom metric.

        """
        self.scorer = scorer
        self.score_func = score_func
        self.score_func_vector = score_func_vector
        self.oid = oid
        # Flags.
        self.greater_is_better = greater_is_better
        self.needs_proba = needs_proba
        self.needs_threshold = needs_threshold
        self.needs_custom_kwargs = needs_custom_kwargs

    def __call__(self, estimator, *args, **kwargs):
        """Redirect call to scorer object."""
        if self.needs_cutom_kwargs:
            self._set_custom_kwargs(estimator)
        return self.scorer(estimator, *args, **kwargs)

    def __getattr__(self, name):
        """Redirect unknown methods to scorer object."""
        def wrapper(*args, **kwargs):
            getattr(self.scorer, name)(*args, **kwargs)
        return wrapper

    @property
    def kwargs(self):
        """dict: Additional kwargs passed to `score_func` (unchanged if step
        `pass_custom` not used)"""
        return self.scorer._kwargs

    def pprint(self, score):
        """Pretty print metric result.

        Parameters
        ----------
        score : arbitrary object
            `score_func` output.

        Returns
        -------
        score : str
            Ready to print result.

        """
        if self.score_func.__name__ == 'confusion_matrix':
            labels = self.scorer._kwargs.get('labels', None)  # classes
            score = tabulate.tabulate(
                pd.DataFrame(data=score, columns=labels, index=labels),
                headers='keys', tablefmt='psql'
            ).replace('\n', '\n    ')
        elif self.score_func.__name__ == 'classification_report':
            if isinstance(score, dict):
                score = tabulate.tabulate(
                    pd.DataFrame(score), headers='keys', tablefmt='psql'
                ).replace('\n', '\n    ')
            else:
                score = score.replace('\n', '\n    ')
        elif isinstance(score, np.ndarray):
            score = np.array2string(score, prefix='    ')
        return str(score)

    def _set_custom_kwargs(self, estimator):
        # Allow to get custom kwargs
        if hasattr(estimator, 'steps'):
            for step in estimator.steps:
                if step[0] == 'pass_custom':
                    temp = step[1].kwargs.get(self.oid, {})
                    self.scorer._kwargs.update(temp)


class MetricProducer(pycnfg.Producer):
    """Class includes methods to produce scorer.

    Interface: make.

    Parameters
    ----------
    objects : dict {'section_id__config__id', object,}
        Dictionary with resulted objects from previous executed producers.
    oid : str
        Unique identifier of produced object.
    path_id : str
        Project path identifier in `objects`.
    logger_id : str
        Logger identifier in `objects`.

    Attributes
    ----------
    objects : dict {'section_id__config__id', object,}
        Dictionary with resulted objects from previous executed producers.
    oid : str
        Unique identifier of produced object.
    logger : logger object
        Default logger logging.getLogger().
    project_path: str
        Absolute path to project dir.

    """
    _required_parameters = ['objects', 'oid', 'path_id', 'logger_id']

    def __init__(self, objects, oid, path_id, logger_id):
        pycnfg.Producer.__init__(self, objects, oid)
        self.logger = objects[logger_id]
        self.project_path = objects[path_id]

    def make(self, scorer, score_func, score_func_vector=None,
             needs_custom_kwargs=False, **kwargs):
        """Make scorer from metric callable.

        Parameters
        ----------
        scorer : mlshell.Scorer interface
        score_func : callback or str,
            Custom function or sklearn built-in metric name.
        score_func_vector: callback, optional (default=None)
            Vectorized `score_func` returning vector of values for all samples.
            Mainly for result visualization purpose.
        needs_custom_kwargs : bool
            If True, allow to pass kwargs while grid search for custom metric.
        **kwargs : dict
            Additional kwargs to pass in make_scorer (if not str `func`).

        """
        # Convert to scorer
        if isinstance(score_func, str):
            # built_in = sklearn.metrics.SCORERS.keys().
            # Ignore kwargs, built-in `str` metrics has hard-coded kwargs.
            scorer.scorer = sklearn.metrics.get_scorer(score_func)
        else:
            # Non-scalar output metric also possible.
            scorer.scorer = sklearn.metrics.make_scorer(score_func, **kwargs)
        scorer.score_func = score_func
        scorer.score_func_vector = score_func_vector
        scorer.needs_custom_kwargs = needs_custom_kwargs
        scorer.greater_is_better = scorer.scorer._sign > 0
        scorer.needs_proba =\
            isinstance(scorer.scorer, sklearn.metrics._scorer._ProbaScorer)
        scorer.needs_threshold =\
            isinstance(scorer.scorer, sklearn.metrics._scorer._ThresholdScorer)
        scorer.needs_custom_kwargs = needs_custom_kwargs
        return scorer


if __name__ == '__main__':
    pass
