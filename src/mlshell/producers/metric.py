"""
The :mod:`mlshell.producers.metric` contains examples of `Metric` class to make
empty metric object and `MetricProducer` class to fill it.

:class:`mlshell.Metric` proposes unified interface to work with underlying
scorer. Intended to be used in :class:`mlshell.Workflow`. For new metric
formats no need to edit `Workflow` class, just adapt `Metric` interface logic.

:class:`mlshell.MetricProducer` specifies methods to make metric from custom
function. Current implementation inherits :func:`sklearn.metrics.make_scorer`
logic.

"""


import pycnfg
import numpy as np
import pandas as pd
import sklearn
import tabulate

__all__ = ['Metric', 'MetricProducer']


class Metric(object):
    """Unified pipeline interface.

    Implements interface to access arbitrary scorer.
    Interface: pprint and all underlying scorer methods.

    Attributes
    ----------
    scorer: callable
        Underlying scorer.
    oid : str
        Instance identifier.
    score_func: callable
        Scorer score function, return scalar value.
    score_func_vector: callable
        Scorer vectorized score function, return vector of values for all
        samples.
    greater_is_better : bool, optional (default=True)
        Whether `score_func` is a score function (default), meaning high
        is good, or a loss function, meaning low is good. In the latter
        case, the scorer object should sign-flip the outcome of the
        `score_func`.
    needs_proba : bool, optional (default=False)
        Whether `score_func` requires predict_proba to get probability
        estimates out of a classifier.
    needs_threshold : bool, default=False
        Whether `score_func` takes a continuous decision certainty.
        This only works for classification using estimators that
        have either a decision_function or predict_proba method.
    needs_custom_kwargs : bool, optional (default=False)
        If True, before score evaluation extract scorer kwargs from pipeline
        'pass_custom' step (if existed).

    Notes
    -----
    Extended :term:`sklearn:scorer` object:

    * Additional ``needs_custom_kwargs`` kwarg.
     Allows to optimize custom scorer kwargs as hyper-parameters.
    * Additional ``score_func_vector`` kwarg.
     Allows to evaluate vectorized score for more detailed analyze.

    """
    def __init__(self, scorer=None, oid=None, score_func=None,
                 score_func_vector=None, greater_is_better=True,
                 needs_proba=False, needs_threshold=False,
                 needs_custom_kwargs=False):
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
        """dict: Additional kwargs passed to `score_func`."""
        # Unchanged if no `pass_custom` step in pipeline.
        return self.scorer._kwargs

    def pprint(self, score):
        """Pretty print metric result.

        Parameters
        ----------
        score : any object
            `score_func` output.

        Returns
        -------
        score : str
            Input converted to string.

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
    """Factory to produce metric.

    Interface: make.

    Parameters
    ----------
    objects : dict
        Dictionary with resulted objects from previous executed producers:
        {'section_id__config__id', object,}.
    oid : str
        Unique identifier of produced object.
    path_id : str
        Project path identifier in `objects`.
    logger_id : str
        Logger identifier in `objects`.

    Attributes
    ----------
    objects : dict
        Dictionary with resulted objects from previous executed producers:
        {'section_id__config__id', object,}.
    oid : str
        Unique identifier of produced object.
    logger : :class:`logging.Logger`
        Logger.
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
        scorer : :class:`mlshell.Metric`
            Scorer object, will be updated.
        score_func : callback or str
            Custom function or key from :data:`sklearn.metrics.SCORERS` .
        score_func_vector: callback, optional (default=None)
            Vectorized `score_func` returning vector of values for all samples.
            Mainly for result visualization purpose.
        needs_custom_kwargs : bool, optional (default=False)
            If True, before score evaluation extract scorer kwargs from
            pipeline 'pass_custom' step (if existed).
        **kwargs : dict
            Additional kwargs to pass in :func:`sklearn.metrics.make_scorer`
            (if ``score_func`` is not str).

        Notes
        -----
        Extended :func:`sklearn.metrics.make_scorer` in compliance with
        :class:`mlshell.Metric` .

        """
        # Convert to scorer.
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
