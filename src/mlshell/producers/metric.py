"""

`mlshell.Scorer also allow to pass custom kwargs while grid search.

Notes
-----


Scorer object need ._kwargs for pass_csutom, that is all, sklearn mainly
in Producer defined.
"""


import mlshell.pycnfg as pycnfg
import sklearn
import tabulate
import pandas as pd
import numpy as np


class Metric(object):
    def __init__(self, scorer=None, oid=None, score_func=None,
                 greater_is_better=True, needs_proba=False,
                 needs_threshold=False, needs_custom_kwargs=False):
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
            Scorer underlying score function.
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
        kwargs : dict
            Additional kwargs passed to `score_func` (unchanged if step
            `pass_custom` not used).

        """
        self.scorer = scorer
        self.score_func = score_func
        self.oid = oid
        # Flags.
        self.greater_is_better = greater_is_better
        self.needs_proba = needs_proba
        self.needs_threshold = needs_threshold
        self.needs_cutom_kwargs = needs_custom_kwargs
        # [deprecated] not necessery to storage init.
        # they are in scorer._kwargs
        # self.kwargs = kwargs
        # **kwargs : dict
        #    Additional kwargs, passed to func (initially).

    @property
    def kwargs(self):
        return self.scorer._kwargs

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

    def _set_custom_kwargs(self, estimator):
        # Allow to get custom kwargs
        # [deprecated] not need if recover pass_custom.
        # self.scorer._kwargs.update(self.init_kwargs)
        if hasattr(estimator, 'steps'):
            for step in estimator.steps:
                if step[0] == 'pass_custom':
                    temp = step[1].kwargs.get(self.oid, {})
                    self.scorer._kwargs.update(temp)

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

    def make(self, scorer, score_func=None, needs_custom_kwargs=False,
             **kwargs):
        """Make scorer from metric callable.

        Parameters
        ----------
        scorer : mlshell.Scorer interface
        score_func : callback or str
            Custom function or sklearn built-in metric name.
        needs_custom_kwargs : bool
            If True, allow to pass kwargs while grid search for custom metric.
        **kwargs : dict
            Additional kwargs to pass in make_scorer (if not str `func`).

        """
        if score_func is None:
            raise ValueError('Specify metric function')

        # Convert to scorer
        if isinstance(score_func, str):
            # built_in = sklearn.metrics.SCORERS.keys().
            # Ignore kwargs, built-in `str` metrics has hard-coded kwargs.
            scorer.scorer = sklearn.metrics.get_scorer(score_func)
        else:
            # Non-scalar output metric also possible.
            scorer.scorer = sklearn.metrics.make_scorer(score_func, **kwargs)
            # [deprecated] combined ExtendedScorer and Scorer
            # if needs_custom_kwargs:
            #     # Create special object.
            #     custom_scorer = sklearn.metrics.make_scorer(func, **kwargs)
            #     scorer = ExtendedScorer(custom_scorer)
            #     # [alternative] Rewrite _BaseScorer.
            # else:
            #     scorer.scorer = sklearn.metrics.make_scorer(func, **kwargs)
        scorer.score_func = score_func
        scorer.needs_custom_kwargs = needs_custom_kwargs
        scorer.greater_is_better = scorer.scorer._sign > 0
        scorer.needs_proba =\
            isinstance(scorer.scorer, sklearn.metrics._scorer._ProbaScorer)
        scorer.needs_threshold =\
            isinstance(scorer.scorer, sklearn.metrics._scorer._ThresholdScorer)
        scorer.needs_cutom_kwargs = needs_custom_kwargs
        # [deprecate]
        # scorer.init_kwargs = copy.deepcopy(scorer.scorer._kwargs)
        return scorer


class Validator(object):
    def __init__(self):
        pass

    def validate(self, pipeline, metrics, datasets, logger, method='metric'):
        """Evaluate pipeline metrics.

        Parameters
        ----------
        pipeline : sklearm.pipeline.Pipeline interface
            model
        metrics : list of mlshell.Metric
            Metrics to evaluate.
        datasets : list of mlshell.Dataset
            Dataset to evaluate on.
        method : 'metric' or 'scorer'
            If 'metric', efficient (reuse y_pred) evaluation via
            `score_func(y, y_pred, **kwargs)`. If 'scorer', evaluate via
            `scorer(pipeline, x, y)`.
        logger : logger object
            Logs.

        """
        if not metrics:
            logger.warning("Warning: no metrics to evaluate.")
            return
        if method not in ['metric', 'scorer']:
            raise ValueError("Unknown 'method' value.")

        # Storage to prevent multiple inference (via metric).
        infer = {}
        for dataset in datasets:
            infer[dataset.oid] = {
                'predict_proba': None,
                'decision_function': None,
                'predict': None
            }
        for metric in metrics:
            logger.log(5, f"{metric.oid}:")
            for dataset in datasets:
                x = dataset.get_x()
                y = dataset.get_y()
                try:
                    if method == 'metric':
                        score = self._via_metric(pipeline, x, y, metric,
                                                 dataset, infer[dataset.oid])
                    elif method == 'scorer':
                        score = metric(pipeline, x, y)
                    else:
                        assert False
                except AttributeError as e:
                    # Pipeline has not 'predict_proba'/'decision_function'.
                    logger.warning(f"Ignore metric: {e}")
                    break
                score = metric.pprint(score)
                logger.log(5, f"{dataset.oid}:\n    {score}")
        return

    def _via_metric(self, pipeline, x, y, metric, dataset, infer):
        """Evaluate score via score functions.

        Reutilize inference, more efficient than via scorers.

        """
        y_pred = self._get_y_pred(pipeline, x, metric, infer, dataset)
        # Update metric kwargs with pass_custom kwarg from pipeline.
        if getattr(metric, 'needs_custom_kwargs', False):
            if hasattr(pipeline, 'steps'):
                for step in pipeline.steps:
                    if step[0] == 'pass_custom':
                        temp = step[1].kwargs.get(metric.oid, {})
                        metric.kwargs.update(temp)
        # Score.
        score = metric.score_func(y, y_pred, **metric.kwargs)
        return score

    def _get_y_pred(self, pipeline, x, metric, infer, dataset):
        if getattr(metric, 'needs_proba', False):
            # [...,i] equal to [:,i]/[:,:,i]/.. (for multi-output target)
            if not infer['predict_proba']:
                # Pipeline predict_proba shape would be based on train
                # (pos_labels_ind/classes not guaranteed in test).
                pos_labels_ind = dataset.get_classes()
                # For multi-output return list of arrays.
                pp = pipeline.predict_proba(x)
                if isinstance(pp, list):
                    y_pred = [i[..., pos_labels_ind] for i in pp]
                else:
                    y_pred = pp[..., pos_labels_ind]
                infer['predict_proba'] = y_pred
            else:
                y_pred = infer['predict_proba']
        elif getattr(metric, 'needs_threshold', False):
            if not infer['decision_function']:
                y_pred = pipeline.decision_function(x)
                infer['decision_function'] = y_pred
            else:
                y_pred = infer['decision_function']
        else:
            if not infer['predict']:
                y_pred = pipeline.predict(x)
                infer['predict'] = y_pred
            else:
                y_pred = infer['predict']
        return y_pred


if __name__ == '__main__':
    pass
