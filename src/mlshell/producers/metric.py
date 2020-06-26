"""

`mlshell.Scorer also allow to pass custom kwargs while grid search.

Notes
-----


Scorer object need ._kwargs for pass_csutom, that is all, sklearn mainly
in Producer defined.
"""


import copy
import mlshell.pycnfg as pycnfg
import sklearn


class Scorer(object):
    def __init__(self, scorer=None, sid=None, score_func=None,
                 greater_is_better=True, needs_proba=False,
                 needs_threshold=False, needs_custom_kwargs=False):
        """Unified pipeline interface.

        Implements interface to access arbitrary scorer.
        Interface: all underlying scorer methods.

        Attributes
        ----------
        scorer: callable
            Object for which wrapper is created.
        sid : str
            Scorer identifier.
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
        self.sid = sid
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
                    temp = step[1].kwargs.get(self.id, {})
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


#TODO: [deprecated]
class ExtendedScorer(object):
    def __init__(self, scorer):
        # Scorer to extend.
        self.scorer = scorer

        # [deprecated] now not memorize state, if no pass_csutom step, use default.
        # Last kwargs state to use in score for second stage optimizers.
        # self.cache_custom_kwargs = {}
        # TODO: here is actually problem when multiple pipeline are used.
        #   it is better to inheret pass_custom step for next level
        #   so the attribute error will never rise.

        self.init_kwargs = self.scorer._kwargs


    def __call__(self, estimator, x, y, **kwargs):
        """Read custom_kwargs from current pipeline, pass to scorer.

        Note:
            In gs self object copy, we can dynamically get param only from estimator.

            Use initial kwargs for score if:
                pipeline not contain steps
                no `pass_custom`
                kwargs empty {}.

        """
        # Use initial kwargs for score if pipeline not contain steps.
        self.scorer._kwargs.update(self.init_kwargs)
        if hasattr(estimator, 'steps'):
            if estimator.steps[0][0] == 'pass_custom':
                if estimator.steps[0][1].kwargs:
                    self.scorer._kwargs.update(estimator.steps[0][1].kwargs)
        # [deprecated] need tests.
        # except AttributeError:
        #     # ThresholdClassifier object has no attribute 'steps'.

        #     # [deprecated] Now use init kwargs in score,
        #     #   not last if no step or `pass_custom`.
        #     # self.scorer._kwargs.update(self.cache_custom_kwargs)
        #     pass

        return self.scorer(estimator, x, y, **kwargs)


class ScorerProducer(pycnfg.Producer):
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

    def make(self, scorer, score_func=None, needs_custom_kwargs=False, **kwargs):
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
            # Valid sklearn.metrics.SCORERS.keys().
            # Ignore kwargs, built-in `str` metrics has hard-coded kwargs.
            scorer.scorer = sklearn.metrics.get_scorer(score_func)
        else:
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
        scorer.sid = self.oid.split('__')[-1]
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


import operator
import tabulate
import pandas as pd
import numpy as np

class Validator(object):
    def validate(self, metrics, pipeline, train, test, **kwargs):
        """

        Parameters
        ----------
        metrics
        pipeline
        train
        test
        kwargs

        Returns
        -------

        """

        return

    def _via_metrics(self, metrics, pipeline, train, test, logger=logger, **kwargs):
        """Calculate ccore via score functions.

        Utilize inference, more efficient than use scorers.

        """
        if not metrics:
            logger.warning("Warning: no metrics to evaluate.")
            return

        x_train = train.get_x()
        y_train = train.get_y()
        x_test = test.get_x()
        y_test = test.get_y()
        # Storage to prevent multiple inference.
        infer = {'predict_proba': None,
                 'decision_function': None,
                 'predict': None}
        for name, metric in metrics.items():
            try:
                y_pred_train, y_pred_test = self._get_y_pred()
            except AttributeError as e:
                # Pipeline has not 'predict_proba', 'decision_function'.
                logger.warning(f"Ignore metric: {e}")
                continue

            # Update metric kwargs with pass_custom kwarg from pipeline.
            if getattr(metric, 'needs_custom_kwargs', False):
                if hasattr(pipeline, 'steps'):
                    for step in pipeline.steps:
                        if step[0] == 'pass_custom':
                            temp = step[1].kwargs.get(metric.id, {})
                            metric.kwargs.update(temp)

            # Score on train.
            score_train = metric.score_func(y_train, y_pred_train,
                                            **metric.kwargs)
            # Score on test.
            score_test = metric.score_func(y_test, y_pred_test,
                                           **metric.kwargs)

            score_train = metric.pprint(score_train)
            score_test = metric.pprint(score_test)
            score_train = self._pprint(metric, score_train, classes)
            score_test = self._pprint(metric, score_test, classes)
            logger.log(5, f"{name}:")
            logger.log(5, f"Train:\n    {score_train}\n"
                          f"Test:\n    {score_test}")

        return

    def _get_y_pred(self, name, metric, pipeline, infer, x_train, x_test,):
        if getattr(metric, 'needs_proba', False):
            # [...,i] equal to [:,i]/[:,:,i]/.. (for multi-output target)
            if not infer['predict_proba']:
                # Pipeline awaits for as much classes as in train.
                classes, pos_labels_ind = \
                    operator.itemgetter(
                        'classes',
                        'pos_labels_ind'
                    )(train.get_classes())
                # For multi-output return list of arrays.
                pp_train = pipeline.predict_proba(x_train)
                pp_test = pipeline.predict_proba(x_test)
                if isinstance(pp_train, list):
                    y_pred_train = [i[..., pos_labels_ind] for i in pp_train]
                else:
                    y_pred_train = pp_train[..., pos_labels_ind]
                if isinstance(pp_test, list):
                    y_pred_test = [i[..., pos_labels_ind] for i in pp_test]
                else:
                    y_pred_test = pp_test[..., pos_labels_ind]
                infer['predict_proba'] = (y_pred_train, y_pred_test)
            else:
                y_pred_train, y_pred_test = infer['predict_proba']
        elif getattr(metric, 'needs_threshold', False):
            if not infer['decision_function']:
                y_pred_train = pipeline.decision_function(x_train)
                y_pred_test = pipeline.decision_function(x_test)
                infer['decision_function'] = (y_pred_train, y_pred_test)
            else:
                y_pred_train, y_pred_test = infer['decision_function']
        else:
            if not infer['predict']:
                y_pred_train = pipeline.predict(x_train)
                y_pred_test = pipeline.predict(x_test)
                infer['predict'] = (y_pred_train, y_pred_test)
            else:
                y_pred_train, y_pred_test = infer['predict']
        return y_pred_train, y_pred_test

    def _pprint(self, metric, score, classes):
        if metric.score_func.__name__ == 'confusion_matrix':
            labels = metric[1].get('labels', classes)
            score = tabulate.tabulate(
                pd.DataFrame(data=score, columns=labels, index=labels),
                headers='keys', tablefmt='psql'
            ).replace('\n', '\n    ')
        elif metric.score_func.__name__ == 'classification_report':
            if isinstance(score, dict):
                score = tabulate.tabulate(
                    pd.DataFrame(score), headers='keys', tablefmt='psql'
                ).replace('\n', '\n    ')
            else:
                score = score.replace('\n', '\n    ')
        elif isinstance(score, np.ndarray):
            # pretty printing numpy arrays
            score = np.array2string(score, prefix='    ')
        return score

    # [deprecated] need rearrange
    # def _via_scorers(self, scorers, pipeline,
    #                  train, test, pos_labels_ind, classes):
    #     # via scorers
    #     # upside: sklearn consistent
    #     # downside:
    #     #   requires multiple inference
    #     #   non-score metric not possible (confusion_matrix)

    #     for name, scorer in scorers.items():
    #         self.logger.log(25, f"{name}:")
    #         self.logger.log(5, f"{name}:")
    #         # result score on Train
    #         score_train = scorer(pipeline, train.get_x(), train.get_y())
    #         # result score on test
    #         score_test = scorer(pipeline, test.get_x(), test.get_y())
    #         self.logger.log(25, f"Train:\n    {score_train}\n"
    #                             f"Test:\n    {score_test}")
    #         self.logger.log(5, f"Train:\n    {score_train}\n"
    #                            f"Test:\n    {score_test}")
    #     # non-score metrics
    #     add_metrics = {name: self.p['metrics'][name] for name in self.p['metrics'] if name not in scorers}
    #     self._via_metrics(add_metrics, pipeline, train, test, pos_labels_ind, classes)