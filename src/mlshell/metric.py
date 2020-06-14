import mlshell
from mlshell.libs import *


class Scorer(object):
    def __init__(self, scorer=None):
        """

        Attributes:
            scorer:
                Object for which wrapper is created.

        """
        self.scorer = scorer

    def __call__(self, *args, **kwargs):
        """Redirect call to scorer object."""
        return self.scorer(*args, **kwargs)

    def __getattr__(self, name):
        """Redirect unknown methods to scorer object."""
        def wrapper(*args, **kwargs):
            getattr(self.scorer, name)(*args, **kwargs)
        return wrapper


#TODO: [beta]
class ExtendedScorer(object):
    def __init__(self, scorer):
        # Scorer to extend.
        self.scorer = scorer

        # [deprecated] now not memorize state, if no pass_csutom step, use default
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
        try:
            if estimator.steps[0][0] == 'pass_custom':
                if estimator.steps[0][1].kwargs:
                    self.scorer._kwargs.update(estimator.steps[0][1].kwargs)
        except AttributeError:
            # ThresholdClassifier object has no attribute 'steps'.

            # [deprecated] Now use init kwargs in score,
            #   not last if no step or `pass_custom`.
            # self.scorer._kwargs.update(self.cache_custom_kwargs)
            pass

        return self.scorer(estimator, x, y, **kwargs)


class ScorerProducer(mlshell.Producer):
    def __init__(self, project_path='', logger=None):
        self.logger = logger if logger else logging.Logger(__class__.__name__)
        self.project_path = project_path
        super().__init__(self.project_path, self.logger)

    # [alternative]
    # def __init__(self, *args, **kwargs):
    #    self.logger = kwargs.get('logger', logging.Logger('Validator'))

    def make_scorer(self, scorer, func=None, kwargs=None):
        """Make scorer from metric function.

        func : callback or str.
            Custom function or sklearn built-in metric name.
        kwargs
        """
        if func is None:
            raise ValueError('Specify metric function')
        if kwargs is None:
            kwargs = {}

        if isinstance(func, str):
            # convert to callable
            # ignore kwargs (built-in `str` metrics has hard-coded kwargs)
            scorer.scorer = sklearn.metrics.get_scorer(func)
        else:
            kwargs = copy.deepcopy(kwargs)
            if 'needs_custom_kwargs' in kwargs:
                # [deprecated] Now check will not work, separate objects.
                # if self.custom_scorer:
                #     raise ValueError("Only one custom metric can be set with 'needs_custom_kwargs'.")
                del kwargs['needs_custom_kwargs']
                # create special object.
                # [alternative] Rewrite _BaseScorer.
                custom_scorer = sklearn.metrics.make_scorer(func, **kwargs)
                scorer.scorer = _ExtendedScorer(custom_scorer)._scorer_shell
            else:
                scorer.scorer = sklearn.metrics.make_scorer(func, **kwargs)

        # TODO: move out check somewhere.
        # if metric_id not in custom_metrics:
        #     # valid sklearn.metrics.SCORERS.keys().
        #     scorers[metric_id] = sklearn.metrics.get_scorer(metric_id)
        return scorer

# [deprecated] Now is the new object.
#     def resolve_scoring(self, metric_ids, custom_metrics, **kwargs):
#         """Make scorers from user_metrics.
#
#         Args:
#             metric_ids (sequence of str): user_metrics names to use in gs.
#             custom_metrics (dict): {'name': (sklearn metric object, bool greater_is_better), }
#
#         Returns:
#             scorers (dict): {'name': sklearn scorer object, }
#
#         Note:
#             if 'gs__metric_id' is None, estimator default will be used.
#
#         """
#         scorers = {}
#         self.cache_custom_kwargs = kwargs.get('pass_custom__kwargs', {})
#         self.custom_scorer = None
#
#         # [deprecated]
#         # if not names:
#         #     # need to set explicit, because always need not None 'refit' name
#         #     # can`t extract estimator built-in name, so use all from validation user_metrics
#         #     names = user_metrics.keys()
#         for metric_id in metric_ids:
#             if metric_id in custom_metrics:
#                 metric = custom_metrics[metric_id]
#                 if isinstance(metric[0], str):
#                     # convert to callable
#                     # ignore kwargs (built-in `str` metrics has hard-coded kwargs)
#                     scorers[metric_id] = sklearn.metrics.get_scorer(metric[0])
#                     continue
#                 if len(metric) == 1:
#                     kwargs = {}
#                 else:
#                     kwargs = copy.deepcopy(metric[1])
#                     if 'needs_custom_kwargs' in kwargs:
#                         if self.custom_scorer:
#                             raise ValueError("Only one custom metric can be set with 'needs_custom_kwargs'.")
#                         del kwargs['needs_custom_kwargs']
#                         self.custom_scorer = sklearn.metrics.make_scorer(metric[0], **kwargs)
#                         # [alternative] Rewrite _BaseScorer.
#                         scorers[metric_id] = self._custom_scorer_shell
#                         continue
#                 scorers[metric_id] = sklearn.metrics.make_scorer(metric[0], **kwargs)
#             else:
#                 # valid sklearn.metrics.SCORERS.keys().
#                 scorers[metric_id] = sklearn.metrics.get_scorer(metric_id)
#         return scorers

    def resolve_metric(self, metric_ids, custom_metrics):
        res = {}
        for metric_id in metric_ids:
            if metric_id in custom_metrics:
                res.update({metric_id: custom_metrics[metric_id]})
            else:
                # Sklearn built-in, resolve through scorer.
                scorer = sklearn.metrics.get_scorer(metric_id)
                metric = (
                    scorer._score_func,
                    {'greater_is_better': scorer._sign > 0,
                     'needs_proba':
                         isinstance(scorer,
                                    sklearn.metrics._scorer._ProbaScorer),
                     'needs_threshold':
                         isinstance(scorer,
                                    sklearn.metrics._scorer._ThresholdScorer),}
                )
                res.update({metric_id: metric})
        return res

    def _via_metrics(self, metrics, pipeline, train, test, **kwargs):
        # via metrics
        #   upside: only one inference
        #   donwside: errpr-prone
        #   strange: xgboost lib contain auto detection y_type logic.
        if not metrics:
            self.logger.warning("Warning: no metrics to evaluate.")
            return
        x_train, y_train = train
        x_test, y_test = test

        # [deprecated] not need anymore
        # if hasattr(pipeline, 'predict_proba'):
        #     th_ = pipeline.get_params().get('estimate__apply_threshold__threshold', 0.5)
        #     y_pred_train = self.prob_to_pred(y_pred_proba_train, th_)
        #     y_pred_test = self.prob_to_pred(y_pred_proba_test, th_)

        # Need to prevent multiple prediction.
        temp = {'predict_proba': None,
                'decision_function': None,
                'predict': None}
        for name, metric in metrics.items():
            if metric[1].get('needs_proba', False):
                if not hasattr(pipeline, 'predict_proba'):
                    self.logger.warning(f"Warning: pipeline object has no method 'predict_proba':\n"
                                        "    ignore metric '{name}'")
                    continue
                # [...,i] equal to [:,i]/[:,:,i]/.. (for multi-output target)
                if not temp['predict_proba']:
                    # Pipeline assume that there are as much classes as in train was.
                    pos_labels_ind = operator.itemgetter('pos_labels_ind')(train.get_classes())
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
                    temp['predict_proba'] = (y_pred_train, y_pred_test)
                else:
                    y_pred_train, y_pred_test = temp['predict_proba']
            elif metric[1].get('needs_threshold', False):
                if not hasattr(pipeline, 'decision_function'):
                    self.logger.warning(f"Warning: pipeline object has no method 'predict_proba':\n"
                                        "    ignore metric '{name}'")
                    continue
                if not temp['decision_function']:
                    y_pred_train = pipeline.decision_function(x_train)
                    y_pred_test = pipeline.decision_function(x_test)
                    temp['decision_function'] = (y_pred_train, y_pred_test)
                else:
                    y_pred_train, y_pred_test = temp['decision_function']
            else:
                if not temp['predict']:
                    y_pred_train = pipeline.predict(x_train)
                    y_pred_test = pipeline.predict(x_test)
                    temp['predict'] = (y_pred_train, y_pred_test)
                else:
                    y_pred_train, y_pred_test = temp['predict']

            # skip make_scorer params
            kwargs = {key: metric[1][key] for key in metric[1]
                       if key not in ['greater_is_better', 'needs_proba',
                                      'needs_threshold', 'needs_custom_kwargs']}

            # pass_custom steo inly
            if metric[1].get('needs_custom_kwargs', False):
                if (hasattr(pipeline, 'steps') and
                    pipeline.steps[0][0] == 'pass_custom' and
                    pipeline.steps[0][1].kwargs):
                    kwargs.update(pipeline.steps[0][1].kwargs)

            # result score on Train
            score_train = metric[0](y_train, y_pred_train, **kwargs)
            # result score on test
            score_test = metric[0](y_test, y_pred_test, **kwargs)

            self.logger.log(25, f"{name}:")
            self.logger.log(5, f"{name}:")
            self._score_pretty_print(metric, score_train, score_test, classes)
        return

    def _score_pretty_print(self, metric, score_train, score_test, classes):
        if metric[0].__name__ == 'confusion_matrix':
            labels = metric[1].get('labels', classes)
            score_train = tabulate.tabulate(pd.DataFrame(data=score_train,
                                                         columns=labels,
                                                         index=labels),
                                            headers='keys', tablefmt='psql').replace('\n', '\n    ')
            score_test = tabulate.tabulate(pd.DataFrame(data=score_test,
                                                        columns=labels,
                                                        index=labels),
                                           headers='keys', tablefmt='psql').replace('\n', '\n    ')
        elif metric[0].__name__ == 'classification_report':
            if isinstance(score_train, dict):
                score_train = tabulate.tabulate(pd.DataFrame(score_train),
                                                headers='keys', tablefmt='psql').replace('\n', '\n    ')
                score_test = tabulate.tabulate(pd.DataFrame(score_test),
                                               headers='keys', tablefmt='psql').replace('\n', '\n    ')
            else:
                score_train = score_train.replace('\n', '\n    ')
                score_test = score_test.replace('\n', '\n    ')
        elif isinstance(score_train, np.ndarray):
            # pretty printing numpy arrays
            score_train = np.array2string(score_train, prefix='    ')
            score_test = np.array2string(score_test, prefix='    ')
        self.logger.log(25, f"Train:\n    {score_train}\n"
                            f"Test:\n    {score_test}")
        self.logger.log(5, f"Train:\n    {score_train}\n"
                           f"Test:\n    {score_test}")
        return

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