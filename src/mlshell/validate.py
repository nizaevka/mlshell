from mlshell.libs import *


class Validator(object):
    def __init__(self, *args, **kwargs):
        self.logger = kwargs.get('logger', logging.Logger('Validator'))
        self.custom_scorer = None
        self.cache_custom_kw_args = {}

    def resolve_scoring(self, metric_ids, custom_metrics, **kwargs):
        """Make scorers from user_metrics.

        Args:
            metric_ids (sequence of str): user_metrics names to use in gs.
            custom_metrics (dict): {'name': (sklearn metric object, bool greater_is_better), }

        Returns:
            scorers (dict): {'name': sklearn scorer object, }

        Note:
            if 'gs__metric_id' is None, estimator default will be used.

        """
        scorers = {}
        self.cache_custom_kw_args = kwargs.get('pass_custom__kw_args', {})
        self.custom_scorer = None

        # [deprecated]
        # if not names:
        #     # need to set explicit, because always need not None 'refit' name
        #     # can`t extract estimator built-in name, so use all from validation user_metrics
        #     names = user_metrics.keys()
        for metric_id in metric_ids:
            if metric_id in custom_metrics:
                metric = custom_metrics[metric_id]
                if isinstance(metric[0], str):
                    # convert to callable
                    # ignore kw_args (built-in `str` metrics has hard-coded kwargs)
                    scorers[metric_id] = sklearn.metrics.get_scorer(metric[0])
                    continue
                if len(metric) == 1:
                    kw_args = {}
                else:
                    kw_args = copy.deepcopy(metric[1])
                    if 'needs_custom_kw_args' in kw_args:
                        if self.custom_scorer:
                            raise ValueError("Only one custom metric can be set with 'needs_custom_kw_args'.")
                        del kw_args['needs_custom_kw_args']
                        self.custom_scorer = sklearn.metrics.make_scorer(metric[0], **kw_args)
                        scorers[metric_id] = self._custom_scorer_shell
                        continue
                scorers[metric_id] = sklearn.metrics.make_scorer(metric[0], **kw_args)
            else:
                # valid sklearn.metrics.SCORERS.keys().
                scorers[metric_id] = sklearn.metrics.get_scorer(metric_id)
        return scorers

    def _custom_scorer_shell(self, estimator, x, y):
        """Read custom_kw_args from current pipeline, pass to scorer.

        Note: in gs self object copy, we can dynamically get param only from estimator.
        """
        try:
            if estimator.steps[0][0] == 'pass_custom':
                if estimator.steps[0][1].kw_args:
                    self.custom_scorer._kwargs.update(estimator.steps[0][1].kw_args)
        except AttributeError:
            # ThresholdClassifier object has no attribute 'steps'
            self.custom_scorer._kwargs.update(self.cache_custom_kw_args)

        return self.custom_scorer(estimator, x, y)

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
                                   sklearn.metrics._scorer._ThresholdScorer),})
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

        # prevent multiple prediction
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
            kw_args = {key: metric[1][key] for key in metric[1]
                       if key not in ['greater_is_better', 'needs_proba',
                                      'needs_threshold', 'needs_custom_kw_args']}

            # pass_custom steo inly
            if metric[1].get('needs_custom_kw_args', False):
                if (hasattr(pipeline, 'steps') and
                    pipeline.steps[0][0] == 'pass_custom' and
                    pipeline.steps[0][1].kw_args):
                    kw_args.update(pipeline.steps[0][1].kw_args)

            # result score on Train
            score_train = metric[0](y_train, y_pred_train, **kw_args)
            # result score on test
            score_test = metric[0](y_test, y_pred_test, **kw_args)

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

    # [deprecated] need rearange
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