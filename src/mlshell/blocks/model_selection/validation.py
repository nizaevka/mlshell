"""
The :mod:`mlshells.model_selection.validation` includes model validation
utils.

:class:`mlshell.model_selection.Validator` processes metrics evaluation on
fitted pipeline.

:func:`mlshell.model_selection.cross_val_predict` extends
:func:`sklearn.model_selection.cross_val_predict`
with partial splitters (TimeSplitter for example).

"""


import numpy as np
import sklearn

__all__ = ['Validator', 'cross_val_predict', 'partial_cross_val_predict']


class Validator(object):
    """Validate fitted pipeline."""
    def __init__(self):
        pass

    def validate(self, pipeline, metrics, datasets, logger, method='metric',
                 vector=False):
        """Evaluate metrics on pipeline.

        Parameters
        ----------
        pipeline : :class:`mlshell.Pipeline`
            Fitted model.
        metrics : list of :class:`mlshell.Metric`
            Metrics to evaluate.
        datasets : list of :class:`mlshell.Dataset`
            Dataset to evaluate on. For classification ``dataset.meta``
            should contains ``pos_labels_ind`` key.
        method : 'metric', 'scorer' or 'vector'
            If 'metric', efficient evaluation (reuse y_pred) via
            ``score_func(y, y_pred, **kwargs)``. If 'scorer', evaluate via
            ``scorer(pipeline, x, y)``. If 'vector', evaluate vectorized score
            via ``score_func_vector(y, y_pred, **kwargs)``.
        vector: bool
            If True and ``method='metric'``, ``score_func_vector`` used instead
            of ``score_func`` to evaluate vectorized score (if available).
            Ignored for ``method='scorer'``.
        logger : :class:`logging.Logger`
            Logger.

        Returns
        -------
        scores : dict
            Resulted scores {'dataset_id':{'metric_id': score}}.

        """
        if not metrics:
            logger.warning("Warning: no metrics to evaluate.")
            return
        if method not in ['metric', 'scorer']:
            raise ValueError("Unknown 'method' value.")

        scores = {}
        # Storage to prevent multiple inference (via metric).
        infer = {}
        for dataset in datasets:
            infer[dataset.oid] = {
                'predict_proba': None,
                'decision_function': None,
                'predict': None
            }
            scores[dataset.oid] = {}
        for metric in metrics:
            # Level 5 needs for tests.
            logger.log(25, f"{metric.oid}:")
            for dataset in datasets:
                x = dataset.x
                y = dataset.y
                try:
                    if method == 'metric':
                        score = self._via_metric(pipeline.pipeline, x, y,
                                                 metric, dataset,
                                                 infer[dataset.oid], vector)
                    elif method == 'scorer':
                        score = metric.scorer(pipeline.pipeline, x, y)
                    else:
                        assert False
                except AttributeError as e:
                    # Pipeline has not 'predict_proba'/'decision_function'.
                    logger.warning(f"Ignore metric: {e}")
                    break
                scores[dataset.oid][metric.oid] = score
                score_ = metric.pprint(score[-1] if vector else score)
                logger.log(5, f"    {score_}")
                logger.log(25, f"    {dataset.oid}:\n"
                               f"    {score_}")
        return scores

    def _via_metric(self, pipeline, x, y, metric, dataset, infer, vector):
        """Evaluate score via score functions.

        Re-utilize inference, more efficient than via scorers.

        """
        y_pred = self._get_y_pred(pipeline, x, metric, infer, dataset)
        # Update metric kwargs with pass_custom kwarg from pipeline.
        if getattr(metric, 'needs_custom_kw_args', False):
            if hasattr(pipeline, 'steps'):
                for step in pipeline.steps:
                    if step[0] == 'pass_custom':
                        temp = step[1].kw_args.get(metric.oid, {})
                        metric.kw_args.update(temp)

        # Score.
        score = metric.score_func(y, y_pred, **metric.kw_args)
        if vector:
            score_vec = metric.score_func_vector(y, y_pred, **metric.kw_args) \
                if metric.score_func_vector is not None else [score]
            if score_vec[-1] == score:
                raise ValueError(
                    f"Found inconsistent between score and score_vector[-1]:\n"
                    f"    {metric.oid} {pipeline.oid} {dataset.oid}")
            score = score_vec
        return score

    def _get_y_pred(self, pipeline, x, metric, infer, dataset):
        if getattr(metric, 'needs_proba', False):
            # [...,i] equal to [:,i]/[:,:,i]/.. (for multi-output target)
            if infer['predict_proba'] is None:
                # Pipeline predict_proba shape would be based on train
                # (pos_labels_ind/classes not guaranteed in test).
                pos_labels_ind = dataset.meta['pos_labels_ind']
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
            if infer['decision_function'] is None:
                y_pred = pipeline.decision_function(x)
                infer['decision_function'] = y_pred
            else:
                y_pred = infer['decision_function']
        else:
            if infer['predict'] is None:
                y_pred = pipeline.predict(x)
                infer['predict'] = y_pred
            else:
                y_pred = infer['predict']
        return y_pred


def cross_val_predict(*args, **kwargs):
    """Extended :func:`sklearn.model_selection.cross_val_predict`.

    TimeSplitter support added (first fold prediction absent).

    Parameters
    ----------
    *args : list
        Passed to :func:`sklearn.model_selection.cross_val_predict` .
    **kwargs : dict
        Passed to :func:`sklearn.model_selection.cross_val_predict` .

    Returns
    -------
    y_pred_oof : :class:`numpy.ndarray`, list of :class:`numpy.ndarray`
        If method=predict_proba: OOF probability predictions of shape
        [n_test_samples, n_classes] or [n_outputs, n_test_samples, n_classes]
        for multi-output. If method=predict OOF predict of shape [n_test_samples]
        or [n_test_samples, n_outputs].
    index_oof : :class:`numpy.ndarray`, list of :class:`numpy.ndarray`
        Samples reset indices where predictions available of shape
        [n_test_samples,].

    """
    # Debug key (compare predictions for no TimeSplitter cv strategy).
    _debug = kwargs.pop('_debug', False)
    temp_pp = None
    temp_ind = None
    x = args[1]
    try:
        y_pred_oof = sklearn.model_selection.cross_val_predict(
            *args, **kwargs)
        index_oof = np.arange(0, x.shape[0])
        # [deprecated] y could be None in common case
        # y_index_oof = np.arange(0, y_pred_oof.shape[0])
        if _debug:
            temp_pp = y_pred_oof
            temp_ind = index_oof
            raise ValueError('debug')
    except ValueError:
        y_pred_oof, index_oof = partial_cross_val_predict(*args, **kwargs)
    if _debug:
        assert np.array_equal(temp_pp, y_pred_oof)
        assert np.array_equal(temp_ind, index_oof)
    return y_pred_oof, index_oof


def partial_cross_val_predict(estimator, x, y, cv, fit_params=None,
                              method='predict_proba', **kwargs):
    """Extension to cross_val_predict for TimeSplitter."""
    if fit_params is None:
        fit_params = {}
    if method is not 'predict_proba':
        raise ValueError("Currently only 'predict_proba' method supported.")
    if y.ndim == 1:
        # Add dimension, for compliance to multi-output.
        y = y[..., None]

    def single_output(x_, y_):
        """Single output target."""
        y_pred_oof_ = []
        index_oof_ = []
        ind = 0
        for fold_train_index, fold_test_index in cv.split(x_):
            # Not necessary both x,y of the same type.
            if hasattr(x_, 'loc'):
                fold_train_x_ = x_.loc[x_.index[fold_train_index]]
                fold_test_x_ = x_.loc[x_.index[fold_test_index]]
            else:
                fold_train_x_ = x_[fold_train_index]
                fold_test_x_ = x_[fold_test_index]
            if hasattr(y_, 'loc'):
                fold_train_y_ = y_.loc[y_.index[fold_train_index]]
                fold_test_y_ = y_.loc[y_.index[fold_test_index]]
            else:
                fold_train_y_ = y_[fold_train_index]
                fold_test_y_ = y_[fold_test_index]
            estimator.fit(fold_train_x_, fold_train_y_, **fit_params)
            # In order of pipeline.classes_.
            fold_y_pred = estimator.predict_proba(fold_test_x_)
            index_oof_.extend(fold_test_index)
            y_pred_oof_.extend(fold_y_pred)
            ind += 1
        y_pred_oof_ = np.array(y_pred_oof_)
        index_oof_ = np.array(index_oof_)
        return y_pred_oof_, index_oof_

    # Process targets separately.
    y_pred_oof = []
    index_oof = []
    for i in range(y.shape[1]):
        l, r = single_output(x, y[:, i])
        y_pred_oof.append(l)
        index_oof.append(r)
    # In single output should 1d.
    y_pred_oof = y_pred_oof[0] if len(y_pred_oof) == 1 else y_pred_oof
    # Index is the same for all target (later apply y[index_oof] for
    # single-/multi-output).
    index_oof = index_oof[0]
    index_oof = np.array(index_oof).T
    return y_pred_oof, index_oof


if __name__ == '__main__':
    pass
