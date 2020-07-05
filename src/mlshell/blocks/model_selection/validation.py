"""
The :mod:`mlshell.blocks.model_selection.validation` module includes classes
and functions to validate the model.

`Validator` class to process metrics evaluation on fitted pipeline.

'cross_val_predict' extended sklearn.model_selection.cross_val_predict
that supports partial splitters (TimeSplitter for example).

"""


import numpy as np
import sklearn

__all__ = ['Validator', 'cross_val_predict', 'partial_cross_val_predict']


class Validator(object):
    """Validate fitted pipeline."""
    def __init__(self):
        pass

    def validate(self, pipeline, metrics, datasets, logger, method='metric'):
        """Evaluate metrics on pipeline.

        Parameters
        ----------
        pipeline : mlshell.Pipeline
            Fitted model.
        metrics : list of mlshell.Metric
            Metrics to evaluate.
        datasets : list of mlshell.Dataset ('meta'[pos_labels_ind] for c)
            Dataset to evaluate on. For classification 'dataset.meta'
            should contains `pos_labels_ind` key.
        method : 'metric' or 'scorer'
            If 'metric', efficient (reuse y_pred) evaluation via
            `score_func(y, y_pred, **kwargs)`. If 'scorer', evaluate via
            `scorer(pipeline, x, y)`.
        logger : logger object
            Logs.

        """
        # pipeline
        # [kwargs, steps, pipeline.predict(x)/predict_proba(x)/
        #  decision_function(x)]
        # metric
        # [oid, scorer(pipeline, x,y), score_func(y, y_pred, **kwargs),
        #  pprint(score) ]
        # dataset
        # [oid, x, y, meta[pos_labels_ind]]
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
                x = dataset.x
                y = dataset.y
                try:
                    if method == 'metric':
                        score = self._via_metric(pipeline.pipeline, x, y,
                                                 metric, dataset,
                                                 infer[dataset.oid])
                    elif method == 'scorer':
                        score = metric.scorer(pipeline.pipeline, x, y)
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


def cross_val_predict(*args, **kwargs):
    """Extended sklearn.model_selection.cross_val_predict.

    Add TimeSplitter support (when no first fold prediction).

    Parameters
    ----------
    *args : list
        Passed to sklearn.model_selection.cross_val_predict.
    **kwargs : dict
        Passed to sklearn.model_selection.cross_val_predict.

    Returns
    -------
    y_pred_oof : 2d numpy.ndarray or list of 2d numpy.ndarray
        OOF probability predictions of shape [n_test_samples, n_classes]
        or [n_outputs, n_test_samples, n_classes] for multi-output.
    index_oof : numpy.ndarray or list of numpy.ndarray
        Samples reset indices where predictions available of shape
        [n_test_samples,] or [n_test_samples, n_outputs] for multi-output.

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
            if hasattr(x_, 'loc'):
                estimator.fit(x_.loc[x_.index[fold_train_index]],
                              y_.loc[y_.index[fold_train_index]],
                              **fit_params)
                # In order of pipeline.classes_.
                fold_y_pred = estimator.predict_proba(
                    x_.loc[x_.index[fold_test_index]])
            else:
                estimator.fit(x_[fold_train_index], y_[fold_train_index],
                              **fit_params)
                # In order of pipeline.classes_.
                fold_y_pred = estimator.predict_proba(x_[fold_test_index])
            index_oof_.extend(fold_test_index)
            y_pred_oof_.extend(fold_y_pred)
            ind += 1
        y_pred_oof_ = np.array(y_pred_oof_)
        index_oof_ = np.array(index_oof_)
        return y_pred_oof_, index_oof_

    # Process targets separately.
    y_pred_oof = []
    index_oof = []
    for i in range(len(y)):
        l, r = single_output(x, y[:, i])
        y_pred_oof.append(l)
        index_oof.append(r)
    index_oof = np.array(index_oof).T
    return y_pred_oof, index_oof


if __name__ == '__main__':
    pass
