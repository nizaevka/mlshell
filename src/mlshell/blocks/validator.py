"""
The :mod:`mlshell.blocks.validator` contains examples of `Validator` class to
process metrics evaluation on fitted pipeline.
"""


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


if __name__ == '__main__':
    pass
