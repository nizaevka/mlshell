"""The :mod:`mlshell.plot` includes utils to visualize results."""

__all__ = ['Plotter']


class Plotter(object):
    """Visualize results."""
    def __init__(self):
        pass

    def plot(self, pipeline, metrics, datasets, runs, validator, logger):
        """Plot results.

        Parameters
        ----------
        pipeline : :class:`mlshell.Pipeline`
            Pipeline.
        metrics : list of class:`mlshell.Metric`
            Metrics to evaluate.
        datasets : dict of class:`mlshell.Dataset`
            Datasets to evaluate on: {'dataset_id': dataset}. For classifier
            ``dataset.meta`` should contains ``pos_labels_ind`` key.
        runs: dict
            Resilts for pipeline-dataset pair:
            {'dataset_id': optimizer.update_best output}
        validator : :class:`mlshell.model_selection.Validator`
            Object to evaluate vectorized metrics: ``validator.validate(
            pipeline, metrics, datasets, logger, vector=True)`` .
        logger : :class:`logging.Logger`
            Logger.

        """
        # TODO: some logic.
        logger.warning('Plotter not implemented.')


if __name__ == '__main__':
    pass
