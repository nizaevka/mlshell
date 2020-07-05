"""The :mod:`mlshell.blocks.plot` includes class to visualize results."""

__all__ = ['Plotter']


class Plotter(object):
    """Visualize results (template)."""
    def __init__(self):
        pass

    def plot(self, pipeline, metrics, datasets, runs, validator, logger):
        """Plot results.

        Parameters
        ----------
        pipeline : mlshell.Pipeline
            Pipeline.
        metrics : list of mlshell.Metric
            Metrics to evaluate.
        datasets : dict of mlshell.Dataset
            Dataset to evaluate on. For classification 'dataset.meta' should
            contains `pos_labels_ind` key.
        runs: dict
            {'subset_id': optimizer.update_best output fot pipeline-data pair}
        validator : mlshell.model_selection.Validator
            Object to evaluate vectorized metrics.
            validator.validate(pipeline, metrics, datasets, logger, vector=True)
        logger : logger object
            Logs.

        """
        # TODO: some logic.
        logger.info('Plotter OK.')


        pass


if __name__ == '__main__':
    pass
