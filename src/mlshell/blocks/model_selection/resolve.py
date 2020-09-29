"""
The :mod:`mlshells.model_selection.resolve` includes utils to resolve
hp before optimization.

:class:`mlshell.model_selection.Resolver` extracts dataset-based parameters.
Adapt :func:`mlshell.Resolver.resolve` to support more hp.
"""

import itertools
import operator

import matplotlib.pyplot as plt
import mlshell
import numpy as np
import sklearn

__all__ = ['Resolver']


class Resolver(object):
    """Resolve dataset-related pipeline hyper-parameter.

    Interface: resolve, th_resolver.

    For example, numeric/categorical features indices are dataset dependent.
    Resolver allows to set them before fit/optimize step.

    """
    _required_parameters = []

    def __init__(self):
        pass

    def resolve(self, hp_name, value, pipeline, dataset, **kwargs):
        """Resolve hyper-parameter value.

        Parameters
        ----------
        hp_name : str
            Hyper-parameter identifier.
        value: any objects
            Value to resolve.
        pipeline : :class:`mlshell.Pipeline`
            Pipeline contained ``hp_name`` in ``pipeline.get_params()``.
        dataset : :class:`mlshell.Dataset`
            Dataset.
        **kwargs : dict
            Additional kwargs to pass in corresponding resolver endpoint.

        Returns
        -------
        value : some object
            Resolved value. If no resolver endpoint, return value unchanged.

        Notes
        -----
        Currently supported hp_name for :class:`mlshell.pipeline.Steps` :

        ``process_parallel__pipeline_categoric__select_columns__kw_args``
            dataset.meta['categoric_ind_name'].
        ``process_parallel__pipeline_numeric__select_columns__kw_args``
            dataset.meta['numeric_ind_name'].
        ``estimate__apply_threshold__threshold``
            ``Resolver.th_resolver()``.
        ``estimate__apply_threshold__params``
            {i: dataset.meta[i]
            for i in ['pos_labels_ind', 'pos_labels', 'classes']}

        """
        flag = True
        if hp_name ==\
                'process_parallel__pipeline_categoric__select_columns__kw_args':
            categoric_ind_name = dataset.meta['categoric_ind_name']
            value = {'indices': list(categoric_ind_name.keys())}
        elif hp_name ==\
                'process_parallel__pipeline_numeric__select_columns__kw_args':
            numeric_ind_name = dataset.meta['numeric_ind_name']
            value = {'indices': list(numeric_ind_name.keys())}
        elif hp_name == 'estimate__apply_threshold__threshold':
            value = self.th_resolver(pipeline, dataset, **kwargs)
        elif hp_name == 'estimate__apply_threshold__params':
            value = {i: dataset.meta[i] for i in
                     ['pos_labels_ind', 'pos_labels', 'classes']}
        else:
            flag = False

        if flag:
            print(f"Resolve hp: {hp_name}")
        return value

    def th_resolver(self, pipeline, dataset, **kwargs):
        """Calculate threshold range.

        If necessary to optimize threshold simultaneously with other hps,
        extract optimal thresholds values from data in advance could provides
        more directed tuning, than use random values.

        * Get predict_proba:
            :class:`mlshell.model_selection.cross_val_predict` .
        * Get tpr, fpr, th_range relative to positive label:
            :func:`sklearn.metrics.roc_curve` .
        * Sampling thresholds close to optimum of predefined metric:
            :func:`mlshell.model_selection.Resolver.calc_th_range` .


        Parameters
        ----------
        pipeline : :class:`mlshell.Pipeline`
            Pipeline.
        dataset : :class:`mlshell.Dataset`
            Dataset.
        **kwargs : dict
            kwargs['cross_val_predict'] to pass in:
             :func:`sklearn.model_selection.cross_val_predict` . ``method``
             always should be set to 'predict_proba', ``y`` argument ignored.
            kwargs['calc_th_range'] to pass in:
             :func:`mlshell.model_selection.Resolver.calc_th_range` .

        Raises
        ------
        ValueError
            If kwargs key 'method' is absent or kwargs['method'] !=
            'predict_proba'.

        Returns
        -------
        th_range : :class:`numpy.ndarray`, list of :class:`numpy.ndarray`
            Thresholds array sorted ascending of shape [samples] or
            [n_outputs * samples]/[samples, n_outputs] for multi-output. In
            multi-output case each target has separate th_range of length
            ``samples``, output contains concatenate / merge or combined ranges
            depends on :func:`mlshell.model_selection.Resolver.calc_th_range`
            ``multi_output`` argument.

        """
        kwargs_cvp = kwargs.get('cross_val_predict', {})
        kwargs_ctr = kwargs.get('calc_th_range', {})
        if kwargs_cvp.get('method', None) is not 'predict_proba':
            raise ValueError("cross_val_predict 'method'"
                             " should be 'predict_proba' to resolve th_ range.")
        if 'y' in kwargs_cvp:
            del kwargs_cvp['y']

        x = dataset.x
        y = dataset.y
        classes, pos_labels, pos_labels_ind =\
            operator.itemgetter('classes',
                                'pos_labels',
                                'pos_labels_ind')(dataset.meta)
        # Extended sklearn.model_selection.cross_val_predict (TimeSplitter).
        y_pred_proba, index = mlshell.model_selection.cross_val_predict(
            pipeline.pipeline, x, y=y, **kwargs_cvp)
        # y_true!=y for TimeSplitter.
        y_true = y.values[index] if hasattr(y, 'loc') else y[index]
        # Calculate roc_curve, sample th close to tpr/(tpr+fpr) maximum.
        th_range = self.calc_th_range(y_true, y_pred_proba, pos_labels,
                                      pos_labels_ind,
                                      **kwargs_ctr)
        return th_range

    def calc_th_range(self, y_true, y_pred_proba, pos_labels, pos_labels_ind,
                      metric=None, samples=10, sampler=None, multi_output='concat',
                      plot_flag=False, roc_auc_kwargs=None):
        """Calculate threshold range from OOF ROC curve.

        Parameters
        ----------
        y_true : :class:`numpy.ndarray`
            Target(s) of shape [n_samples,] or [n_samples, n_outputs] for
            multi-output.
        y_pred_proba : :class:`numpy.ndarray`, list of :class:`numpy.ndarray`
            Probability prediction of shape [n_samples, n_classes]
            or [n_outputs, n_samples, n_classes] for multi-output.
        pos_labels: list
           List of positive labels for each target.
        pos_labels_ind: list
           List of positive labels index in :func:`numpy.unique` for each
           target.
        metric : callable, optional (default=None)
            ``metric(fpr, tpr, th_)`` should returns optimal threshold value,
            corresponding ``th_`` index and vector for metric visualization
            ofshape [n_samples,]. If None, ``tpr/(fpr+tpr)`` is used.
        samples : int, optional (default=10)
            Number of unique threshold values to sample (should be enough data).
        sampler : callable, optional (default=None)
            ``sampler(optimum, th_, samples)`` should returns: (sub-range of
            th_, original index of sub-range).If None, linear sample from
            ``[optimum/100; 2*optimum]`` with limits ``[np.min(th_), 1]``.
        multi_output: str {'merge','product','concat'}, optional (default='concat')
            For multi-output case, either merge th_range for each target or
            find all combination or concatenate ranges. See notes below.
        plot_flag : bool, optional (default=False)
            If True, plot ROC curve and resulted th range.
        roc_auc_kwargs : dict, optional (default=None)
            Additional kwargs to pass in :func:`sklearn.metrics.roc_auc_score` .
            If None, {}.

        Returns
        -------
        th_range : :class:`numpy.ndarray`, list of :class:`numpy.ndarray`
            Thresholds array sorted ascending of shape [samples] or
            [n_outputs * samples]  for multi-output.

        Notes
        -----
        For multi-output if  th_range_1 = [0.1,0.2], th_range_1 = [0.3, 0.4]:
        concat => [(0.1, 0.3), (0.2, 0.4)]
        product => [(0.1, 0.3), (0.1, 0.4), (0.2, 0.3), (0.2, 0.4)]
        merge => [(0.1, 0.1), (0.2, 0.2), (0.3, 0.3), (0.4, 0.4)]

        """
        if metric is None:
            metric = self._metric
        if sampler is None:
            sampler = self._sampler
        if roc_auc_kwargs is None:
            roc_auc_kwargs = {}
        if y_true.ndim == 1:
            # Add dimension, for compliance to multi-output.
            y_true = y_true[..., None]
            y_pred_proba = [y_pred_proba]

        # Process targets separately.
        res = []
        for i in range(y_true.shape[1]):
            # Calculate ROC curve.
            fpr, tpr, th_ = sklearn.metrics.roc_curve(
                y_true[:, i], y_pred_proba[i][:, pos_labels_ind[i]],
                pos_label=pos_labels[i], drop_intermediate=True)
            # th_ sorted descending.
            # fpr sorted ascending.
            # tpr sorted ascending.
            # Calculate metric at every point.
            best_th_, best_ind, q = metric(fpr, tpr, th_)
            # Sample near metric optimum.
            th_range, index = sampler(best_th_, th_, samples)
            # Plot.
            if plot_flag:
                self._th_plot(y_true[:, i], y_pred_proba[i], pos_labels[i],
                              pos_labels_ind[i], best_th_, q, tpr, fpr,
                              th_, th_range, index, roc_auc_kwargs)
            res.append(th_range)
        if len(res) == 1:
            return res[0]
        # multi-output
        if multi_output == 'concat':
            th_range = list(zip(*res))  # min length.
        elif multi_output == 'merge':
            merged = np.unique([j for i in res for j in i])
            th_range = [(el, el) for el in merged]
        elif multi_output == 'product':
            th_range = list(itertools.product(*res))
        else:
            raise ValueError(f"Unknown argument value:"
                             f" multi_output={multi_output}")
        return th_range

    def _metric(self, fpr, tpr, th_):
        """Find threshold value maximizing metric.

        Parameters
        ----------
        fpr : :class:`numpy.ndarray`, shape = [>2]
            Increasing false positive rates such that element i is the false
            positive rate of predictions with score >= thresholds[i].
        tpr : :class:`numpy.ndarray`, shape = [>2]
            Increasing true positive rates such that element i is the true
            positive rate of predictions with score >= thresholds[i].
        th_ : :class:`numpy.ndarray`, shape = [n_thresholds]
            Decreasing thresholds on the decision function used to compute
            fpr and tpr.`thresholds[0]` represents no instances being predicted
            and is arbitrarily set to `max(y_score) + 1`.

        Returns
        -------
        best_th_ : float
            Optimal threshold value.
        best_ind : int
            Optimal threshold index in `th_`.
        q : :class:`numpy.ndarray`
            Vectorized metric score (for all ROC curve points).

        """
        def np_divide(a, b):
            """Numpy array division.

            ignore / 0, div( [-1, 0, 1], 0 ) -> [0, 0, 0]
            """
            with np.errstate(divide='ignore', invalid='ignore'):
                c = np.true_divide(a, b)
                c[~np.isfinite(c)] = 0  # -inf inf NaN
            return c

        # q = tpr/(fpr+tpr) go through max.
        q = np_divide(tpr, fpr + tpr)
        best_ind = np.argmax(q)
        best_th_ = th_[best_ind]
        return best_th_, best_ind, q

    def _sampler(self, best_th_, th_, samples):
        """Sample threshold near optimum.

        Parameters
        ----------
        best_th_ : float
            Optimal threshold value.
        th_ : :class:`numpy.ndarray`, shape = [n_thresholds]
            Threshold range to sample fro, sorted descending.
        samples : int, optional (default=10)
            Number of samples.

        Returns
        -------
        th_range : :class:`numpy.ndarray`
            Resulted ``th_`` sub-range, sorted ascending.
        index : :class:`numpy.ndarray`
            Samples indices in ``th_``.

        Notes
        -----
        Linear sampling from [best/100; 2*best] with limits [np.min(th), 1].

        """
        # th_ descending.
        # th_range ascending.
        desired = np.linspace(max(best_th_ / 100, np.min(th_)),
                              min(best_th_ * 2, 1), samples)
        # Find index of nearest from th_reverse (a[i-1] < v <= a[i]).
        index_rev = np.searchsorted(th_[::-1], desired, side='left')  # Asc.
        # If not enough data, th could duplicate (remove).
        index_rev = np.unique(index_rev)
        index = len(th_) - index_rev - 1
        th_range = np.clip(th_[index], a_min=None, a_max=1)
        return th_range, index

    def _th_plot(self, y_true, y_pred_proba, pos_label, pos_label_ind,
                 best_th_, q, tpr, fpr, th_, th_range, index, roc_auc_kwargs):
        """Plot ROC curve and metric for current target."""
        fig, axs = plt.subplots(nrows=2, ncols=1)
        fig.set_size_inches(10, 10)
        # Roc score.
        y_type = sklearn.utils.multiclass.type_of_target(y_true)
        if y_type == "binary":
            roc_auc = sklearn.metrics.roc_auc_score(
                y_true, y_pred_proba[:, pos_label_ind], **roc_auc_kwargs)
        elif y_type == "multiclass":
            roc_auc = sklearn.metrics.roc_auc_score(
                y_true, y_pred_proba, **roc_auc_kwargs)
        else:
            assert False, f"Unhandled y_type {y_type}"
        # Roc curve.
        axs[0].plot(fpr, tpr, 'darkorange',
                    label=f"ROC curve (AUC = {roc_auc:.3f}).")
        axs[0].scatter(fpr[index], tpr[index], c='b', marker="o")
        axs[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_title(f"Receiver operating characteristic "
                         f"(label '{pos_label}')")
        axs[0].legend(loc="lower right")
        # Metric q.
        axs[1].plot(th_, q, 'green')
        axs[1].vlines(best_th_, np.min(q), np.max(q))
        axs[1].vlines(th_range, np.min(q), np.max(q), colors='b',
                      linestyles=':')
        axs[1].set_xlim([0.0, 1.0])
        axs[1].set_xlabel('Threshold')
        axs[1].set_ylabel('TPR/(TPR+FPR)')
        axs[1].set_title('Selected th values objective maximum')
        # plt.plot(th_, fpr, 'red')
        plt.show()


if __name__ == '__main__':
    pass
