import operator

import matplotlib.pyplot as plt
import mlshell
import mlshell.custom
import numpy as np
import sklearn


class Resolver(object):
    """Resolve hyper-parameter based on dataset.

    Interface: resolve, th_resolver

    For example, categorical features indices are dataset dependent.
    Resolve allows to set it before fit/optimize step.

    """
    _required_parameters = []

    def __init__(self):
        pass

    def resolve(self, hp_name, pipeline, dataset, **kwargs):
        """Resolve hyper-parameter value.

        Parameters
        ----------
        hp_name : str
            Hyper-parameter identifier.
        pipeline : mlshell.Pipeline
            Pipeline contain `hp_name` in get_params().
        dataset : mlshell.Dataset
            Dataset to to extract from.
        **kwargs : dict
            Additional kwargs to pass in corresponding resolver endpoint.

        Returns
        -------
        value : some object
            Resolved value. If no resolver endpoint, return 'auto'.

        Notes
        -----
        Currently supported hp_name for mlshell.PipelineSteps:
        'process_parallel__pipeline_categoric__select_columns__kwargs'
            dataset['categoric_ind_name']/extract from dataset if absent.
        'process_parallel__pipeline_numeric__select_columns__kwargs'
            dataset['numeric_ind_name']/extract from dataset if empty.
        'estimate__apply_threshold__threshold'
            HpResolver.th_resolver()
        'estimate__apply_threshold__kwargs'
            {i:dataset.meta[i]
                for i in ['pos_labels_ind', 'pos_labels', 'classes']}

        """
        if hp_name ==\
                'process_parallel__pipeline_categoric__select_columns__kwargs':
            categoric_ind_name = dataset.meta['categoric_ind_name']
            value = {'indices': list(categoric_ind_name.keys())}
        elif hp_name ==\
                'process_parallel__pipeline_numeric__select_columns__kwargs':
            numeric_ind_name = dataset.meta['numeric_ind_name']
            value = {'indices': list(numeric_ind_name.keys())}
        elif hp_name == 'estimate__apply_threshold__threshold':
            value = self.th_resolver(pipeline, dataset, **kwargs)
        elif hp_name == 'estimate__apply_threshold__kwargs':
            value = {i: dataset.meta[i]
                     for i in ['pos_labels_ind', 'pos_labels', 'classes']}
        else:
            value = 'auto'
        if value != 'auto':
            print(f"Resolve hp: {hp_name}")
        return value

    def th_resolver(self, pipeline, dataset, **kwargs):
        """Get threshold range from ROC curve on OOF probabilities predictions.

        If necessary to grid search threshold simultaneously with other hps,
        extract optimal thresholds values from data in advance could provides
        more directed tuning, than use random values.

        * Get predict_proba from `cross_val_predict`.
        * Get tpr, fpr, th_range from `roc_curve` relative to positive label.
        * Sample thresholds close to `metric` optimum.

        As alternative, use mlshell.ThresholdOptimizer to grid search threshold
        separately after others hyper-parameters tuning.

        Parameters
        ----------
        pipeline : mlshell.Pipeline
            Pipeline to resolve thresholds for.
        dataset : mlshell.Dataset
            Dataset to to extract from.
        **kwargs : dict
            Kwargs['cross_val_predict'] to pass in `sklearn.model_selection`.
            cross_val_predict. Sub-key 'method' value always should be set to
            'predict_proba', sub-key 'y' will be ignored.
            Kwargs['calc_th_range'] to pass in `mlshell.calc_th_range`.

        Raises
        ------
        ValueError
            If kwargs key 'method' absent or kwargs['method']='predict_proba'.

        Returns
        -------
        th_range : array-like
            Resulted array of thresholds values, sorted ascending.

        """
        if 'method' not in kwargs or kwargs['method'] != 'predict_proba':
            raise ValueError("cross_val_predict 'method'"
                             " should be 'predict_proba'.")
        if 'y' in kwargs:
            del kwargs['y']

        x = dataset.x
        y = dataset.y
        classes, pos_labels, pos_labels_ind =\
            operator.itemgetter('classes',
                                'pos_labels',
                                'pos_labels_ind')(dataset.meta)
        # Extended sklearn.model_selection.cross_val_predict (TimeSplitter).
        y_pred_proba, index = mlshell.custom.cross_val_predict(
            pipeline, x, y=y, **kwargs['cross_val_predict'])
        # y_true!=y for TimeSplitter.
        y_true = y.values[index] if hasattr(y, 'loc') else y[index]
        # Calculate roc_curve, sample th close to tpr/(tpr+fpr) maximum.
        th_range = self.calc_th_range(y_true, y_pred_proba, pos_labels,
                                      pos_labels_ind,
                                      **kwargs['calc_th_range'])
        return th_range

    def calc_th_range(self, y_true, y_pred_proba, pos_labels, pos_labels_ind,
                      metric=None, samples=10, sampler=None, plot_flag=False):
        """Calculate th range from OOF roc_curve.

        Parameters
        ----------
        y_true : numpy.ndarray or 2d numpy.ndarray
            Target(s) of shape [n_samples,] or [n_samples, n_outputs] for
            multi-output.
        y_pred_proba :  2d numpy.ndarray  or list of 2d numpy.ndarray
            Probability prediction of shape [n_samples, n_classes]
            or [n_outputs, n_samples, n_classes] for multi-output.
        pos_labels: list
           List of positive labels for each target.
        pos_labels_ind: list
           List of positive labels index in np.unique(target) for each
           target.
        metric : callable, None, optional (default=None)
            Will be called with roc_curve output metric(fpr, tpr, th_). Should
            return optimal threshold value, corresponding `th_` index and
            vector for metric visualization (shape as tpr). If None,
            tpr/(fpr+tpr) used.
        samples : int, optional (default=10)
            Desired number of threshold values.
        sampler : callable, optional (default=None)
            Will be called sampler(optimum, th_, samples), Should return:
            (sub-range of th_, original index of sub-range).If None, linear
            sample from [optimum/100; 2*optimum] with limits [np.min(th_), 1].
        plot_flag : bool, optional (default=False)
            If True, plot ROC curve and resulted th range.

        Returns
        -------
        th_range : numpy.ndarray or list of numpy.ndarray
            Resulted array of thresholds sorted ascending values of shape
            [samples] or [n_outputs, samples]  for multi-output.

        """
        if not metric:
            metric = self._metric
        if not sampler:
            sampler = self._sampler
        if y_true.ndim == 1:
            # Add dimension, for compliance to multi-output.
            y_true = y_true[..., None]

        # Process targets separately.
        res = []
        for i in range(len(y_true)):
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
                              th_, th_range, index)
            res.append(th_range)
        th_range = res if len(res) > 1 else res[0]
        return th_range

    def _metric(self, fpr, tpr, th_):
        """Find threshold value maximizing metric.

        Parameters
        ----------
        fpr : numpy.ndarray, shape = [>2]
            Increasing false positive rates such that element i is the false
            positive rate of predictions with score >= thresholds[i].
        tpr : numpy.ndarray, shape = [>2]
            Increasing true positive rates such that element i is the true
            positive rate of predictions with score >= thresholds[i].
        th_ : numpy.ndarray, shape = [n_thresholds]
            Decreasing thresholds on the decision function used to compute
            fpr and tpr.`thresholds[0]` represents no instances being predicted
            and is arbitrarily set to `max(y_score) + 1`.

        Returns
        -------
        best_th_ : float
            Optimal threshold value.
        best_ind : int
            Optimal threshold index in `th_`.
        q : numpy.ndarray
            Metric score for all roc_curve points.

        """
        def np_divide(a, b):
            """Numpy array division.

            ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0]
            """
            with np.errstate(divide='ignore', invalid='ignore'):
                c = np.true_divide(a, b)
                c[~np.isfinite(c)] = 0  # -inf inf NaN
            return c

        # q go through max.
        q = np_divide(tpr, fpr + tpr)  # tpr/(fpr+tpr)
        best_ind = np.argmax(q)
        best_th_ = th_[best_ind]
        # [alternative] faster go from left
        # use reverse view, need last occurrence
        # best_th_ = th_[::-1][np.argmax(q[::-1])]
        return best_th_, best_ind, q

    def _sampler(self, best_th_, th_, samples):
        """Sample threshold near optimum.

        Parameters
        ----------
        best_th_ : float
            Optimal threshold value.
        th_ : numpy.ndarray, shape = [n_thresholds]
            Threshold range to sample fro, sorted descending.
        samples : int, optional (default=10)
            Number of samples.

        Returns
        -------
        th_range : numpy.ndarray
            Resulted th_ sub-range, sorted ascending.
        index : numpy.ndarray
            Samples indices in th_.

        Notes
        -----
        Linear sampling from [best/100; 2*best] with limits [np.min(th), 1].

        """
        # th_ descending
        # th_range ascending
        desired = np.linspace(max(best_th_ / 100, np.min(th_)),
                              min(best_th_ * 2, 1), samples)
        # Find index of nearest from th_reverse (a[i-1] < v <= a[i]).
        index_rev = np.searchsorted(th_[::-1], desired, side='left')
        index = len(th_) - index_rev - 1
        th_range = np.clip(th_[index], a_min=None, a_max=1)
        return th_range, index

    def _th_plot(self, y_true, y_pred_proba, pos_label, pos_label_ind,
                 best_th_, q, tpr, fpr, th_, th_range, index):
        """Plot roc curve and metric."""
        fig, axs = plt.subplots(nrows=2, ncols=1)
        fig.set_size_inches(10, 10)
        # Roc curve.
        roc_auc = sklearn.metrics.roc_auc_score(y_true,
                                                y_pred_proba[:, pos_label_ind])
        axs[0].plot(fpr, tpr, 'darkorange',
                    label=f"ROC curve (area = {roc_auc:.3f})")
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
        axs[1].set_title('Selected th values near maximum')
        # plt.plot(th_, fpr, 'red')
        plt.show()


if __name__ == '__main__':
    pass
