import operator

import matplotlib.pyplot as plt
import mlshell
import mlshell.custom
import numpy as np
import sklearn


class HpResolver(object):
    """Include methods to resolve dataset-dependent hyperparameters.

    Interface: resolve, th_resolver

    Parameters
    ----------
    project_path: str
        Absolute path to current project dir.
    logger : logger object
        Logs.

    """
    _required_parameters = ['project_path', 'logger']

    def __init__(self, project_path, logger):
        self.logger = logger
        self.project_path = project_path

    def resolve(self, hp_name, dataset, pipeline, **kwargs):
        """Resolve hyperparameter value.

        Parameters
        ----------
        hp_name : str
            Hyperparameter identifier.
        dataset : mlshell.Dataset interface object
            Dataset to to extract from.
        pipeline : object with sklearn.pipeline.Pipeline interface
            Pipeline contain `hp_name` in get_params().
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
            dataset.get_classes()

        """
        if hp_name == 'process_parallel__pipeline_categoric__select_columns__kwargs':
            if 'categoric_ind_name' in dataset:
                categoric_ind_name = dataset['categoric_ind_name']
            else:
                categoric_ind_name = self._extract_ind_name(dataset)[1]
            value = {'indices': list(categoric_ind_name.keys())}
        elif hp_name == 'process_parallel__pipeline_numeric__select_columns__kwargs':
            if 'numeric_ind_name' in dataset:
                value = {'indices': list(dataset['numeric_ind_name'].keys())}
            else:
                value = self._extract_ind_name(dataset.get('data'))[2]
        elif hp_name == 'estimate__apply_threshold__threshold':
            value = self.th_resolver(pipeline, dataset, **kwargs)
        elif hp_name == 'estimate__apply_threshold__kwargs':
            value = dataset.get_classes()
        else:
            value = 'auto'
        if value != 'auto':
            self.logger.info(f"Resolve hp: {hp_name}")
        return value

    def _extract_ind_name(self, dataset):
        """Extract categoric/numeric names and index."""
        data = dataset.get('data')
        raw_names = dataset.get('raw_names')
        categoric_ind_name = {}
        numeric_ind_name = {}
        count = 0
        for ind, column_name in enumerate(data):
            if column_name in raw_names['targets']:
                count += 1
                continue
            if column_name in raw_names['categor_features']:
                # Loose categories names.
                categoric_ind_name[ind - count] = (column_name,
                                                   np.unique(data[column_name]))
            else:
                numeric_ind_name[ind - count] = (column_name,)
        return data, categoric_ind_name, numeric_ind_name

    def th_resolver(self, dataset, pipeline, plot_flag=False, samples=10,
                    **kwargs):
        """Get threshold range from ROC curve on OOF probabilities predictions.

        If necessary to grid search threshold simultaneously with other hps,
        extract optimal thresholds values from data in advance could provides
        more directed tuning, than use random values.
            * get predict_proba from `cross_val_predict`
            * get tpr, fpr from `roc_curve`
            * sample thresholds close to tpr/(fpr+tpr) maximum.

        As alternative, use mlshell.ThresholdOptimizer to grid search threshold
        separately after others hyper-parameters tuning.

        Parameters
        ----------
        dataset : mlshell.Dataset interface object
            Dataset to to extract from.
        pipeline : object with sklearn.pipeline.Pipeline interface, supported
            `predict_proba` method
            Pipeline.
        plot_flag : bool, optional (default=False)
            If True, plot ROC curve and resulted th range.
        samples : int, optional (default=10)
            Desired length of th range.
        **kwargs : dict
            Kwargs to pass in sklearn.model_selection.cross_val_predict.
            Kwarg 'method' always should be set to 'predict_proba','y' ignored.

        Raises
        ------
        ValueError
            If kwargs key 'method' absent or kwargs['method']='predict_proba'.

        Returns
        -------
        th_range : array-like
            Resulted array of thresholds values.

        """
        if 'method' not in kwargs or kwargs['method'] != 'predict_proba':
            raise ValueError("cross_val_predict 'method'"
                             " should be 'predict_proba'.")
        if 'y' in kwargs:
            del kwargs['y']

        x = dataset.get_x()
        y = dataset.get_y()
        classes, pos_labels, pos_labels_ind =\
            operator.itemgetter('classes',
                                'pos_labels',
                                'pos_labels_ind')(dataset.get_classes())
        # Extended sklearn.model_selection.cross_val_predict (TimeSplitter).
        y_pred_proba, _, y_true = mlshell.custom.cross_val_predict(
            pipeline, x, y=y, **kwargs)
        # Calculate roc_curve, sample th close to tpr/(tpr+fpr) maximum.
        th_range = self.calc_th_range(y_true, y_pred_proba, pos_labels,
                                       pos_labels_ind, plot_flag, samples)
        return th_range

    def calc_th_range(self, y_true, y_pred_proba, pos_labels, pos_labels_ind,
                       plot_flag=False, samples=10):
        """Calculate th range from OOF roc_curve.

        Parameters
        ----------
        y_true
        y_pred_proba
        pos_labels
        pos_labels_ind
        plot_flag
        samples

        Returns
        -------

        """

        best_th_, best_ind, q, fpr, tpr, th_range =\
            self._brut_th_(y_true, y_pred_proba, pos_labels, pos_labels_ind)
        coarse_th_range, coarse_index =\
            self._coarse_th_range(best_th_, th_range, samples)
        if plot_flag:
            self._th_plot(y_true, y_pred_proba, pos_labels, pos_labels_ind,
                         best_th_, q, tpr, fpr, th_range, coarse_th_range,
                         coarse_index)
        return coarse_th_range

    def _brut_th_(self, y_true, y_pred_proba, pos_labels, pos_labels_ind):
        """ Measure th value that maximize tpr/(fpr+tpr).

        Note:
            for th gs will be used values near best th.

        TODO:
            It is possible to bruforce based on self.metric,
            early-stopping if q decrease.

        """
        fpr, tpr, th_ = sklearn.metrics.roc_curve(
            y_true, y_pred_proba[:, pos_labels_ind],
            pos_labels=pos_labels, drop_intermediate=True)
        # th_ sorted descending
        # fpr sorted ascending
        # tpr sorted ascending
        # q go through max

        def np_divide(a, b):
            """ ignore / 0, div0( [-1, 0, 1], 0 ) -> [0, 0, 0] """
            with np.errstate(divide='ignore', invalid='ignore'):
                c = np.true_divide(a, b)
                c[~np.isfinite(c)] = 0  # -inf inf NaN
            return c
        q = np_divide(tpr, fpr+tpr)  # tpr/(fpr+tpr)
        best_ind = np.argmax(q)
        best_th_ = th_[best_ind]
        # [deprecated] faster go from left
        # use reverse view, need last occurrence
        # best_th_ = th_[::-1][np.argmax(q[::-1])]
        return best_th_, best_ind, q, fpr, tpr, th_

    def _coarse_th_range(self, best_th_, th_, samples):
        """Get most possible th range.

        Note:
            linear sample from [best/100; 2*best] with limits [np.min(th), 1]
            th descending
            th_range ascending
        """
        th_range_desire = np.linspace(max(best_th_ / 100, np.min(th_)), min(best_th_ * 2, 1), samples)
        # find index of nearest from th_reverse
        index_rev = np.searchsorted(th_[::-1], th_range_desire, side='left')  # a[i-1] < v <= a[i]
        index = len(th_) - index_rev - 1
        th_range = np.clip(th_[index], a_min=None, a_max=1)
        return th_range, index

    def _th_plot(self, y_true, y_pred_proba, pos_labels, pos_labels_ind,
                 best_th_, q, tpr, fpr, th_, coarse_th_, coarse_index):
        """

        TODO: built_in roc curve plotter
            https: // scikit - learn.org / stable / modules / generated / sklearn.metrics.RocCurveDisplay.html  # sklearn.metrics.RocCurveDisplay

        """
        fig, axs = plt.subplots(nrows=2, ncols=1)
        fig.set_size_inches(10, 10)
        # roc_curve
        roc_auc = sklearn.metrics.roc_auc_score(y_true, y_pred_proba[:, pos_labels_ind])
        axs[0].plot(fpr, tpr, 'darkorange', label=f"ROC curve (area = {roc_auc:.3f})")
        axs[0].scatter(fpr[coarse_index], tpr[coarse_index], c='b', marker="o")
        axs[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_title(f"Receiver operating characteristic (label '{pos_labels}')")
        axs[0].legend(loc="lower right")
        # tpr/(tpr+fpr)
        axs[1].plot(th_, q, 'green')
        axs[1].vlines(best_th_, np.min(q), np.max(q))
        axs[1].vlines(coarse_th_, np.min(q), np.max(q), colors='b', linestyles=':')
        axs[1].set_xlim([0.0, 1.0])
        axs[1].set_xlabel('Threshold')
        axs[1].set_ylabel('TPR/(TPR+FPR)')
        axs[1].set_title('Selected th values near maximum')
        # plt.plot(th_, fpr, 'red')
        plt.show()





    # TODO: Better move to utills y_pred_to_probe, also get pos_labels_ind
    # move out
    def prob_to_pred(self, y_pred_proba, th_):
        """Fix threshold on predict_proba"""
        y_pred = np.where(y_pred_proba[:, self.pos_labels_ind] > th_, [self.pos_labels], [self.neg_label])
        return y_pred

