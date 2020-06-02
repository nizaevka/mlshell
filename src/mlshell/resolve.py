from mlshell.libs import *


class HpResolver(object):
    def __init__(self):
        pass

    def resolve(self, pipeline, dataset, name, **kwargs):
        res = 'auto'
        if name == 'process_parallel__pipeline_categoric__select_columns__kw_args':
            if 'categoric_ind_name' in dataset:
                categoric_ind_name = dataset.get('categoric_ind_name', {})
            else:
                categoric_ind_name = self._extract_ind_name(dataset)[1]
            res = {'indices': list(categoric_ind_name.keys())}
        elif name == 'process_parallel__pipeline_numeric__select_columns__kw_args':
            if 'numeric_ind_name' in dataset:
                res = {'indices': list(dataset.get('numeric_ind_name', {}).keys())}
            else:
                res = self._extract_ind_name(dataset.get('data'))[2]
        elif name == 'estimate__apply_threshold__threshold':
            sub_kwargs = kwargs.get('resolve_params', {}).get(name, {})
            res = self._th_resolver(pipeline, dataset, **sub_kwargs, **kwargs)
        elif name == 'estimate__apply_threshold__kw_args':
            res = dataset.get_classes()
        # [deprecated] built-in auto good enough
        # elif name == 'process_parallel__pipeline_categoric__encode_onehot__categories':
        #     res = [list(range(len(i[1]))) for i in dataset.get('categoric_ind_name', {}).values()]
        if res != 'auto':
            print(f"Resolve hp: {name}")
        return res

    def _extract_ind_name(self, dataset):
        """Extract categoric/numeric names and index in case if no unify applied."""
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
                # loose categories names
                categoric_ind_name[ind - count] = (column_name, np.unique(data[column_name]))
            else:
                numeric_ind_name[ind - count] = (column_name,)
        return data, categoric_ind_name, numeric_ind_name

    def _th_resolver(self, pipeline, dataset, cv=None, fit_params=None, plot_flag=False, samples=10, **kwargs):
        classes, pos_labels, pos_labels_ind = operator.itemgetter('classes',
                                                                'pos_labels',
                                                                'pos_labels_ind')(dataset.get_classes())
        x = dataset.get_x()
        y = dataset.get_y()
        if fit_params is None:
            fit_params = {}
        y_pred_proba, _, y_true = mlshell.custom.cross_val_predict(
            pipeline, x, y=y, groups=None,
            cv=cv, fit_params=fit_params,
            method='predict_proba')
        th_range = self.calc_th_range(y_true, y_pred_proba, pos_labels, pos_labels_ind, plot_flag, samples)

        return th_range

    def calc_th_range(self, y_true, y_pred_proba, pos_labels, pos_labels_ind, plot_flag=False, samples=10):
        """Calculate th range from OOF roc_curve.

        TODO: add plot
            https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py

        """
        best_th_, best_ind, q, fpr, tpr, th_range = self.brut_th_(y_true, y_pred_proba, pos_labels, pos_labels_ind)
        coarse_th_range, coarse_index = self.coarse_th_range(best_th_, th_range, samples)
        if plot_flag:
            self.th_plot(y_true, y_pred_proba, pos_labels, pos_labels_ind,
                         best_th_, q, tpr, fpr, th_range, coarse_th_range, coarse_index)
        return coarse_th_range

    def th_plot(self, y_true, y_pred_proba, pos_labels, pos_labels_ind,
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

    def coarse_th_range(self, best_th_, th_, samples):
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

    def brut_th_(self, y_true, y_pred_proba, pos_labels, pos_labels_ind):
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
        q = np_divide(tpr, fpr+tpr)  # tpr/(fpr+tpr)
        best_ind = np.argmax(q)
        best_th_ = th_[best_ind]
        # [deprecated] faster go from left
        # use reverse view, need last occurrence
        # best_th_ = th_[::-1][np.argmax(q[::-1])]
        return best_th_, best_ind, q, fpr, tpr, th_

    # TODO: Better move to utills y_pred_to_probe, also get pos_labels_ind
    # move out
    def prob_to_pred(self, y_pred_proba, th_):
        """Fix threshold on predict_proba"""
        y_pred = np.where(y_pred_proba[:, self.pos_labels_ind] > th_, [self.pos_labels], [self.neg_label])
        return y_pred

