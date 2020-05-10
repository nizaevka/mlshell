from mlshell.libs import *


class HpResolver(object):
    def __init__(self):
        pass

    def resolve(self, pipeline, data, name, kwargs):
        # TODO: self not needed
        res = 'auto'
        if name == 'process_parallel__pipeline_categoric__select_columns__kw_args':
            res = {'indices': list(data.get('categoric_ind_name', {}).keys())}
        elif name == 'process_parallel__pipeline_numeric__select_columns__kw_args':
            res = {'indices': list(data.get('numeric_ind_name', {}).keys())}
        elif name == 'estimate__apply_threshold__threshold' or name == 'threshold':
            res = self._th_resolver(pipeline, data, kwargs)
        # [deprecated] built-in auto good enough
        # elif name == 'process_parallel__pipeline_categoric__encode_onehot__categories':
        #     res = [list(range(len(i[1]))) for i in data.get('categoric_ind_name', {}).values()]
        if res != 'auto':
            print(f"Resolve hp: {name}")
        return res

    def _th_resolver(self, pipeline, data, kwargs):
        y_pred_proba, _, y_true = self.cross_val_predict(
            pipeline, x, y=y, groups=None,
            cv=self.cv(), fit_params=fit_params,
            method='predict_proba')
        th_range = self.calc_th_range(y, y_pred_proba)

        return th_range

    def calc_th_range(self, y_true, y_pred_proba, th_range=None, th__plot_flag=False):
        """ Сalculate th range from OOF roc_curve.

        Used in th__strategy (1)(2)(3.2).

        Args:
            th_range (array-like, optional(default=None)): if None, will be calculated from roc_curve.

        Returns:
            th_range
            predict_proba
            y_true (np.ndarray): true labels for self.y_train.

        Note:
            For classification task it is possible to tune classification threshold ``th_`` on CV.
            For details see `Concepts <./Concepts.html#classification-threshold>`__.
            Mlshell support multiple strategy for ``th_`` tuning.

                (0) Don't use ``th_`` (common case).

                    * Not all classificator provide predict_proba (SVM).
                    * We can use f1, logloss.
                    * | If necessary you can dynamically pass params in custom scorer function to tune them in CV,
                      | use 'pass_custom__kw_args' step in hp_grid.

                (1) First GS best hps with CV, then GS best ``th_`` (common case).

                    * For GS hps by default used auc-roc as score.
                    * For GS ``th_`` main score.

                (2) Use additional step in pipeline (metaestimator) to GS ``th_`` in predefined range (experimental).

                    * Tune ``th_`` on a par with other hps.
                    * ``th_`` range should be unknown in advance:

                        (2.1) set in arbitrary in hp_grid

                        (2.2) take typical values from ROC curve OOF

                (3) While GS best hps with CV, select best ``th_`` for each fold in separately (experimental).

                    * For current hp combination maximize tpr/(tpr+fpr) on each fold by ``th_``.
                    * | Although there will different best ``th_`` on folds,
                      | the generalizing ability of classifier might be better.
                    * Then select single overall best ``th_`` on GS with main score.


        TODO:
            | add support TimeSeriesSplit
            | add plot
            |    https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html#sphx-glr-auto-examples-model-selection-plot-roc-crossval-py
            | strategy 4: GS set here

        """
        if th_range is None:
            best_th_, best_ind, q, fpr, tpr, th_range = self.brut_th_(y_true, y_pred_proba)
            coarse_th_range, coarse_index = self.coarse_th_range(best_th_, th_range)
            if th__plot_flag:
                self.th_plot(y_true, y_pred_proba, best_th_, q, tpr, fpr, th_range, coarse_th_range, coarse_index)
        else:
            coarse_th_range = th_range
        return coarse_th_range

    def th_plot(self, y_true, y_pred_proba, best_th_, q, tpr, fpr, th_, coarse_th_, coarse_index):
        # TODO: built_in roc curve plotter
        #    https: // scikit - learn.org / stable / modules / generated / sklearn.metrics.RocCurveDisplay.html  # sklearn.metrics.RocCurveDisplay
        fig, axs = plt.subplots(nrows=2, ncols=1)
        fig.set_size_inches(10, 10)
        # roc_curve
        roc_auc = sklearn.metrics.roc_auc_score(y_true, y_pred_proba[:, self.pos_label_ind])
        axs[0].plot(fpr, tpr, 'darkorange', label=f"ROC curve (area = {roc_auc:.3f})")
        axs[0].scatter(fpr[coarse_index], tpr[coarse_index], c='b', marker="o")
        axs[0].plot([0, 1], [0, 1], color='navy', linestyle='--')
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate')
        axs[0].set_title(f"Receiver operating characteristic (label '{self.pos_label}')")
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

    def cross_val_predict(self, *args, **kwargs):
        """Function to make bind OOF prediction/predict_proba.

        Args:
            args
            kwargs

        Returns:
            folds_predict_proba (2d np.ndarray): OOF probability predictions [n_test_samples x n_classes].
            folds_test_index (1d np.ndarray): test indices for OOF subset (reseted, not raw).
            y_true (1d np.ndarray): test for OOF subset (for Kfold whole dataset).

        TODO:
            in some fold could be not all classes, need to check.
        """
        # dev check for custom OOF
        debug = False
        estimator = args[0]
        x = args[1]
        y = kwargs['y']
        cv = kwargs['cv']
        temp_pp = None
        temp_ind = None
        try:
            folds_predict_proba = sklearn.model_selection.cross_val_predict(*args, **kwargs)
            folds_test_index = np.arange(0, folds_predict_proba.shape[0])
            if debug:
                temp_pp = folds_predict_proba
                temp_ind = folds_test_index
                raise ValueError('debug')
        except ValueError as e:
            # custom OOF
            # for TimeSplitter no prediction at first fold
            # self.logger.warning('Warning: {}'.format(e))
            folds_predict_proba = []  # list(range(self.cv_n_splits))
            folds_test_index = []  # list(range(self.cv_n_splits))
            # th_ = [[2, 1. / self.n_classes] for i in self.classes_]  # init list for th_ for every class
            ind = 0
            for fold_train_index, fold_test_index in cv.split(x):
                # stackingestimator__sample_weight=train_weights[fold_train_subindex]
                if hasattr(x, 'loc'):
                    estimator.fit(x.loc[x.index[fold_train_index]],
                                  y.loc[y.index[fold_train_index]],
                                  **self.p['pipeline__fit_params'])
                    # in order of pipeline.classes_
                    fold_predict_proba = estimator.predict_proba(x.loc[x.index[fold_test_index]])
                else:
                    estimator.fit(x[fold_train_index], y[fold_train_index], **self.p['pipeline__fit_params'])
                    # in order of pipeline.classes_
                    fold_predict_proba = estimator.predict_proba(x[fold_test_index])
                # merge th_ for class
                # metrics.roc_curve(y[fold_test_index], y_test_prob, pos_label=self.pos_label)
                # th_[self.pos_label].extend(fold_th_)
                folds_test_index.extend(fold_test_index)
                folds_predict_proba.extend(fold_predict_proba)
                ind += 1
            folds_predict_proba = np.array(folds_predict_proba)
            folds_test_index = np.array(folds_test_index)
            # delete duplicates
            # for i in range(self.n_classes):
            #    th_[i] = sorted(list(set(th_[i])), reverse=True)
        if debug:
            assert np.array_equal(temp_pp, folds_predict_proba)
            assert np.array_equal(temp_ind, folds_test_index)

        y_true = y.values[folds_test_index] if hasattr(y, 'loc') else y[folds_test_index]
        return folds_predict_proba, folds_test_index, y_true

    def coarse_th_range(self, best_th_, th_):
        """Get most possible th range.

        Note:
            linear sample from [best/100; 2*best] with limits [np.min(th), 1]
            th descending
            th_range ascending
        """
        th_range_desire = np.linspace(max(best_th_ / 100, np.min(th_)), min(best_th_ * 2, 1), self.p['th__samples'])
        # find index of nearest from th_reverse
        index_rev = np.searchsorted(th_[::-1], th_range_desire, side='left')  # a[i-1] < v <= a[i]
        index = len(th_) - index_rev - 1
        th_range = np.clip(th_[index], a_min=None, a_max=1)
        return th_range, index

    def brut_th_(self, y_true, y_pred_proba):
        """ Measure th value that maximize tpr/(fpr+tpr).

        Note:
            for th gs will be used values near best th.

        TODO:
            it is possible to bruforce based on self.metric
            early-stopping if q decrease

        """
        fpr, tpr, th_ = sklearn.metrics.roc_curve(
            y_true, y_pred_proba[:, self.pos_label_ind],
            pos_label=self.pos_label, drop_intermediate=True)
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

    def prob_to_pred(self, y_pred_proba, th_):
        """Fix threshold on predict_proba"""
        y_pred = np.where(y_pred_proba[:, self.pos_label_ind] > th_, [self.pos_label], [self.neg_label])
        return y_pred
