class CustomImputer(object):
    """Custom imputer template."""

    def __init__(self, *args, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        sklearn.impute.SimpleImputer()
        raise NotImplementedError
        # replcae np.nan на уникальное + add indicator feature
        # если просто заменить на 0, признак давать вклад в предсказание данного объекта не будет,
        # то потеряем информацию без индикатора
        # для линейных моделей - не должен быть выбросом, можно сделать регрессию и предсказать
        # для деревьев - просто отнесет в свою ветку
        # надо кастомный класс, в зависимости от типа алгоритма своё ставить


# TODO: Better move to utills y_pred_to_probe, also get pos_labels_in
def prob_to_pred(y_pred_proba, th_, pos_labels, neg_labels, pos_labels_ind):
    """Set threshold on predict_proba."""
    y_pred = np.where(y_pred_proba[..., pos_labels_ind] > th_, [pos_labels], [neg_labels])
    return y_pred


class SMWrapper(sklearn.base.BaseEstimator, sklearn.base.RegressorMixin):
    """Sklearn-style wrapper for statsmodels regressor."""

    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept
        self.model_ = None
        self.results_ = None

    def fit(self, x, y):
        if self.fit_intercept:
            x = sm.add_constant(x)
        self.model_ = self.model_class(y, x)
        self.results_ = self.model_.fit()

    def predict(self, x):
        if self.fit_intercept:
            x = sm.add_constant(x)
        return self.results_.predict(x)


if any(len(i) > 2 for i in classes):
    raise ValueError('Currently only binary classification supported.')