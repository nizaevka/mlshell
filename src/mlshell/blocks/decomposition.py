import sklearn.decomposition


class PCA(sklearn.decomposition.PCA):
    """Extended PCA.

     Skip argument added.

     """
    def __init__(self, n_components=None, *, copy=True, whiten=False,
                 svd_solver='auto', tol=0.0, iterated_power='auto',
                 random_state=None, skip=True):
        self.skip = skip
        super().__init__(
            n_components=n_components, copy=copy, whiten=whiten,
            svd_solver=svd_solver, tol=tol, iterated_power=iterated_power,
            random_state=random_state)

    def fit(self, x, y=None):
        if self.skip:
            return self
        else:
            return super().fit(x, y)

    def fit_transform(self, x, y=None):
        if self.skip:
            return x
        else:
            return super().fit_transform(x, y)

    def transform(self, x):
        if self.skip:
            return x
        else:
            return super().transform(x)

    def inverse_transform(self, x):
        if self.skip:
            return x
        else:
            return super().inverse_transform(x)
