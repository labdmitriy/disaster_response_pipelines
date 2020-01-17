import sys
import subprocess

import importlib

import numpy as np
from sklearn.linear_model import LogisticRegression


def install(package, import_name=None):
    if import_name is None:
        import_name = package
        
    subprocess.check_call(['pip', 'install', '-U', package])
#     subprocess.check_call(['conda', 'install', '-y', package])
#     globals()[import_name] = importlib.import_module(import_name)

class MyLogisticRegression(LogisticRegression):
    def __init__(self, penalty='l2', dual=False, tol=1e-4, C=1.0,
                 fit_intercept=True, intercept_scaling=1, class_weight=None,
                 random_state=None, solver='lbfgs', max_iter=100,
                 multi_class='auto', verbose=0, warm_start=False, n_jobs=None,
                 l1_ratio=None):
        self._single_class_label = None
        super().__init__(penalty=penalty, dual=dual, tol=tol, C=C,
                         fit_intercept=fit_intercept, intercept_scaling=intercept_scaling, class_weight=class_weight,
                         random_state=random_state, solver=solver, max_iter=max_iter,
                         multi_class=multi_class, verbose=verbose, warm_start=warm_start, n_jobs=n_jobs,
                         l1_ratio=l1_ratio)

    @staticmethod
    def _has_only_one_class(y):
        return len(np.unique(y)) == 1

    def _fitted_on_single_class(self):
        return self._single_class_label is not None

    def fit(self, X, y=None):
        if self._has_only_one_class(y):
            self._single_class_label = y[0]
            self.classes_ = np.unique(y)
        else:
            super().fit(X, y)
        return self

    def predict(self, X):
        if self._fitted_on_single_class():
            return np.full(X.shape[0], self._single_class_label)
        else:
            return super().predict(X)