# SOURCE : https://www.kaggle.com/naveenasaithambi/optimizedrounder-improved

import pandas as pd
import scipy as sp
import numpy as np
from functools import partial
from sklearn.metrics import cohen_kappa_score


class OptimizedRounder_v2(object):
    def __init__(self, num_class):
        self.coef_ = 0
        self.num_class = num_class

    def _kappa_loss(self, coef, X, y):
        preds = pd.cut(X, [-np.inf] + list(np.sort(coef)) + [np.inf], labels=list(range(self.num_class)))
        kappa = cohen_kappa_score(y, preds, weights='quadratic')
        return -kappa

    def fit(self, X, y):
        loss_partial = partial(self._kappa_loss, X=X, y=y)
        initial_coef = [i + 1 / 2 for i in range(0, self.num_class-1)]
        self.coef_ = sp.optimize.minimize(loss_partial, initial_coef,
                                          method='Nelder-Mead')

    def predict(self, X):
        preds = pd.cut(X, [-np.inf] + list(np.sort(self.coef_['x'])) + [np.inf], labels=list(range(self.num_class)))
        return preds

    def coefficients(self):
        return self.coef_['x']


# opt=OptimizedRounder_v2(5)
# x = np.array([0.2, 0.4, 0.55, 0.49, 1.2, 1.1, 1.9, 2.3, 3.3, 4.3])
# y = np.array([0, 1, 0, 1, 2, 2, 2, 3, 3, 4])
#
# opt.fit(x, y)
# preds = pd.cut(x, [-np.inf] + list(np.sort([0.3, 1.5, 2.5, 3.5])) + [np.inf], labels=[0, 1, 2, 3, 4])
# kappa = cohen_kappa_score(y, preds, weights='quadratic')
# print(preds)
# print(kappa)
# print(opt.coefficients())
# print(opt.predict(x))