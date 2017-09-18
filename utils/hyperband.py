from sklearn.cross_validation import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from random import random
from math import log, ceil, floor
from sklearn.model_selection import ParameterSampler
from math import sqrt
from sklearn.metrics import roc_auc_score as AUC, log_loss
from sklearn.metrics import mean_squared_error as MSE, mean_absolute_error as MAE


class Hyperband(BaseEstimator, TransformerMixin):

    def __init__(self, classifier, feat_space, task):

        self.classifier = classifier
        self.feat_space = feat_space
        self.task = task
        self.max_iter = 81        # maximum iterations per configuration
        self.eta = 3            # defines configuration downsampling rate (default = 3)
        self.s_max = floor(log(self.max_iter)/log(self.eta))
        self.B = (self.s_max + 1) * self.max_iter
        self.best_model = None
        self.best_loss = np.inf

    def fit(self, X, y=None):

        data_dic = dict()
        data_dic['x_train'], data_dic['x_test'], data_dic['y_train'], data_dic['y_test'] = \
            train_test_split(X, y, test_size=0.3, random_state=0)

        self.run(data=data_dic)
        return self

    def predict_proba(self, X):
        return self.best_model.predict_proba(X)

    def predict(self, X):
        return self.best_model.predict(X)

    def run(self, data, skip_last=1, dry_run=False):

        for s in reversed(range(self.s_max + 1)):

            # initial number of configurations
            n = int(ceil(self.B / self.max_iter / (s+1) * self.eta ** s))

            # initial number of iterations per config
            r = self.max_iter * self.eta ** (-s)

            # n random configurations
            configs = list(ParameterSampler(self.feat_space, n_iter=n, random_state=None))
            T = [handle_integers(config) for config in configs]

            for i in range((s+1) - int(skip_last)):    # changed from s + 1

                # Run each of the n configs for <iterations>
                # and keep best (n_configs / eta) configurations

                n_configs = n * self.eta ** (-i)
                n_iterations = r * self.eta ** (i)

                val_losses = []
                early_stops = []

                for t in T:

                    if dry_run:
                        result = {'loss': random(), 'log_loss': random(), 'auc': random()}
                    else:
                        result = try_params(n_iterations, self.classifier, t, data, task=self.task)

                    assert(type(result) == dict)
                    assert('loss' in result)

                    loss = result['loss']
                    val_losses.append(loss)
                    model = result['model']

                    early_stop = result.get('early_stop', False)
                    early_stops.append(early_stop)

                    if loss < self.best_loss:
                        self.best_loss = loss
                        self.best_model = model

                # select a number of best configurations for the next loops
                indices = np.argsort(val_losses)
                T = [T[i] for i in indices if not early_stops[i]]
                T = T[0:int(n_configs / self.eta)]


def handle_integers(params):

    new_params = {}
    for k, v in params.items():
        if type(v) == float and int(v) == v:
            new_params[k] = int(v)
        else:
            new_params[k] = v

    return new_params


def train_and_eval_sklearn_classifier(clf, data):

    x_train = data['x_train']
    y_train = data['y_train']

    x_test = data['x_test']
    y_test = data['y_test']

    clf.fit(x_train, y_train)

    try:
        p = clf.predict_proba(x_test)[:, 1]    # sklearn convention
    except IndexError:
        p = clf.predict_proba(x_test)

    ll = log_loss(y_test, p)
    auc = AUC(y_test, p)

    return {'loss': ll, 'log_loss': ll, 'auc': auc, 'model': clf}


def train_and_eval_sklearn_regressor(reg, data):

    x_train = data['x_train']
    y_train = data['y_train']

    x_test = data['x_test']
    y_test = data['y_test']

    reg.fit(x_train, y_train)

    p = reg.predict(x_test)

    mse = MSE(y_test, p)
    rmse = sqrt(mse)
    mae = MAE(y_test, p)

    return {'loss': rmse, 'rmse': rmse, 'mae': mae, 'model': reg}


def try_params(n_iterations, classifier, params, data, task='classification'):
    trees_per_iteration = 5
    n_estimators = int(round(n_iterations * trees_per_iteration))
    classifier.set_params(n_estimators=n_estimators, nthread=-1, **params)

    if task == 'classification':
        return train_and_eval_sklearn_classifier(classifier, data)
    else:
        return train_and_eval_sklearn_regressor(classifier, data)
