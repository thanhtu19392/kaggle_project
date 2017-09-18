import numpy as np
from xgboost import XGBClassifier, XGBRegressor, DMatrix
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.pipeline import make_pipeline, make_union
from sklearn.preprocessing import StandardScaler, Imputer
from scipy.stats.distributions import uniform, randint
from hyperband import Hyperband
from pipeline import ColumnsSelector, FillNaN, ColumnApplier, TolerantLabelEncoder


class XGBClassifierWithEarlyStopping(XGBClassifier):
    def __init__(self, max_depth=3, learning_rate=0.1, silent=True, objective='binary:logistic', nthread=-1, gamma=0,
                 min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0, missing=None,
                 early_stopping_rounds=10, eval_metric='auc', valid_size=0.3, cv=None, cv_averaging=False):
        super(XGBClassifierWithEarlyStopping, self).__init__(max_depth=max_depth,
                                                             learning_rate=learning_rate,
                                                             n_estimators=100000,
                                                             silent=silent, objective=objective,
                                                             nthread=nthread, gamma=gamma,
                                                             min_child_weight=min_child_weight,
                                                             max_delta_step=max_delta_step,
                                                             subsample=subsample,
                                                             colsample_bytree=colsample_bytree,
                                                             colsample_bylevel=colsample_bylevel,
                                                             reg_alpha=reg_alpha,
                                                             reg_lambda=reg_lambda,
                                                             scale_pos_weight=scale_pos_weight,
                                                             base_score=base_score,
                                                             seed=seed,
                                                             missing=missing)
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.valid_size = valid_size
        self.cv = cv
        self.cv_averaging = cv_averaging
        self.models = []

    def fit(self, X, y=None, verbose=False):
        if self.cv is None:
            x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=self.valid_size,
                                                                  random_state=self.seed)
            super(XGBClassifierWithEarlyStopping, self).fit(x_train, y_train,
                                                            early_stopping_rounds=self.early_stopping_rounds,
                                                            eval_metric=self.eval_metric,
                                                            eval_set=[(x_valid, y_valid)],
                                                            verbose=verbose)
            self.n_estimators = self.best_ntree_limit
            super(XGBClassifierWithEarlyStopping, self).fit(X, y, verbose=verbose)
            self.models = [self.booster()]
        else:
            skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
            best = []
            for train_index, valid_index in skf.split(X, y):
                x_train, y_train = X[train_index], y[train_index]
                x_valid, y_valid = X[valid_index], y[valid_index]
                super(XGBClassifierWithEarlyStopping, self).fit(x_train, y_train,
                                                                early_stopping_rounds=self.early_stopping_rounds,
                                                                eval_metric=self.eval_metric,
                                                                eval_set=[(x_valid, y_valid)],
                                                                verbose=verbose)
                if self.cv_averaging:
                    self.models.append(self.booster())
                best.append(self.best_ntree_limit)
            self.n_estimators = int(np.max(best))
            if not self.cv_averaging:
                super(XGBClassifierWithEarlyStopping, self).fit(X, y, verbose=verbose)
                self.models = [self.booster()]

        return self

    def _compute_class_probs(self, test_dmatrix, output_margin=False, ntree_limit=0):
        class_probs = []
        for model in self.models:
            class_probs.append(model.predict(test_dmatrix,
                                             output_margin=output_margin,
                                             ntree_limit=ntree_limit))
        return np.array(class_probs).mean(axis=0)

    def predict_proba(self, data, output_margin=False, ntree_limit=0):
        test_dmatrix = DMatrix(data, missing=self.missing)
        class_probs = self._compute_class_probs(test_dmatrix, output_margin, ntree_limit)

        if self.objective == "multi:softprob":
            return class_probs
        else:
            classone_probs = class_probs
            classzero_probs = 1.0 - classone_probs
            return np.vstack((classzero_probs, classone_probs)).transpose()

    def predict(self, data, output_margin=False, ntree_limit=0):
        test_dmatrix = DMatrix(data, missing=self.missing)
        class_probs = self._compute_class_probs(test_dmatrix, output_margin, ntree_limit)

        if len(class_probs.shape) > 1:
            column_indexes = np.argmax(class_probs, axis=1)
        else:
            column_indexes = np.repeat(0, class_probs.shape[0])
            column_indexes[class_probs > 0.5] = 1
        return self._le.inverse_transform(column_indexes)


class XGBRegressorWithEarlyStopping(XGBRegressor):
    def __init__(self, max_depth=3, learning_rate=0.1, silent=True, objective='reg:linear',
                 nthread=-1, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1,
                 colsample_bylevel=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, seed=0,
                 missing=None, early_stopping_rounds=10, eval_metric='auc', valid_size=0.3, cv=None,
                 cv_averaging=False):
        super(XGBRegressorWithEarlyStopping, self).__init__(max_depth=max_depth,
                                                            learning_rate=learning_rate,
                                                            n_estimators=100000,
                                                            silent=silent, objective=objective,
                                                            nthread=nthread, gamma=gamma,
                                                            min_child_weight=min_child_weight,
                                                            max_delta_step=max_delta_step,
                                                            subsample=subsample,
                                                            colsample_bytree=colsample_bytree,
                                                            colsample_bylevel=colsample_bylevel,
                                                            reg_alpha=reg_alpha,
                                                            reg_lambda=reg_lambda,
                                                            scale_pos_weight=scale_pos_weight,
                                                            base_score=base_score,
                                                            seed=seed,
                                                            missing=missing)
        self.early_stopping_rounds = early_stopping_rounds
        self.eval_metric = eval_metric
        self.valid_size = valid_size
        self.cv = cv
        self.cv_averaging = cv_averaging
        self.models = []

    def fit(self, X, y=None, verbose=False):
        if self.cv is None:
            x_train, x_valid, y_train, y_valid = train_test_split(X, y, test_size=self.valid_size,
                                                                  random_state=self.seed)
            super(XGBRegressorWithEarlyStopping, self).fit(x_train, y_train,
                                                           early_stopping_rounds=self.early_stopping_rounds,
                                                           eval_metric=self.eval_metric,
                                                           eval_set=[(x_valid, y_valid)],
                                                           verbose=verbose)
            self.n_estimators = self.best_ntree_limit
            super(XGBRegressorWithEarlyStopping, self).fit(X, y, verbose=verbose)
            self.models = [self.booster()]
        else:
            skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
            best = []
            for train_index, valid_index in skf.split(X, y):
                x_train, y_train = X[train_index], y[train_index]
                x_valid, y_valid = X[valid_index], y[valid_index]
                super(XGBRegressorWithEarlyStopping, self).fit(x_train, y_train,
                                                               early_stopping_rounds=self.early_stopping_rounds,
                                                               eval_metric=self.eval_metric,
                                                               eval_set=[(x_valid, y_valid)],
                                                               verbose=verbose)
                if self.cv_averaging:
                    self.models.append(self.booster())
                best.append(self.best_ntree_limit)
            self.n_estimators = int(np.max(best))
            if not self.cv_averaging:
                super(XGBRegressorWithEarlyStopping, self).fit(X, y, verbose=verbose)
                self.models = [self.booster()]

        return self

    def _compute_preds(self, test_dmatrix, output_margin, ntree_limit):
        preds = []
        for model in self.models:
            preds.append(model.predict(test_dmatrix,
                                       output_margin=output_margin,
                                       ntree_limit=ntree_limit))
        return np.array(preds).mean(axis=0)

    def predict(self, data, output_margin=False, ntree_limit=0):
        test_dmatrix = DMatrix(data, missing=self.missing)
        preds = self._compute_preds(test_dmatrix, output_margin, ntree_limit)

        return preds


def xgboost_hyperband_classifier(numeric_features, categoric_features, learning_rate=0.08):
    return _xgboost_hyperband_model('classification', numeric_features, categoric_features, learning_rate)


def xgboost_hyperband_regressor(numeric_features, categoric_features, learning_rate=0.08):
    return _xgboost_hyperband_model('regression', numeric_features, categoric_features, learning_rate)


def _xgboost_hyperband_model(task, numeric_features, categoric_features, learning_rate):
    param_space = {
        'max_depth': randint(2, 11),
        'min_child_weight': randint(1, 11),
        'subsample': uniform(0.5, 0.5),
        'colsample_bytree': uniform(0.5, 0.5),
        'colsample_bylevel': uniform(0.5, 0.5),
        'gamma': uniform(0, 1),
        'reg_alpha': uniform(0, 1),
        'reg_lambda': uniform(0, 10),
        'base_score': uniform(0.1, 0.9),
        'scale_pos_weight': uniform(0.1, 9.9)
    }

    model = XGBClassifier(learning_rate=learning_rate) \
        if task == 'classification' else XGBRegressor(learning_rate=learning_rate)

    return make_pipeline(
        make_union(
            make_pipeline(
                ColumnsSelector(categoric_features),
                FillNaN('nan'),
                ColumnApplier(TolerantLabelEncoder())
            ),
            make_pipeline(
                ColumnsSelector(numeric_features),
                Imputer(strategy='mean'),
                StandardScaler()
            )
        ),
        Hyperband(
            model,
            feat_space=param_space,
            task=task
        )
    )
