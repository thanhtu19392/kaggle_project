from sklearn.base import BaseEstimator, TransformerMixin
from sklearn import clone
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler


class ColumnsSelector(BaseEstimator, TransformerMixin):

    def __init__(self, columns):
        assert isinstance(columns, list)
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.columns]


class UniqueCountColumnSelector(BaseEstimator, TransformerMixin):
    """
    To select those columns whose unique-count values are between
    lowerbound (inclusive) and upperbound (exclusive)
    """

    def __init__(self, lowerbound, upperbound):
        self.lowerbound = lowerbound
        self.upperbound = upperbound

    def fit(self, X, y=None):
        counts = X.apply(lambda vect: vect.unique().shape[0])
        self.columns = counts.index[counts.between(self.lowerbound, self.upperbound + 1)]
        return self

    def transform(self, X):
        return X[self.columns]


class ColumnApplier(BaseEstimator, TransformerMixin):
    """
    Some sklearn transformers can apply only on ONE column at a time
    Wrap them with ColumnApplier to apply on all the dataset
    """

    def __init__(self, underlying):
        self.underlying = underlying

    def fit(self, X, y=None):
        m = {}
        X = pd.DataFrame(X)  # TODO: :( reimplement in pure numpy?
        for c in X.columns:
            k = clone(self.underlying)
            k.fit(X[c])
            m[c] = k
        self._column_stages = m
        return self

    def transform(self, X):
        ret = {}
        X = pd.DataFrame(X)
        for c, k in self._column_stages.items():
            ret[c] = k.transform(X[c])
        return pd.DataFrame(ret)[X.columns]  # keep the same order


class TolerantLabelEncoder(LabelEncoder):
    """
    LabelEncoder is not tolerant to unseen values
    """

    def transform(self, y):
        return np.searchsorted(self.classes_, y)


class OrdinalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode the categorical value by natural number based on alphabetical order
    N/A are encoded to -2
    rare values to -1
    Very similar to TolerentLabelEncoder
    TODO: improve the implementation
    """

    def __init__(self, min_support):
        self.min_support = min_support
        self.vc = {}

    def _mapping(self, vc):
        mapping = {}
        for i, v in enumerate(vc[vc >= self.min_support].index):
            mapping[v] = i
        for v in vc.index[vc < self.min_support]:
            mapping[v] = -1
        mapping['nan'] = -2
        return mapping

    def _transform_column(self, x):
        x = x.astype(str)
        vc = self.vc[x.name]

        mapping = self._mapping(vc)

        output = pd.DataFrame()
        output[x.name] = x.map(lambda a: mapping[a] if a in mapping.keys() else -3)
        output.index = x.index
        return output.astype(int)

    def fit(self, x, y=None):
        x = x.astype(str)
        self.vc = dict((c, x[c].value_counts()) for c in x.columns)
        return self

    def transform(self, df):
        dfs = [self._transform_column(df[c]) for c in df.columns]
        out = pd.DataFrame(index=df.index)
        for df in dfs:
            out = out.join(df)
        return out.as_matrix()


class CountFrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encode the value by their frequency observed in the training set
    """

    def __init__(self, min_card=5, count_na=False):
        self.min_card = min_card
        self.count_na = count_na
        self.vc = None

    def fit(self, x, y=None):
        x = pd.Series(x)
        vc = x.value_counts()
        self.others_count = vc[vc < self.min_card].sum()
        self.vc = vc[vc >= self.min_card].to_dict()
        self.num_na = x.isnull().sum()
        return self

    def transform(self, x):
        vc = self.vc
        output = x.map(lambda a: vc.get(a, self.others_count))
        if self.count_na:
            output = output.fillna(self.num_na)
        return output.as_matrix()


class BoxCoxTransformer(BaseEstimator, TransformerMixin):
    """
    Boxcox transformation for numerical columns
    To make them more Gaussian-like
    """

    def __init__(self):
        self.scaler = MinMaxScaler()
        self.shift = 0.0001

    def fit(self, x, y=None):
        x = x.values.reshape(-1, 1)
        x = self.scaler.fit_transform(x) + self.shift
        self.boxcox_lmbda = stats.boxcox(x)[1]
        return self

    def transform(self, x):
        x = x.values.reshape(-1, 1)
        scaled = np.maximum(self.shift, self.scaler.transform(x) + self.shift)
        ret = stats.boxcox(scaled, self.boxcox_lmbda)
        return ret[:, 0]


class Logify(BaseEstimator, TransformerMixin):
    """
    Log transformation
    """

    def __init__(self):
        self.shift = 2

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return np.log10(x - x.min() + self.shift)


class YToLog(BaseEstimator, TransformerMixin):
    """
    Transforming Y to log before fitting
    and transforming back the prediction to real values before return
    """

    def __init__(self, delegate, shift=0):
        self.delegate = delegate
        self.shift = shift

    def fit(self, X, y):
        logy = np.log(y + self.shift)
        self.delegate.fit(X, logy)
        return self

    def predict(self, X):
        pred = self.delegate.predict(X)
        return np.exp(pred) - self.shift


class FillNaN(BaseEstimator, TransformerMixin):
    def __init__(self, replace):
        self.replace = replace

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        return x.fillna(self.replace)
