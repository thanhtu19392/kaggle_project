'''
An attempt to make marcotcr's LIME to work with pandas dataframe and sklearn pipeline.
The implementation should work with any dataset and model pipeline
Not just for numerical matrix as it is the case of the orginal implementation

https://github.com/marcotcr/lime/blob/master/lime/lime_tabular.py
'''

from lime.lime_base import LimeBase
from lime.explanation import Explanation
from lime.lime_tabular import TableDomainMapper
from lime.discretize import DecileDiscretizer, EntropyDiscretizer, BaseDiscretizer

from sklearn.preprocessing import StandardScaler, Imputer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import pairwise_distances
import numpy as np
import pandas as pd


class QuartileDiscretizer(BaseDiscretizer):

    def __init__(self, data, categorical_features, feature_names, labels=None):

        BaseDiscretizer.__init__(self, data, categorical_features,
                                 feature_names, labels=labels)

    def bins(self, data, labels):
        bins = []
        for feature in self.to_discretize:
            x = data[:, feature].astype(float)
            x = x[~np.isnan(x)]
            qts = np.percentile(x, [25, 50, 75])
            bins.append(qts)
        return bins

    def undiscretize(self, data):
        ret = data.copy()
        for feature in self.means:
            mins = self.mins[feature]
            maxs = self.maxs[feature]
            means = self.means[feature]
            stds = self.stds[feature]

            def get_inverse(q):
                if np.isnan(q):
                    return q
                return max(mins[q],
                           min(np.random.normal(means[q], stds[q]), maxs[q]))
            if len(data.shape) == 1:
                q = int(ret[feature])
                ret[feature] = get_inverse(q)
            else:
                ret[:, feature] = (
                    [get_inverse(int(x)) for x in ret[:, feature]])
        return ret


class PredictorLime:

    def __init__(self, Xtrain, Ytrain, numeric_features, categoric_features, class_names,
                 kernel_width=None, feature_selection='auto', discretizer_type='quartile'):

        self.feature_names = Xtrain.columns.tolist()
        self.discretizer = self._discretizer(discretizer_type, Xtrain, Ytrain, categoric_features, self.feature_names)
        if self.discretizer:
            discretized_Xtrain = pd.DataFrame(self.discretizer.discretize(Xtrain.values), columns=Xtrain.columns)
            self.categoric_features = self.feature_names
            self.numeric_features = []
            self.feature_values = self._compute_freqs(discretized_Xtrain, self.feature_names)
        else:
            self.scaler = make_pipeline(Imputer(), StandardScaler(with_mean=False))
            self.scaler.fit(Xtrain[numeric_features])
            self.feature_values = self._compute_freqs(Xtrain, categoric_features)

        if kernel_width is None:
            kernel_width = np.sqrt(Xtrain.shape[1]) * .75
        kernel_width = float(kernel_width)

        def kernel(d):
            return np.sqrt(np.exp(-(d ** 2) / kernel_width ** 2))

        self.base = LimeBase(kernel)
        self.class_names = class_names
        self.feature_selection = feature_selection

    def _discretizer(self, discretizer_type, Xtrain, Ytrain, categoric_features, feature_names):
        all_types = {
            'quartile': QuartileDiscretizer,
            'decile': DecileDiscretizer,
            'entropy': EntropyDiscretizer
        }

        if discretizer_type in all_types:
            return all_types[discretizer_type](
                Xtrain.values, [i for i, c in enumerate(feature_names) if c in categoric_features], feature_names, Ytrain
            )
        return None

    def _compute_freqs(self, Xtrain, categoric_features):
        '''
        Get by column, a tuple of values and frequences
        Ex: {'Sex': (['Male', 'Female', np.nan], [0.3, 0.4, 0.3])}
        '''

        feature_values = {}
        for col in categoric_features:
            vc = Xtrain[col].value_counts()
            vc = (vc / Xtrain.shape[0])
            values = vc.index.tolist() + [np.nan]
            freqs = vc.values.tolist() + [max(0, 1 - vc.sum())]  # computation error
            feature_values[col] = (values, freqs)
        return feature_values

    def _sample_numeric(self, data_row, num_samples):
        if self.discretizer:
            return {}, {}
        scaler = self.scaler.steps[1][1]
        scale_, mean_ = scaler.scale_, scaler.mean_
        scaled_samples, samples = {}, {}
        for idx, column in enumerate(self.numeric_features):
            scaled = np.random.normal(size=num_samples)
            data = scaled * scale_[idx] + mean_[idx]
            data[0] = data_row[column]
            scaled[0] = (data[0] - mean_[idx])/scale_[idx] if not np.isnan(data[0]) else 0
            scaled_samples[column] = scaled
            samples[column] = data
        return scaled_samples, samples

    def _sample_categoric(self, data_row, num_samples):
        samples, binary_samples = {}, {}
        for column in self.categoric_features:
            values, freqs = self.feature_values[column]
            sample = np.random.choice(values, size=num_samples, replace=True, p=freqs)
            sample[0] = data_row[column]
            binary = (sample == data_row[column])
            samples[column] = sample
            binary_samples[column] = binary
        return binary_samples, samples

    def _concat(self, dictA, dictB):
        dictB.update(dictA)
        return dictB

    def _local_sampling(self, data_row, num_samples):
        '''
        Generates a neighborhood around a prediction.

        For numerical features, perturb them by sampling from a Normal(0,1) and
        doing the inverse operation of mean-centering and scaling, according to
        the means and stds in the training data. For categorical features,
        perturb by sampling according to the training distribution, and making
        a binary feature that is 1 when the value is the same as the instance
        being explained.
        '''

        if self.discretizer:
            data_row = self.discretizer.discretize(data_row)
        num_scaled, num_data = self._sample_numeric(data_row, num_samples)
        cat_binary, cat_data = self._sample_categoric(data_row, num_samples)

        data = pd.DataFrame(self._concat(cat_data, num_data))[self.feature_names]
        binary = pd.DataFrame(self._concat(cat_binary, num_scaled))[self.feature_names]
        if self.discretizer:
            data = pd.DataFrame(self.discretizer.undiscretize(data.values), columns=data.columns)
        return binary, data

    def _domain_mapper(self, data_row, scaled_data_row):
        n_cols = len(data_row)
        feature_names = list(self.feature_names)
        if feature_names is None:
            feature_names = [str(x) for x in range(n_cols)]

        values = [0] * n_cols
        for feature in self.categoric_features:
            i = self.feature_names.index(feature)
            if self.discretizer is not None and i in self.discretizer.lambdas:
                continue
            name = data_row[i]
            feature_names[i] = '%s=%s' % (feature_names[i], name)
            values[i] = 'True'

        discretized_feature_names = None
        if self.discretizer is not None:
            discretized_instance = self.discretizer.discretize(data_row)
            discretized_feature_names = list(feature_names)
            for f in self.discretizer.names:
                discretized_feature_names[f] = self.discretizer.names[f][int(
                    discretized_instance[f])]

        return TableDomainMapper(
            self.feature_names,
            [str(v) for v in data_row],
            scaled_data_row,
            categorical_features=range(len(self.categoric_features)),
            discretized_feature_names=discretized_feature_names
        )

    def explain_instance(self, data_row, classifier_fn, labels=(1,), n_features=10, n_samples=5000,
                         distance_metric='euclidean', model_regressor=None):

        data_row = data_row[self.feature_names]
        scaled_data, data = self._local_sampling(data_row, n_samples)

        distances = pairwise_distances(
            scaled_data,
            scaled_data.iloc[0].reshape(1, -1),
            metric=distance_metric
        ).ravel()

        yss = classifier_fn(data)

        ret_exp = Explanation(self._domain_mapper(data_row, scaled_data.iloc[0]), class_names=self.class_names)
        ret_exp.predict_proba = yss[0]
        for label in labels:
            (ret_exp.intercept[label],
             ret_exp.local_exp[label],
             ret_exp.score) = self.base.explain_instance_with_data(
                scaled_data.values, yss, distances, label, n_features,
                model_regressor=model_regressor,
                feature_selection=self.feature_selection)
        return ret_exp
