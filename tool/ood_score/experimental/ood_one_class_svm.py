from scipy.special import softmax

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn import preprocessing
from sklearn import svm
from sklearn.model_selection import train_test_split

from tool import data_types
from sklearn.neighbors import LocalOutlierFactor


class OneClassOoD(object):
    def __init__(self, embeddings_pkl: str):
        self.clfs = [svm.OneClassSVM(nu=0.1, kernel="rbf", gamma='scale'),
                     svm.OneClassSVM(nu=0.1, kernel="rbf", gamma='auto'),
                     svm.OneClassSVM(nu=0.1, kernel="poly", gamma='scale')]
        data_df = pd.read_pickle(embeddings_pkl)
        X = data_df[data_types.EmbeddingsType.name()].tolist()
        self.X = np.array(X, dtype=np.dtype('float64'))
        self.X_train, _ = train_test_split(X, test_size=0.6, random_state=42)
        self.scores = np.zeros(shape=(self.X.shape[0], len(self.clfs)))
        self.result = {data_types.RelativePathType.name(): data_df[data_types.RelativePathType.name()].tolist(),
                       data_types.OoDScoreType.name(): []}

    def __call__(self):
        def calculate_score(row):
            return entropy(row)

        for clf, idx in zip(self.clfs, range(len(self.clfs))):
            clf.fit(self.X_train)
            score = clf.score_samples(self.X)
            self.scores[:, idx] = softmax(score)
        ood = np.apply_along_axis(calculate_score, axis=1, arr=self.scores)
        normalized_factor = preprocessing.MinMaxScaler().fit_transform(ood.reshape(-1, 1))
        normalized_factor = normalized_factor.flatten()
        normalized_factor = 1.0 - normalized_factor
        self.result[data_types.OoDScoreType.name()] = normalized_factor.tolist()
        return pd.DataFrame.from_dict(self.result)


class LocalOutlierOoD(object):
    def __init__(self, embeddings_pkl: str):
        self.clf = LocalOutlierFactor(n_neighbors=2)
        data_df = pd.read_pickle(embeddings_pkl)
        X = data_df[data_types.EmbeddingsType.name()].tolist()
        self.X = np.array(X, dtype=np.dtype('float64'))
        self.X_train, _ = train_test_split(X, test_size=0.6, random_state=42)
        self.scores = np.zeros(shape=(self.X.shape[0]))
        self.result = {data_types.RelativePathType.name(): data_df[data_types.RelativePathType.name()].tolist(),
                       data_types.OoDScoreType.name(): []}

    def __call__(self):
        self.clf.fit(self.X)
        normalized_factor = preprocessing.MinMaxScaler().fit_transform(self.clf.negative_outlier_factor_.reshape(-1, 1))
        normalized_factor = normalized_factor.flatten()
        normalized_factor = 1.0 - normalized_factor
        self.result[data_types.OoDScoreType.name()] = normalized_factor.tolist()
        return pd.DataFrame.from_dict(self.result)
