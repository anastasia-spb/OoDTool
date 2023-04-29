import os
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn import preprocessing

from tool import data_types


class OoDScore:
    def __init__(self, probabilities, output_dir):
        self.probabilities = probabilities
        self.output_dir = output_dir

    def run(self):
        score = self.__calculate_ent_ood_score()
        if score is None:
            return None

        timestamp_str = datetime.now().strftime("%y%m%d_%H%M%S")
        name = "".join(('./ood_score_', timestamp_str, '.ood.pkl'))
        output_file = os.path.join(self.output_dir, name)
        with open(output_file, 'wb') as handle:
            pickle.dump(score, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return output_file

    def __calculate_ent_ood_score(self):
        def calculate_score(row):
            y_pred_columns = [col for col in row.axes[0].values if
                              col.startswith(data_types.ClassProbabilitiesType.name())]
            mat = row[y_pred_columns].values
            mat = np.array([np.array(el) for el in mat])
            mean_dist = np.mean(mat, axis=0)
            score = entropy(mean_dist)
            if not np.isfinite(score):
                print(score, mean_dist)
            return entropy(mean_dist)

        class_distributions_df = pd.DataFrame()
        for i, probability in enumerate(self.probabilities):
            df = pd.read_pickle(probability)
            if class_distributions_df.empty:
                class_distributions_df[data_types.RelativePathType.name()] = df[data_types.RelativePathType.name()]
                class_distributions_df[data_types.ClassProbabilitiesType.name()] = df[
                    data_types.ClassProbabilitiesType.name()]
            else:
                suffix = '_' + str(i)
                class_distributions_df = pd.merge(class_distributions_df, df[
                    [data_types.RelativePathType.name(), data_types.ClassProbabilitiesType.name()]],
                                                  on=data_types.RelativePathType.name(), how='inner',
                                                  suffixes=('', suffix))

        if class_distributions_df.empty:
            return None

        class_distributions_df[data_types.OoDScoreType.name()] = class_distributions_df.apply(
            lambda row: calculate_score(row), axis=1)

        class_distributions_df[data_types.OoDScoreType.name()] = \
            self.__normalize_score(class_distributions_df[data_types.OoDScoreType.name()].values)

        return class_distributions_df[[data_types.OoDScoreType.name(), data_types.RelativePathType.name()]]

    @staticmethod
    def __normalize_score(x):
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
        rescaled = min_max_scaler.fit_transform(np.reshape(x, (x.shape[0], 1)))
        return np.reshape(rescaled, (x.shape[0]))

