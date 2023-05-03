import os
import pickle
from typing import List
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.stats import entropy
from sklearn import preprocessing

from tool.core import data_types
from tool.core.utils import data_helpers


class OoDScore:
    def __init__(self):
        self.ood_df = pd.DataFrame()

    def run(self, probabilities_files: List[str], output_dir):
        probabilities_df = data_helpers.merge_data_files(probabilities_files)
        probabilities_columns = \
            data_helpers.get_columns_which_start_with(probabilities_df, data_types.ClassProbabilitiesType.name())

        def calculate_entropy(features_row):
            mat = np.stack(features_row)
            mean_dist = np.mean(mat, axis=0)
            return entropy(mean_dist)

        probabilities_mat = probabilities_df[probabilities_columns].values
        score = np.apply_along_axis(calculate_entropy, axis=1, arr=probabilities_mat)

        self.ood_df[data_types.OoDScoreType.name()] = self.__normalize_score(score)
        self.ood_df[data_types.RelativePathType.name()] = probabilities_df[data_types.RelativePathType.name()]

        return self.__store(output_dir)

    def get_ood_df(self):
        return self.ood_df

    def __store(self, output_dir) -> str:
        timestamp_str = datetime.utcnow().strftime("%y%m%d_%H%M%S.%f")[:-3]
        name = "".join(('./ood_score_', timestamp_str, '.ood.pkl'))
        output_file = os.path.join(output_dir, name)
        with open(output_file, 'wb') as handle:
            pickle.dump(self.ood_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return output_file

    @staticmethod
    def __normalize_score(x):
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
        rescaled = min_max_scaler.fit_transform(np.reshape(x, (x.shape[0], 1)))
        return np.reshape(rescaled, (x.shape[0]))
