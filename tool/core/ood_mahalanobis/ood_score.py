import os
import pickle
from typing import List, Optional
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.mixture import GaussianMixture
from scipy.spatial import distance
from sklearn import preprocessing

from tool.core import data_types
from tool.core.classifier_wrappers import classifier_pipeline


class OoDMahalanobisScore:
    """
    Implementation of https://arxiv.org/pdf/2106.09022.pdf
    Requirements: Train set shall be In-Distribution
    """

    def __init__(self):
        self.ood_df = pd.DataFrame()

    def run(self, embeddings_files: List[str], use_gt_for_training: bool, output_dir: str,
            probabilities_file: Optional[str]):
        X_train, y_train, X, num_classes, relative_paths = classifier_pipeline.ClassifierPipeline.prepare_data(
            embeddings_files=embeddings_files,
            use_gt_for_training=use_gt_for_training,
            probabilities_file=probabilities_file,
            inference_mode=False)
        self.ood_df[data_types.RelativePathType.name()] = relative_paths
        self.__calculate_score(X_train, y_train, X, num_classes)
        return self.__store(output_dir)

    def get_ood_df(self):
        return self.ood_df

    def __calculate_score(self, X_train, y_train, X, num_classes):
        # Select mixture model
        mm = GaussianMixture

        # Fit the K class conditional Gaussian
        gaussians = []
        for label_id in range(num_classes):
            label_indices = np.argwhere(y_train == label_id).flatten()
            gm = mm(n_components=1, random_state=42).fit(np.take(X_train, label_indices, axis=0))
            means = gm.means_[0, :]
            covariances = gm.covariances_[0, :]
            gaussians.append((means, covariances))

        if len(gaussians) == 0:
            return None

        # Fit the background Gaussian
        background_gm = mm(n_components=1, random_state=42).fit(X_train)

        # Compute MD0 for each sample
        def mahalanobis(embedding, background_means, covariance_matrix):
            return distance.mahalanobis(embedding, background_means, covariance_matrix)

        background_cov_matrix = background_gm.covariances_[0, :, :]
        background_means_values = background_gm.means_[0, :]
        md_to_background = \
            np.apply_along_axis(func1d=mahalanobis, axis=1, arr=X, background_means=background_means_values,
                                covariance_matrix=background_cov_matrix)

        # Compute shared covariance matrix
        shared_cov = np.zeros(gaussians[0][1].shape)
        for gaus_parameters in gaussians:
            shared_cov += gaus_parameters[1]

        # Compute Relative Mahalanobis distance for each sample (NxK)
        def k_mahalanobis(embedding, gaussians_parameters, shared_covariance):
            dist = np.zeros(shape=(len(gaussians_parameters)))
            for idx, parameters in enumerate(gaussians_parameters):
                dist[idx] = distance.mahalanobis(embedding, parameters[0], shared_covariance)
            return dist

        md = np.apply_along_axis(func1d=k_mahalanobis, axis=1, arr=X, gaussians_parameters=gaussians,
                                 shared_covariance=shared_cov)
        # rmd = (md.T - md_to_background).T
        # md = rmd

        md = np.absolute(md)
        confidence = np.amin(md, axis=1)

        # POSTPROCESSING
        normalized_factor = preprocessing.MinMaxScaler().fit_transform(confidence.reshape(-1, 1))
        normalized_factor = normalized_factor.flatten()
        self.ood_df[data_types.OoDScoreType.name()] = normalized_factor.tolist()

    def __store(self, output_dir) -> str:
        timestamp_str = datetime.now().strftime("%y%m%d_%H%M%S")
        name = "".join(('./mahalanobis_ood_score_', timestamp_str, '.ood.pkl'))
        output_file = os.path.join(output_dir, name)
        with open(output_file, 'wb') as handle:
            pickle.dump(self.ood_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
        return output_file
