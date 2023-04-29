import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from scipy.spatial import distance
from sklearn import preprocessing

from tool import data_types


class MahalanobisOoD(object):
    """
    Implementation of https://arxiv.org/pdf/2106.09022.pdf
    Requirements: Train set shall be In-Distribution
    """

    def __init__(self, embeddings_pkl: str):
        data_df = pd.read_pickle(embeddings_pkl)
        self.labels = data_df[data_types.LabelsType.name()][0]
        train_indices = data_df.index[data_df[data_types.TestSampleFlagType.name()] == False].tolist()
        embeddings = data_df[data_types.EmbeddingsType.name()].tolist()
        self.X = np.array(embeddings, dtype=np.dtype('float64'))
        tags = data_df.apply(lambda row: self.labels.index(row[data_types.LabelType.name()]), axis=1).values
        self.y = np.array(tags, dtype=np.dtype('float64'))
        if len(train_indices) == 0:
            self.X_train, _, self.y_train, _ = train_test_split(self.X, self.y, test_size=0.6, random_state=42)
        else:
            self.X_train, self.y_train = self.X[train_indices, :], self.y[train_indices]
        self.scores = np.zeros(shape=(self.X.shape[0]))
        self.result = {data_types.RelativePathType.name(): data_df[data_types.RelativePathType.name()].tolist(),
                       data_types.OoDScoreType.name(): []}

    def __call__(self, use_rmd=True):

        # Select mixture model
        mm = GaussianMixture
        # mm = BayesianGaussianMixture

        # Fit the K class conditional Gaussian
        gaussians = []
        for label_id in range(len(self.labels)):
            label_indices = np.argwhere(self.y_train == label_id).flatten()
            gm = mm(n_components=1, random_state=42).fit(np.take(self.X_train, label_indices, axis=0))
            means = gm.means_[0, :]
            covariances = gm.covariances_[0, :]
            gaussians.append((means, covariances))

        if len(gaussians) == 0:
            return None

        # Fit the background Gaussian
        background_gm = mm(n_components=1, random_state=42).fit(self.X_train)

        # Compute MD0 for each sample
        def mahalanobis(embedding, background_means, covariance_matrix):
            return distance.mahalanobis(embedding, background_means, covariance_matrix)

        background_cov_matrix = background_gm.covariances_[0, :, :]
        background_means_values = background_gm.means_[0, :]
        md_to_background = \
            np.apply_along_axis(func1d=mahalanobis, axis=1, arr=self.X, background_means=background_means_values,
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

        md = np.apply_along_axis(func1d=k_mahalanobis, axis=1, arr=self.X, gaussians_parameters=gaussians,
                                 shared_covariance=shared_cov)
        # rmd = (md.T - md_to_background).T
        # md = rmd

        md = np.absolute(md)
        confidence = np.amin(md, axis=1)

        # POSTPROCESSING
        normalized_factor = preprocessing.MinMaxScaler().fit_transform(confidence.reshape(-1, 1))
        normalized_factor = normalized_factor.flatten()
        self.result[data_types.OoDScoreType.name()] = normalized_factor.tolist()
        return pd.DataFrame.from_dict(self.result)
