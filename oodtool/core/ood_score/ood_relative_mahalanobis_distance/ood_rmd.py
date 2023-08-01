import pandas as pd
import numpy as np
import os
from typing import List, Callable, Optional, Union
from oodtool.core.data_types import types
from oodtool.core.utils import data_helpers

from sklearn.mixture import GaussianMixture
from sklearn.mixture import BayesianGaussianMixture
from scipy.spatial import distance
from sklearn import preprocessing


def mahalanobis(embedding, background_means, covariance_matrix):
    return distance.mahalanobis(embedding, background_means, covariance_matrix)


def k_mahalanobis(embedding, means, shared_covariance):
    dist = np.zeros(shape=(len(means)))
    for idx, mean in enumerate(means):
        dist[idx] = distance.mahalanobis(embedding, mean, shared_covariance)
    return dist


def rmd(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
        relative: bool = True, bayes: bool = False,
        progress_callback: Callable[[List[int]], None] = None,
        logs_callback: Callable[[str], None] = None) -> np.ndarray:
    """
    Implementation of https://arxiv.org/pdf/2106.09022.pdf
    Requirements: Train set shall be In-Distribution
    """

    label_encoder = preprocessing.LabelEncoder()
    y_train_cat = label_encoder.fit_transform(y_train)
    labels = np.unique(y_train_cat)

    models = []
    number_of_features = []
    for label in labels:
        x_train_k_indices = np.argwhere(y_train_cat == label)
        number_of_features.append(x_train_k_indices.shape[0])
        if bayes:
            model = BayesianGaussianMixture(n_components=1, covariance_type='full', random_state=0).fit(
                X_train[x_train_k_indices.flatten(), :])
        else:
            model = GaussianMixture(n_components=1, covariance_type='full', random_state=0).fit(
                X_train[x_train_k_indices.flatten(), :])
        models.append(model)

    means = [m.means_.flatten() for m in models]

    covariance = (number_of_features[0] - 1) * models[0].covariances_
    for i in range(1, len(models)):
        covariance += (number_of_features[i] - 1) * models[i].covariances_
    covariance = np.divide(covariance, X_train.shape[0])

    # Compute Relative Mahalanobis distance for each sample (NxK)
    md = np.apply_along_axis(func1d=k_mahalanobis, axis=1, arr=X_test, means=means,
                             shared_covariance=covariance)
    if relative:
        if bayes:
            # Fit the background Bayesian
            background_model = BayesianGaussianMixture(n_components=1, random_state=42).fit(X_train)
        else:
            # Fit the background Gaussian
            background_model = GaussianMixture(n_components=1, random_state=42).fit(X_train)

        # Compute MD0 for each sample
        background_cov_matrix = background_model.covariances_[0, :, :]
        background_means_values = background_model.means_[0, :]
        md_to_background = \
            np.apply_along_axis(func1d=mahalanobis, axis=1, arr=X_test, background_means=background_means_values,
                                covariance_matrix=background_cov_matrix)
        rmd_score = (md.T - md_to_background).T
        confidence = -np.amin(rmd_score, axis=1)
    else:
        confidence = np.amin(md, axis=1)

    # POSTPROCESSING
    normalized_confidence = preprocessing.MinMaxScaler().fit_transform(confidence.reshape(-1, 1)).flatten()
    return normalized_confidence


def score(embeddings_file: str, metadata_df: pd.DataFrame,
          relative: bool = True, bayes: bool = False,
          progress_callback: Callable[[List[int]], None] = None,
          logs_callback: Callable[[str], None] = None) -> Union[None, np.ndarray]:
    train_indices = metadata_df.index[~metadata_df[types.TestSampleFlagType.name()]].tolist()
    y_train = metadata_df[types.LabelType.name()].to_numpy()

    emb_df = pd.read_pickle(embeddings_file)
    assert emb_df[types.RelativePathType.name()].equals(metadata_df[types.RelativePathType.name()])
    embeddings = data_helpers.preprocess_embeddings(emb_df)

    return rmd(embeddings[train_indices, :], y_train[train_indices], embeddings, relative=relative, bayes=bayes,
               progress_callback=progress_callback, logs_callback=logs_callback)


def run():
    ood_session_dir = '/home/vlasova/datasets/ood_datasets/Letters_v20_b/dataset/oodsession_0'
    metadata_file_path = os.path.join(ood_session_dir, 'DatasetDescription.meta.pkl')
    metadata_df = pd.read_pickle(metadata_file_path)
    embeddings_file = 'timm_swin_base_patch4_window7_224.emb.pkl'
    ood_score = score(os.path.join(ood_session_dir, embeddings_file), metadata_df)


if __name__ == "__main__":
    run()
