import numpy as np
import time
from typing import Set, List, Callable, Optional
from scipy.stats import entropy

from sklearn import preprocessing

from oodtool.core.ood_score.ood_entropy.ensemble import Ensemble
from oodtool.core.ood_score.utils.split_to_train_and_test import get_train_indices


class OoD:
    default_regularization_coefficients = {1e-5, 1.0, 1e5}
    default_classifier_type = 'saga'

    def __init__(self, regularization_coefficients: Set[float] = None,
                 classifier_type: str = default_classifier_type):
        if regularization_coefficients is None:
            regularization_coefficients = self.default_regularization_coefficients

        self.regularization_coefficients = set(regularization_coefficients)
        self.classifier_type = classifier_type
        self.ensemble = Ensemble(self.regularization_coefficients, classifier_type)

    def fit_and_score(self, X_train: List[np.ndarray], y_train: List[str], X_test: List[np.ndarray],
                      progress_callback: Callable[[List[int]], None] = None,
                      logs_callback: Callable[[str], None] = None,
                      checkpoints: Optional[List[List[str]]] = None, save_model=False,
                      output_dir: Optional[str] = None, checkpoint_tag: Optional[str] = None) -> np.ndarray:

        start_time = time.perf_counter()
        label_encoder = preprocessing.LabelEncoder()
        y_train_cat = label_encoder.fit_transform(y_train)
        train_indices, num_classes = get_train_indices(y_train, y_train_cat, logs_callback)
        mean_probabilities = np.zeros((X_test[0].shape[0], num_classes))
        checkpoint = time.perf_counter()
        if logs_callback is not None:
            logs_callback(f"OoD preprocessing finished in {checkpoint - start_time:0.4f} seconds")

        for idx, train_embeddings in enumerate(X_train):
            start_time = time.perf_counter()
            if checkpoints is None:
                mean_probabilities += self.ensemble.train_and_predict_mean(train_embeddings[train_indices, :],
                                                                           y_train_cat[train_indices],
                                                                           X_test[idx], num_classes, progress_callback,
                                                                           logs_callback, save_model, output_dir,
                                                                           checkpoint_tag)
            else:
                assert len(checkpoints) == len(X_train)
                mean_probabilities += self.ensemble.predict_from_checkpoints(X_test[idx], num_classes, checkpoints[idx],
                                                                             progress_callback,
                                                                             logs_callback)
            checkpoint = time.perf_counter()
            if logs_callback is not None:
                logs_callback(f"OoD ensemble for one embeddings set finished in {checkpoint - start_time:0.4f} seconds")

        start_time = time.perf_counter()
        mean_probabilities = np.divide(mean_probabilities, len(X_train))

        def calculate_entropy(mean_dist):
            return entropy(mean_dist)

        score = np.apply_along_axis(calculate_entropy, axis=1, arr=mean_probabilities)
        normalized_score = self.__normalize_score(score)
        checkpoint = time.perf_counter()
        if logs_callback is not None:
            logs_callback(f"OoD scoring finished in {checkpoint - start_time:0.4f} seconds")

        return normalized_score

    @staticmethod
    def __normalize_score(x):
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(0.0, 1.0))
        rescaled = min_max_scaler.fit_transform(np.reshape(x, (x.shape[0], 1)))
        return np.reshape(rescaled, (x.shape[0]))
