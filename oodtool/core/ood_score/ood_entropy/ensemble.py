import os
from typing import Set, List, Callable, Optional
import numpy as np
import warnings
import time
import joblib

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
ConvergenceWarning('ignore')


class Ensemble:
    def __init__(self, regularization_coefficients: Set[float], classifier_type: str = 'saga'):
        self.classifier_type = classifier_type
        self.regularization_coefficients = regularization_coefficients

    @classmethod
    def __setup_classifier(cls, C: float, classifier_type: str = 'saga', dual: bool = True):
        if (classifier_type == 'saga') or (classifier_type == 'lbfgs'):
            clf = LogisticRegression(random_state=42, C=C, solver=classifier_type, multi_class='multinomial',
                                     max_iter=300)
        elif classifier_type == 'liblinear':
            clf = LogisticRegression(random_state=42, C=C, solver=classifier_type, dual=dual,
                                     max_iter=300)
        else:
            warnings.warn("Unsupported classifier type: {0}. lbfgs selected.".format(classifier_type))
            clf = LogisticRegression(random_state=42, C=C, solver='lbfgs', multi_class='multinomial',
                                     max_iter=300)
        return clf

    def create_classifier_name_classifier(self, output_dir: str, emd_dim: int, C: float,
                                          checkpoint_tag: Optional[str] = None):
        checkpoint_path = os.path.join(output_dir, self.classifier_type)
        if checkpoint_tag is None:
            checkpoint_tag = ""
        filename = "".join(("{:.9f}".format(C), "_", str(emd_dim), "_", checkpoint_tag, ".joblib.pkl"))
        return os.path.join(checkpoint_path, filename)

    def train_and_predict_mean(self, X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, num_classes: int,
                               progress_callback: Callable[[List[int]], None] = None,
                               logs_callback: Callable[[str], None] = None, save_model=False,
                               output_dir: Optional[str] = None, checkpoint_tag: Optional[str] = None):

        mean_probabilities = np.zeros((X_test.shape[0], num_classes))
        num_reg_coefficients = len(self.regularization_coefficients)

        dual = X_train.shape[0] <= X_train.shape[1]
        for idx, C in enumerate(self.regularization_coefficients):
            start_time = time.perf_counter()
            clf = self.__setup_classifier(C, self.classifier_type, dual)
            clf.fit(X_train, y_train)
            mean_probabilities += clf.predict_proba(X_test)
            end_time = time.perf_counter()
            if save_model and (output_dir is not None):
                _ = joblib.dump(clf, self.create_classifier_name_classifier(output_dir, X_train.shape[1], C,
                                                                            checkpoint_tag), compress=9)

            if logs_callback is not None:
                logs_callback(f"Classifier with coefficient {C} finished in {end_time - start_time:0.4f} seconds")
            if progress_callback is not None:
                progress_callback([(idx + 1), num_reg_coefficients])

        return np.divide(mean_probabilities, num_reg_coefficients)

    @staticmethod
    def predict_from_checkpoints(X_test: np.ndarray, num_classes: int,
                                 checkpoints: List[str],
                                 progress_callback: Callable[[List[int]], None] = None,
                                 logs_callback: Callable[[str], None] = None):

        mean_probabilities = np.zeros((X_test.shape[0], num_classes))
        num_checkpoints = len(checkpoints)

        for idx, checkpoint in enumerate(checkpoints):
            start_time = time.perf_counter()
            clf = joblib.load(checkpoint)
            mean_probabilities += clf.predict_proba(X_test)
            end_time = time.perf_counter()

            if logs_callback is not None:
                logs_callback(
                    f"Classifier {idx + 1} from {num_checkpoints} finished in {end_time - start_time:0.4f} seconds")
            if progress_callback is not None:
                progress_callback([(idx + 1), num_checkpoints])

        return np.divide(mean_probabilities, num_checkpoints)
