import os.path
from datetime import datetime
import numpy as np
import joblib
from typing import Optional

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SUPPORTED_CLASSIFIERS = ["LogisticRegression_liblinear", "LogisticRegression_newton-cg",
                         "LogisticRegression_lbfgs", "SGDClassifier"]


class LogisticRegressionWrapper:
    def __init__(self, selected_model: str):
        super().__init__()
        self.selected_model = selected_model
        self.c = 1.0
        self.checkpoint = None

    @classmethod
    def parameters_hint(cls):
        return 'C: Inverse of regularization strength; must be a positive float.'

    @classmethod
    def input_hint(cls):
        return "0.0"

    def get_checkpoint(self):
        return self.checkpoint

    def run(self, X_train: Optional[np.ndarray], y_train: Optional[np.ndarray], X_test: np.ndarray,
            weight_decay: float, output_dir: str, checkpoint: Optional[str] = None) -> np.ndarray:

        assert weight_decay >= 0.0

        if checkpoint is not None:
            clf = joblib.load(checkpoint)
        else:
            if self.selected_model == "LogisticRegression_liblinear":
                clf = LogisticRegression(random_state=42, C=weight_decay, solver="liblinear")
            elif self.selected_model == "LogisticRegression_newton-cg":
                clf = LogisticRegression(random_state=42, C=weight_decay, solver="newton-cg")
            elif self.selected_model == "LogisticRegression_lbfgs":
                clf = LogisticRegression(random_state=42, C=weight_decay, solver="lbfgs")
            else:
                clf = make_pipeline(StandardScaler(), SGDClassifier(max_iter=1000, tol=1e-3, alpha=weight_decay,
                                                                    loss='log_loss'))
                self.selected_model = "SGDClassifier"

            clf.fit(X_train, y_train)

            timestamp_str = datetime.utcnow().strftime("%y%m%d_%H%M%S.%f")[:-3]
            filename = "".join((timestamp_str, ".joblib.pkl"))
            checkpoint_path = os.path.join(output_dir, self.selected_model, "train")
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            self.checkpoint = os.path.join(checkpoint_path, filename)
            _ = joblib.dump(clf, self.checkpoint, compress=9)

        return clf.predict_proba(X_test)
