import os.path
from datetime import datetime
import numpy as np
import joblib
from typing import Optional

from sklearn.linear_model import LogisticRegression
from tool.core.classifier_wrappers.classifiers.i_classifier import IClassifier


class LogisticRegressionWrapper(IClassifier):
    solvers = {"liblinear", "lbfgs", "sag", "saga", "newton-cg"}
    tag = 'LogisticRegression'

    def __init__(self):
        super().__init__()
        self.solver = "liblinear"
        self.c = 1.0
        self.checkpoint = None

    @classmethod
    def parameters_hint(cls):
        return 'C: Inverse of regularization strength; must be a positive float.'

    def input_hint(self):
        return "{0}".format(self.c)

    def get_checkpoint(self):
        return self.checkpoint

    def run(self, X_train: Optional[np.ndarray], y_train: Optional[np.ndarray], X_test: np.ndarray,
            weight_decay: float, num_classes: int, output_dir: str, checkpoint: Optional[str] = None) -> np.ndarray:

        assert weight_decay >= 0.0

        if checkpoint is not None:
            clf = joblib.load(checkpoint)
        else:
            clf = LogisticRegression(random_state=42, C=weight_decay, solver=self.solver)
            clf.fit(X_train, y_train)
            timestamp_str = datetime.utcnow().strftime("%y%m%d_%H%M%S.%f")[:-3]
            filename = "".join((timestamp_str, ".joblib.pkl"))
            checkpoint_path = os.path.join(output_dir, self.tag, "train")
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            self.checkpoint = os.path.join(checkpoint_path, filename)
            _ = joblib.dump(clf, self.checkpoint, compress=9)

        return clf.predict_proba(X_test)

