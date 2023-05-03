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
        return 'Solvers: {0}, C: Inverse of regularization strength; must be a positive float.'.format(cls.solvers)

    @classmethod
    def check_input_kwargs(cls, kwargs: dict):
        checks = [lambda: "C" in kwargs, lambda: "solver" in kwargs,
                  lambda: kwargs["solver"] in cls.solvers, lambda: float(kwargs["C"]) > 0.0]

        for check in checks:
            if not check():
                return False
        return True

    def input_hint(self):
        return "{{'C' : '{0}', 'solver' : '{1}' }}".format(self.c, self.solver)

    def get_checkpoint(self):
        return self.checkpoint

    def run(self, X_train: Optional[np.ndarray], y_train: Optional[np.ndarray], X_test: np.ndarray,
            kwargs: dict, num_classes: int, output_dir: str, checkpoint: Optional[str] = None) -> np.ndarray:

        if checkpoint is not None:
            clf = joblib.load(checkpoint)
        else:
            C = float(kwargs["C"])
            solver = kwargs["solver"]
            clf = LogisticRegression(random_state=42, C=C, solver=solver)
            clf.fit(X_train, y_train)
            timestamp_str = datetime.utcnow().strftime("%y%m%d_%H%M%S.%f")[:-3]
            filename = "".join((timestamp_str, ".joblib.pkl"))
            checkpoint_path = os.path.join(output_dir, self.tag, "train")
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            self.checkpoint = os.path.join(checkpoint_path, filename)
            _ = joblib.dump(clf, self.checkpoint, compress=9)

        return clf.predict_proba(X_test)

