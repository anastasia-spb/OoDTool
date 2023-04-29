import numpy as np
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

    def inference_mode(self):
        # Load from checkpoint is not implemented for LR classifier
        return False

    def input_hint(self):
        return "{{'C' : '{0}', 'solver' : '{1}' }}".format(self.c, self.solver)

    def run(self, X_train: Optional[np.ndarray], y_train: Optional[np.ndarray], X_test: np.ndarray,
            kwargs: dict, num_classes: int, output_dir: str) -> np.ndarray:
        clf = LogisticRegression(random_state=42, C=float(kwargs["C"]), solver=kwargs["solver"])
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test)

