import numpy as np
from typing import Optional

from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tool.core.classifier_wrappers.classifiers.i_classifier import IClassifier


class SVCWrapper(IClassifier):
    kernels = ['rbf', 'linear', 'sigmoid']
    tag = 'SVC'

    def __init__(self):
        super().__init__()
        self.kernel = self.kernels[0]
        self.c = 1.0

    @classmethod
    def parameters_hint(cls):
        return 'Kernels: {0}, C: Regularization parameter; must be strictly positive.'.format(cls.kernels)

    @classmethod
    def check_input_kwargs(cls, kwargs: dict):
        checks = [lambda: "C" in kwargs, lambda: "kernel" in kwargs,
                  lambda: kwargs["kernel"] in cls.kernels, lambda: float(kwargs["C"]) > 0.0]

        for check in checks:
            if not check():
                return False
        return True

    def input_hint(self):
        return "{{'C' : '{0}', 'kernel' : '{1}' }}".format(self.c, self.kernel)

    def inference_mode(self):
        # Load from checkpoint is not implemented for SVC classifier
        return False

    def run(self, X_train: Optional[np.ndarray], y_train: Optional[np.ndarray], X_test: np.ndarray,
            kwargs: dict, num_classes: int, output_dir: str) -> np.ndarray:
        clf = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True,
                                                  kernel=kwargs["kernel"], C=float(kwargs["C"])))
        clf.fit(X_train, y_train)
        return clf.predict_proba(X_test)
