import os.path
from datetime import datetime
import numpy as np
import joblib
from typing import Optional

from sklearn.linear_model import LogisticRegression
from tool.core.classifier_wrappers.classifiers.i_classifier import IClassifier


class LogisticRegressionWrapper(IClassifier):
    def __init__(self, selected_model: str):
        super().__init__()
        self.selected_model = selected_model
        self.c = 1.0
        self.checkpoint = None

    def get_checkpoint(self):
        return self.checkpoint

    def run(self, X_train: Optional[np.ndarray], y_train: Optional[np.ndarray], X_test: np.ndarray,
            weight_decay: float, output_dir: str, checkpoint: Optional[str] = None,
            num_classes: int = None) -> np.ndarray:

        assert weight_decay >= 0.0

        dual = True
        if X_train.shape[0] > X_train.shape[1]:
            dual = False

        if checkpoint is not None:
            clf = joblib.load(checkpoint)
        else:
            if self.selected_model == "LogisticRegression_saga":
                clf = LogisticRegression(random_state=42, C=weight_decay, solver="saga", multi_class='multinomial',
                                         max_iter=1000)
            elif self.selected_model == "LogisticRegression_lbfgs":
                clf = LogisticRegression(random_state=42, C=weight_decay, solver="lbfgs", multi_class='multinomial',
                                         max_iter=1000)
            else:
                clf = LogisticRegression(random_state=42, C=weight_decay, solver="liblinear",
                                         max_iter=1000, dual=dual)
                self.selected_model = "LogisticRegression_liblinear"

            clf.fit(X_train, y_train)

            timestamp_str = datetime.utcnow().strftime("%y%m%d_%H%M%S.%f")[:-3]
            filename = "".join((timestamp_str, ".joblib.pkl"))
            checkpoint_path = os.path.join(output_dir, self.selected_model, "train")
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            self.checkpoint = os.path.join(checkpoint_path, filename)
            _ = joblib.dump(clf, self.checkpoint, compress=9)

        return clf.predict_proba(X_test)
