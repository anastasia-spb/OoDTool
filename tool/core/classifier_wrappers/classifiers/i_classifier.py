import numpy as np
import abc
from typing import Optional


class IClassifier(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def get_checkpoint(self):
        pass

    @abc.abstractmethod
    def run(self, X_train: Optional[np.ndarray], y_train: Optional[np.ndarray],
            X_test: np.ndarray, weight_decay: float, output_dir: str,
            checkpoint: Optional[str] = None, num_classes: int = None) -> np.ndarray:
        pass
