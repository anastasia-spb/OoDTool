import numpy as np
import abc
from typing import Optional


class IClassifier(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @classmethod
    @abc.abstractmethod
    def parameters_hint(cls) -> str:
        pass

    @classmethod
    @abc.abstractmethod
    def check_input_kwargs(cls, kwargs: dict) -> bool:
        pass

    @abc.abstractmethod
    def input_hint(self) -> str:
        pass

    @abc.abstractmethod
    def get_checkpoint(self):
        pass

    @abc.abstractmethod
    def run(self, X_train: Optional[np.ndarray], y_train: Optional[np.ndarray],
            X_test: np.ndarray, kwargs: dict, num_classes: int, output_dir: str,
            checkpoint: Optional[str] = None) -> np.ndarray:
        pass
