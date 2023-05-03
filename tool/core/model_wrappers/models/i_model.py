import numpy as np
import abc
from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class ModelOutput:
    embeddings: torch.Tensor
    probabilities: torch.Tensor
    grads: Optional[np.ndarray]

    def __init__(self, embeddings: torch.Tensor, probabilities: torch.Tensor, grads: Optional[np.ndarray] = None):
        self.embeddings = embeddings
        self.probabilities = probabilities
        self.grads = grads

    def to_dict(self):
        return {
            "embeddings": self.embeddings,
            "probabilities": self.probabilities,
            "grads": self.grads
        }


class IModel(metaclass=abc.ABCMeta):
    def __init__(self, **kwargs):
        pass

    @abc.abstractmethod
    def image_transformation_pipeline(self):
        pass

    @abc.abstractmethod
    def get_image_crop(self):
        pass

    @classmethod
    @abc.abstractmethod
    def get_batchsize(cls):
        pass

    @abc.abstractmethod
    def get_model_labels(self) -> dict:
        pass

    @abc.abstractmethod
    def predict(self, img, requires_grad) -> dict:
        pass

    @classmethod
    @abc.abstractmethod
    def get_model_text(cls):
        pass

    @classmethod
    @abc.abstractmethod
    def get_input_hint(cls):
        pass

    def backward(self, one_hot_target):
        pass

