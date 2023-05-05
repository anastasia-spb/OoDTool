import torch
import abc

from tool.core.model_wrappers.models.i_model import ModelOutput


class ITimmEmbedder(torch.nn.Module, metaclass=abc.ABCMeta):
    def __init__(self):
        super(ITimmEmbedder, self).__init__()
        pass

    @abc.abstractmethod
    def forward(self, inputs, num_classes, device, requires_grad) -> ModelOutput:
        pass
