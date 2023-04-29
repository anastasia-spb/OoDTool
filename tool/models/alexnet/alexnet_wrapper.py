import os
import sys
import torch
from torchvision import transforms

from tool.models.i_model import IModel, ModelOutput
from tool.models.utils import custom_transformations
from tool.models.alexnet.alexnet_module import AlexNet


class AlexNetWrapperParameters:
    def __init__(self):
        self.weights_path = './pretrained_weights/embedders/alexnet_SummerWinter.pth'
        self.model_labels = {0: "Summer",
                             1: "Winter"}
        self.dropout = 0.41
        self.num_classes = 2
        self.batchsize = 16


class AlexNetWrapper(IModel):
    parameters = AlexNetWrapperParameters()

    def __init__(self, device, **kwargs):
        super().__init__()
        self.parameters.weights_path = kwargs["weights_path"]
        self.device = device
        self.model = AlexNet(self.parameters.num_classes, self.parameters.dropout).to(device)
        self.model.load_state_dict(torch.load(self.parameters.weights_path))
        self.model.eval()

    @classmethod
    def get_name(cls):
        return cls.__name__

    def get_model_labels(self):
        return self.parameters.model_labels

    def image_transformation_pipeline(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            custom_transformations.PerImageNormalization()])

    @classmethod
    def get_batchsize(cls):
        return cls.parameters.batchsize

    @classmethod
    def get_model_text(cls):
        return "Provide absolute path to model weights (.pth)"

    @classmethod
    def get_input_hint(cls):
        return "{{'weights_path' : '{0}' }}".format(cls.parameters.weights_path)

    def predict(self, img) -> ModelOutput:
        embeddings = []

        def copy_embeddings(m, i, o):
            embedding = o[:, :, 0, 0]
            embeddings.append(embedding)

        layer = self.model._modules.get('avgpool')
        _ = layer.register_forward_hook(copy_embeddings)
        prediction = self.model(img.to(self.device))

        return ModelOutput(embeddings=embeddings[0].detach().cpu(),
                           probabilities=prediction.detach().cpu()).to_dict()
