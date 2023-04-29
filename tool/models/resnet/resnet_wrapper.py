from transformers import AutoFeatureExtractor, ResNetModel
import torch
from torchvision import transforms
from tool.models.i_model import IModel

SUPPORTED_CHECKPOINTS = ['microsoft/resnet-18', 'microsoft/resnet-50', 'microsoft/resnet-101']


class FeatureExtractor(object):
    def __init__(self, model_checkpoint):
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_checkpoint)

    def __call__(self, image):
        return self.feature_extractor(image, return_tensors="pt")

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Squeeze(object):
    def __call__(self, tensor):
        return torch.squeeze(tensor['pixel_values'])

    def __repr__(self):
        return self.__class__.__name__ + '()'


class ResnetWrapperParameters:
    def __init__(self):
        self.model_checkpoint = SUPPORTED_CHECKPOINTS[0]
        self.batchsize = 16


class ResnetWrapper(IModel):
    parameters = ResnetWrapperParameters()

    def __init__(self, **kwargs):
        super().__init__()
        self.parameters.model_checkpoint = kwargs["model_checkpoint"]

    @classmethod
    def get_model_text(cls):
        return "Select from: {0}".format(SUPPORTED_CHECKPOINTS)

    @classmethod
    def get_input_hint(cls):
        return "{{'model_checkpoint' : '{0}' }}".format(cls.parameters.model_checkpoint)

    @classmethod
    def image_transformation_pipeline(cls):
        return transforms.Compose([
            FeatureExtractor(cls.parameters.model_checkpoint),
            Squeeze()])

    @classmethod
    def get_batchsize(cls):
        return cls.parameters.batchsize

    @classmethod
    def load_model(cls, device):
        return ResNetModel.from_pretrained(cls.parameters.model_checkpoint).to(device)

    @classmethod
    def get_embedding(cls, model, img):
        return model(img)['pooler_output'].detach().cpu()
