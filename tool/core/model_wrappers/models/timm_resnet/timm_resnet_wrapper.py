import torch
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from torch.nn import functional as F
from torchvision import transforms

from tool.core.model_wrappers.models.i_model import IModel, ModelOutput
from tool.core.model_wrappers.models.timm_resnet.imagenet1000_clsidx_to_labels import ImageNetLabels

SUPPORTED_CHECKPOINTS = ['resnet34', 'resnet50', 'densenet121']


class TimmWrapperParameters:
    def __init__(self):
        self.model_checkpoint = SUPPORTED_CHECKPOINTS[0]
        self.batchsize = 32


class TimmResnetEmbedder(torch.nn.Module):
    def __init__(self, model_checkpoint):
        super(TimmResnetEmbedder, self).__init__()
        self.model = timm.create_model(model_name=model_checkpoint, pretrained=True)
        self.model.eval()

    def forward(self, inputs, num_classes, device, requires_grad) -> ModelOutput:
        features = self.model.forward_features(inputs)
        features = self.model.global_pool(features)
        features = features.flatten(1)
        if self.model.drop_rate > 0.:
            predictions = F.dropout(features, p=self.model.drop_rate, training=False)
        else:
            predictions = features
        if hasattr(self.model, 'classifier'):
            predictions = self.model.classifier(predictions)
        elif hasattr(self.model, 'fc'):
            predictions = self.model.fc(predictions)
        else:
            raise Exception("Unknown classification layer.")
        probabilities = torch.nn.functional.softmax(predictions, dim=1)
        grads = None
        if requires_grad:
            classification = torch.argmax(probabilities, dim=1)
            one_hot_encoding = F.one_hot(torch.tensor(classification), num_classes=num_classes).to(device)
            probabilities.backward(gradient=one_hot_encoding)
            grads = inputs.grad.detach().cpu()

        return ModelOutput(embeddings=features.detach().cpu(), probabilities=probabilities.detach().cpu(), grads=grads)


class TimmResnetWrapper(IModel):
    parameters = TimmWrapperParameters()

    def __init__(self, device, **kwargs):
        super().__init__()
        self.parameters.model_checkpoint = kwargs["model_checkpoint"]
        self.device = device
        self.model = TimmResnetEmbedder(self.parameters.model_checkpoint)
        self.model.to(device)
        self.num_classes = len(ImageNetLabels)

    @classmethod
    def get_name(cls):
        return cls.__name__

    def get_model_labels(self):
        return ImageNetLabels

    @classmethod
    def get_model_text(cls):
        return "Select from: {0}".format(SUPPORTED_CHECKPOINTS)

    @classmethod
    def get_input_hint(cls):
        return "{{'model_checkpoint' : '{0}' }}".format(cls.parameters.model_checkpoint)

    def image_transformation_pipeline(self):
        config = resolve_data_config({}, model=self.model.model)
        return create_transform(**config)

    def get_image_crop(self):
        # Get only resize and crop parts
        crop_transform = self.image_transformation_pipeline().transforms[0:2]
        return transforms.Compose(crop_transform)

    @classmethod
    def get_batchsize(cls):
        return cls.parameters.batchsize

    def predict(self, img, requires_grad) -> dict:
        img.requires_grad = requires_grad
        return self.model(img.to(self.device), self.num_classes, self.device, requires_grad).to_dict()

