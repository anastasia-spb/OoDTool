import os
import torch
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from torch.nn import functional as F
from torchvision import transforms

from tool.core.model_wrappers.models.i_model import IModel, ModelOutput
from tool.core.model_wrappers.models.timm_resnet.imagenet1000_clsidx_to_labels import ImageNetLabels, ImageNetLabelsTag

SUPPORTED_CHECKPOINTS = ['resnet34', 'resnet50', 'densenet121']


class TimmWrapperParameters:
    def __init__(self):
        self.model_checkpoint = SUPPORTED_CHECKPOINTS[0]
        self.model_labels = ImageNetLabelsTag
        self.batchsize = 16


class TimmResnetEmbedder(torch.nn.Module):
    def __init__(self, model_checkpoint, num_classes, checkpoint_path=''):
        super(TimmResnetEmbedder, self).__init__()
        self.model = timm.create_model(model_name=model_checkpoint, pretrained=True,
                                       num_classes=num_classes, checkpoint_path=checkpoint_path)
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

        return ModelOutput(embeddings=features.detach().cpu(), probabilities=probabilities.detach().cpu(),
                           grads=grads)


class TimmResnetWrapper(IModel):
    parameters = TimmWrapperParameters()
    tag = "TimmDSRSNet"

    def __init__(self, device, **kwargs):
        super().__init__()
        self.parameters.model_checkpoint = kwargs["model_checkpoint"]
        self.parameters.model_labels = kwargs["model_labels"]
        self.device = device
        self.checkpoint_path = ''
        if "checkpoint_path" in kwargs:
            self.checkpoint_path = self.check_checkpoint_path(kwargs["checkpoint_path"])

        if self.parameters.model_labels == ImageNetLabelsTag:
            self.num_classes = len(ImageNetLabels)
            self.labels_dict = ImageNetLabels
        else:
            labels_list = self.parameters.model_labels.split(", ")
            self.num_classes = len(labels_list)
            self.labels_dict = {i: labels_list[i] for i in range(len(labels_list))}

        if self.parameters.model_checkpoint in SUPPORTED_CHECKPOINTS:
            self.model = TimmResnetEmbedder(self.parameters.model_checkpoint, num_classes=self.num_classes,
                                            checkpoint_path=self.checkpoint_path)
        else:
            raise Exception("Unsupported embedder model")

        self.model.to(device)

    def check_checkpoint_path(self, path) -> str:
        if os.path.isfile(path) and path.endswith(".pth"):
            return path
        return ''

    @classmethod
    def get_name(cls):
        return "".join((cls.__name__, "_", cls.parameters.model_checkpoint))

    def get_model_labels(self):
        return self.labels_dict

    @classmethod
    def get_model_text(cls):
        return "Select from: {0}. Provide labels if they differ from ImageNetLabels and" \
               " absolute path to model checkpoint".format(SUPPORTED_CHECKPOINTS)

    @classmethod
    def get_input_hint(cls):
        return "{{'model_checkpoint' : '{0}', 'model_labels' : '{1}', 'checkpoint_path' : '' }}".format(
            cls.parameters.model_checkpoint,
            cls.parameters.model_labels)

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
