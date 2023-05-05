import os
import torch
import timm
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from torch.nn import functional as F
from torchvision import transforms

from tool.core.model_wrappers.models.i_model import IModel, ModelOutput
from tool.core.model_wrappers.models.timm_resnet.imagenet1000_clsidx_to_labels import ImageNetLabels, ImageNetLabelsTag
from tool.core.model_wrappers.models.timm_resnet.timm_resnet_embedder import TimmResnetEmbedder
from tool.core.model_wrappers.models.timm_resnet.timm_vit_embedder import TimmViTEmbedder

SUPPORTED_RESNET_CHECKPOINTS = ['resnet34', 'resnet50', 'densenet121']
SUPPORTED_VIT_CHECKPOINTS = ['vit_tiny_patch16_224_in21k']
SUPPORTED_CHECKPOINTS = SUPPORTED_RESNET_CHECKPOINTS + SUPPORTED_VIT_CHECKPOINTS


class TimmWrapperParameters:
    def __init__(self):
        self.model_checkpoint = SUPPORTED_CHECKPOINTS[0]
        self.model_labels = ImageNetLabelsTag
        self.batchsize = 32


class TimmResnetWrapper(IModel):
    parameters = TimmWrapperParameters()

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

        if self.parameters.model_checkpoint in SUPPORTED_RESNET_CHECKPOINTS:
            self.model = TimmResnetEmbedder(self.parameters.model_checkpoint, num_classes=self.num_classes,
                                            checkpoint_path=self.checkpoint_path)
        elif self.parameters.model_checkpoint in SUPPORTED_VIT_CHECKPOINTS:
            self.model = TimmViTEmbedder(self.parameters.model_checkpoint, num_classes=self.num_classes,
                                         checkpoint_path=self.checkpoint_path)
        else:
            raise Exception("Unsupported embedder model")

        self.model.to(device)

    def check_checkpoint_path(self, path) -> str:
        if os.path.isfile(self.checkpoint_path) and self.checkpoint_path.endswith(".pth"):
            return path
        return ''

    @classmethod
    def get_name(cls):
        return cls.__name__

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
