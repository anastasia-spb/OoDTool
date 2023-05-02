import torch
from torchvision import transforms

from tool.core.model_wrappers.models.i_model import IModel, ModelOutput
from catalights.models.multihead_regnet import regnet_mh_y_800mf
from catalights.models.labels import labels_map


class RegnetWrapperParameters:
    def __init__(self):
        self.batchsize = 16
        self.model_checkpoint =\
            './pretrained_weights/embedders/shared-regnet_trafficlights_v8/model.best.pth'


class RegnetWrapper(IModel):
    parameters = RegnetWrapperParameters()

    def __init__(self, device, **kwargs):
        super().__init__()
        self.parameters.model_checkpoint = kwargs["model_checkpoint"]
        self.device = device
        self.model = self.__load_model(device)

    @classmethod
    def get_name(cls):
        return cls.__name__

    def get_model_labels(self):
        return labels_map["target1"]

    @classmethod
    def get_model_text(cls):
        return "Provide absolute path to model checkpoint."

    @classmethod
    def get_input_hint(cls):
        return "{{'model_checkpoint' : '{0}' }}".format(cls.parameters.model_checkpoint)

    def image_transformation_pipeline(self):
        return transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    @classmethod
    def get_batchsize(cls):
        return cls.parameters.batchsize

    @classmethod
    def __load_model(cls, device):
        wrapper = regnet_mh_y_800mf(pretrained=True, num_classes=[24, 3], softmax=True)
        checkpoint = torch.load(cls.parameters.model_checkpoint)
        wrapper.load_state_dict(checkpoint, strict=True)
        wrapper.eval()
        return wrapper.to(device)

    def backward(self, one_hot_target):
        pass

    def predict(self, img, requires_grad) -> dict:
        embeddings = []

        def copy_embeddings(m, o):
            embeddings.append(o[0])

        layer = self.model._modules.get('fc')  # fc - for 24
        _ = layer.register_forward_pre_hook(copy_embeddings)
        prediction = self.model(img.to(self.device))

        # Only first head is evaluated
        return ModelOutput(embeddings=embeddings[0].detach().cpu(),
                           probabilities=prediction[0].detach().cpu()).to_dict()

