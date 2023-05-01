import torch

from tool.core.model_wrappers.models.i_model import IModel, ModelOutput
from tool.core.model_wrappers.models.alexnet.alexnet_module import AlexNet, AlexNetTransforms


class AlexNetWrapperParameters:
    def __init__(self):
        self.weights_path = './pretrained_weights/embedders/alexnet_SummerWinter.pth'
        self.model_labels = '[Summer, Winter]'
        self.dropout = 0.41
        self.batchsize = 16


class AlexNetWrapper(IModel):
    parameters = AlexNetWrapperParameters()

    def __init__(self, device, **kwargs):
        super().__init__()
        self.parameters.weights_path = kwargs["weights_path"]
        self.parameters.model_labels = kwargs["model_labels"]
        labels_list = self.parameters.model_labels.split(", ")
        self.idx_to_label = {i: labels_list[i] for i in range(len(labels_list))}
        self.device = device
        self.model = AlexNet(num_classes=len(labels_list), dropout=self.parameters.dropout).to(device)
        self.model.load_state_dict(torch.load(self.parameters.weights_path))
        self.model.eval()

    @classmethod
    def get_name(cls):
        return cls.__name__

    def get_model_labels(self):
        return self.idx_to_label

    def image_transformation_pipeline(self):
        return AlexNetTransforms()

    @classmethod
    def get_batchsize(cls):
        return cls.parameters.batchsize

    @classmethod
    def get_model_text(cls):
        return ""

    @classmethod
    def get_input_hint(cls):
        return "{{'weights_path' : '{0}', 'model_labels' : '{1}' }}".format(cls.parameters.weights_path,
                                                                            cls.parameters.model_labels)

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
