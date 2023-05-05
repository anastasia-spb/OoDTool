import os
import torch
import timm
from torch.nn import functional as F

from tool.core.model_wrappers.models.i_model import ModelOutput
from tool.core.model_wrappers.models.timm_resnet.i_embedder import ITimmEmbedder


class TimmViTEmbedder(ITimmEmbedder):
    def __init__(self, model_checkpoint, num_classes, checkpoint_path=''):
        super(TimmViTEmbedder, self).__init__()
        pretrained = True
        if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".pth"):
            pretrained = False
        self.model = timm.create_model(model_name=model_checkpoint, pretrained=pretrained,
                                       num_classes=num_classes, checkpoint_path=checkpoint_path)
        self.model.eval()

    def forward(self, inputs, num_classes, device, requires_grad) -> ModelOutput:

        embeddings = []

        def copy_embeddings(m, o):
            embeddings.append(o[0])

        layer = self.model._modules.get('fc_norm')
        _ = layer.register_forward_pre_hook(copy_embeddings)
        predictions = self.model(inputs.to(device))

        probabilities = torch.nn.functional.softmax(predictions, dim=1)
        grads = None
        if requires_grad:
            classification = torch.argmax(probabilities, dim=1)
            one_hot_encoding = F.one_hot(classification.clone().detach(), num_classes=num_classes).to(device)
            probabilities.backward(gradient=one_hot_encoding)
            grads = inputs.grad.detach().cpu()

        return ModelOutput(embeddings=embeddings[0].detach().cpu(),
                           probabilities=predictions[0].detach().cpu(),
                           grads=grads)
