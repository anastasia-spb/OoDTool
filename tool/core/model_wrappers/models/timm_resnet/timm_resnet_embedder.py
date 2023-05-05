import os
import torch
import timm
from torch.nn import functional as F

from tool.core.model_wrappers.models.i_model import ModelOutput
from tool.core.model_wrappers.models.timm_resnet.i_embedder import ITimmEmbedder


class TimmResnetEmbedder(ITimmEmbedder):
    def __init__(self, model_checkpoint, num_classes, checkpoint_path=''):
        super(TimmResnetEmbedder, self).__init__()
        pretrained = True
        if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".pth"):
            pretrained = False
        self.model = timm.create_model(model_name=model_checkpoint, pretrained=pretrained,
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

        return ModelOutput(embeddings=features.detach().cpu(), probabilities=probabilities.detach().cpu(), grads=grads)
