# Embedders + Classifiers

OoDTool implements wrappers for <a href="https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py"> AlexNet </a>,
<a href="https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/resnet.py"> ResNet </a> and 
<a href="https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/densenet.py"> Densenet </a> models.

ResNet and Densenet models are created with <a href="https://timm.fast.ai/"> `timm` </a> deep-learning library implemented by <a href="https://github.com/rwightman"> Ross Wightman </a>.
For these models you can either use your own pretrained weights and labels or generate embeddings with pretrained on ImageNet models

https://github.com/pytorch/vision/blob/main/torchvision/models/alexnet.py