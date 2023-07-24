from PIL import Image
import numpy as np
import czebra as cz
from czebra.loader.model_getters import MODEL_GETTERS


def is_model_implemented(model_info):
    model_base = "_".join([model_info.model_framework, model_info.model_arch])
    model = MODEL_GETTERS.get(model_base, None)
    return model is not None


def get_classification_models_ids():
    classification_models = []
    torch_models = cz.search_model(framework="torch")
    for model in torch_models:
        if (model.params.modelType == 'Classification') and is_model_implemented(model):
            classification_models.append(model.model_id)

    return classification_models


def get_saliency_map(path_image: str, model_id: str):
    pil_image = Image.open(path_image).convert('RGB')
    img = np.array(pil_image)
    # Convert RGB to BGR
    img = img[:, :, ::-1].copy()

    model = cz.load_model(model_id, requires_grad=True)
    result = model.predict(img)

    maps = []
    for i, classifications in enumerate(result.classifications):
        for head_classification in classifications:
            saliency_map = head_classification.visualize_saliency_map(img)
            maps.append((np.hstack([img, saliency_map]), head_classification.top_class))
    return maps
