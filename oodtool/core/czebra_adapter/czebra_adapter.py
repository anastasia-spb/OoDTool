from typing import Callable, List, Optional, Union
import time
import os
import torch
import pandas as pd
from tqdm import tqdm
from oodtool.core.czebra_adapter.czebra_dataset import CZebraDataset

import czebra as cz
from czebra.models.timm_embedders import TimmEmbedder

from catalights.trafficlights_classes.torch_shared_regnet_trafficlights_classes import trafficlights_classes

from oodtool.core import data_types
from oodtool.core import czebra_adapter


class ResultWrapper:
    def __init__(self, result: cz.Result, img_path: str, model_labels: Optional[List[List[str]]] = None):
        self.embeddings = result.image_embedding
        self.img_path = img_path
        if result.classifications is None:
            self.confidence = None
            self.predicted_labels = None
            self.predicted_probabilities = None
            self.predicted_probabilities_labels = None
        else:
            confidence = []
            predicted_labels = []
            predicted_probabilities = []
            predicted_probabilities_labels = []
            for idx, head_classification in enumerate(result.classifications[0]):
                confidence.append(head_classification.top_score)
                predicted_labels.append(head_classification.top_class)
                predicted_probabilities.append(head_classification.probabilities)
                if model_labels is None:
                    predicted_probabilities_labels.append(head_classification.classes)
                else:
                    # In case model labels stored in config are outdated
                    predicted_probabilities_labels.append(model_labels[idx])
            self.confidence = confidence
            self.predicted_labels = predicted_labels
            self.predicted_probabilities = predicted_probabilities
            self.predicted_probabilities_labels = predicted_probabilities_labels

    def to_dict(self):
        return {
            data_types.EmbeddingsType.name(): self.embeddings,
            data_types.ClassProbabilitiesType.name(): self.confidence,
            data_types.PredictedLabelsType.name(): self.predicted_labels,
            data_types.RelativePathType.name(): self.img_path,
            data_types.PredictedProbabilitiesType.name(): self.predicted_probabilities,
            data_types.LabelsForPredictedProbabilitiesType.name(): self.predicted_probabilities_labels
        }


class CZebraAdapter:

    def __init__(self, img_df: pd.DataFrame, data_dir: str, output_dir: str):
        self.img_df = img_df
        self.labels = self.img_df.labels[0]
        self.output_dir = output_dir

        self.data_dir = data_dir
        self.embedder = None

    @classmethod
    def _load_embedder_model(cls, model_id: str, device: int):

        model_description = model_id.split('_')
        framework = model_description[0]
        model_classes = None

        if framework == "timm":
            timm_model_id = "_".join(model_description[1:])
            embedder_model = TimmEmbedder(model_name=timm_model_id, device=device)
        else:
            arch = model_description[1]
            usecase_name = model_description[2]

            if arch == 'embedder':
                embedder_model = cz.load_model(model_id, device=device)
            elif arch == 'shared-regnet' and usecase_name == czebra_adapter.TRAFFICLIGHTS:
                from czebra.models.regnet.torch_shared_regnet_embedder import RegNetEmbedder

                underlying_predictor_model = cz.load_model(model_id, device=device)
                embedder_model = RegNetEmbedder(predictor_classifier=underlying_predictor_model,
                                                requires_grad=False)
                model_classes = [list(trafficlights_classes[key].values()) for key in trafficlights_classes.keys()]
            else:
                raise ValueError(f'Unknown model id: {model_id}')

        return embedder_model, model_classes

    def predict(self, model_id: str, progress_callback: Callable[[List[int]], None] = None,
                device: Optional[int] = None):

        if device is None:
            device = 0 if torch.cuda.is_available() else -1

        embedder, model_classes = self._load_embedder_model(model_id, device)
        dataset = CZebraDataset(self.img_df, self.data_dir)

        results = []
        start_time = time.perf_counter()
        with tqdm(dataset) as pbar:
            for img, _, path in pbar:
                classification_result = embedder.predict(img)
                results.append(ResultWrapper(classification_result, path, model_classes).to_dict())
                if progress_callback is not None:
                    progress_callback([pbar.n, pbar.total])
        end_time = time.perf_counter()
        print(f"Embedder finished in {end_time - start_time:0.4f} seconds")
        return self.__store_results(results, model_id)

    def get_filename(self, model_id: str):
        return os.path.join(self.output_dir, "".join((model_id, '.emb.pkl')))

    def get_probabilities_filename(self, model_id: str):
        return os.path.join(self.output_dir, "".join((model_id, '.clf.pkl')))

    def __store_results(self, results: List[dict], model_id: str):
        df = pd.DataFrame(results)
        output_file = self.get_filename(model_id)
        df[[data_types.RelativePathType.name(), data_types.EmbeddingsType.name()]].to_pickle(output_file)

        output_probabilities_file = None
        if not df[data_types.ClassProbabilitiesType.name()].isnull().all():
            df_probabilities = pd.DataFrame(results, columns=[data_types.RelativePathType.name(),
                                                              data_types.ClassProbabilitiesType.name(),
                                                              data_types.PredictedLabelsType.name(),
                                                              data_types.PredictedProbabilitiesType.name(),
                                                              data_types.LabelsForPredictedProbabilitiesType.name()])

            output_probabilities_file = self.get_probabilities_filename(model_id)
            df_probabilities.to_pickle(output_probabilities_file)

        return output_file, output_probabilities_file


def determine_dataset_usecase(labels: Union[List[str], None]) -> Union[str, None]:
    if labels is None:
        return None
    # Check if dataset labels are subset of traffic lights labels for first head
    model_labels = list(trafficlights_classes["target1"].values())
    if (len(labels) > 0) and (len(model_labels) > 0) and (len(set(labels).intersection(set(model_labels))) > 0):
        return czebra_adapter.TRAFFICLIGHTS

    return czebra_adapter.OTHER


def get_towheeresnet50_embedder_id() -> str:
    towheeresnet50_embedders = cz.search_model(framework="torch", arch="embedder", usecase="towheeresnet50")
    return towheeresnet50_embedders[-1].model_id


def get_swin_transformer_embedder_id() -> str:
    return "timm_swin_base_patch4_window7_224"


def get_model_id_from_usecase(usecase_name: str = "") -> Union[str, None]:
    if usecase_name == czebra_adapter.TRAFFICLIGHTS:
        shared_regnet_tl_embedders_id = cz.search_model(framework="torch", arch="shared-regnet",
                                                        usecase=usecase_name)
        return shared_regnet_tl_embedders_id[-1].model_id

    if usecase_name == czebra_adapter.AGRO:
        agro_embedders_ids = cz.search_model(framework="torch", arch="embedder", usecase=usecase_name)
        return agro_embedders_ids[-1].model_id

    return None
