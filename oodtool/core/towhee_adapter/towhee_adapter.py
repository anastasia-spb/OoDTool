from typing import Callable, List, Optional, Union
import time
import os
import pandas as pd
from tqdm import tqdm
from oodtool.core.towhee_adapter.image_path_dataset import ImagePathDataset
from towhee import pipeline


from oodtool.core import data_types
from oodtool.core import towhee_adapter


class ResultWrapper:
    def __init__(self, embedding, img_path: str):
        self.embeddings = embedding
        self.img_path = img_path

        # Internal embedders return also results from classification layer
        self.confidence = None
        self.predicted_labels = None
        self.predicted_probabilities = None
        self.predicted_probabilities_labels = None

    def to_dict(self):
        return {
            data_types.EmbeddingsType.name(): self.embeddings,
            data_types.ClassProbabilitiesType.name(): self.confidence,
            data_types.PredictedLabelsType.name(): self.predicted_labels,
            data_types.RelativePathType.name(): self.img_path,
            data_types.PredictedProbabilitiesType.name(): self.predicted_probabilities,
            data_types.LabelsForPredictedProbabilitiesType.name(): self.predicted_probabilities_labels
        }


class TowheeAdapter:

    def __init__(self, img_df: pd.DataFrame, data_dir: str, output_dir: str):
        self.img_df = img_df
        self.labels = self.img_df.labels[0]
        self.output_dir = output_dir

        self.data_dir = data_dir
        self.embedder = None

    def predict(self, model_id: str, progress_callback: Callable[[List[int]], None] = None,
                device: Optional[int] = None):

        embedding_pipeline = pipeline(model_id)
        dataset = ImagePathDataset(self.img_df, self.data_dir)

        results = []
        start_time = time.perf_counter()
        with tqdm(dataset) as pbar:
            for root_dir, img_path in pbar:
                embedding = embedding_pipeline(os.path.join(root_dir, img_path))
                results.append(ResultWrapper(embedding, img_path).to_dict())
                if progress_callback is not None:
                    progress_callback([pbar.n, pbar.total])
        end_time = time.perf_counter()
        print(f"Embedder finished in {end_time - start_time:0.4f} seconds")
        return self.__store_results(results, model_id)

    def get_filename(self, model_id: str):
        _, model_base_id = os.path.split(model_id)
        return os.path.join(self.output_dir, "".join((model_base_id, '.emb.pkl')))

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
    return towhee_adapter.OTHER


def get_towheeresnet50_embedder_id() -> str:
    """
    https://hub.towhee.io/towhee/image-embedding-resnet50
    """
    return "towhee/image-embedding-resnet50"


def get_swin_transformer_embedder_id() -> str:
    """
    https://towhee.io/towhee/image-embedding-swin-base-patch4-window7-224
    """
    return "towhee/image-embedding-swin-base-patch4-window7-224"

