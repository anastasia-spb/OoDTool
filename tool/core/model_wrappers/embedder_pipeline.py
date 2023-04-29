import gc
import pandas as pd
from tqdm import tqdm
from typing import Callable, List
import numpy as np

import torch
from torch.utils.data import DataLoader

from tool.models.utils.jpeg_dataset import JpegDataset
from tool.core.data_types import types

from tool.models.alexnet.alexnet_wrapper import AlexNetWrapper
from tool.models.timm_resnet.timm_resnet_wrapper import TimmResnetWrapper

MODEL_WRAPPERS = {AlexNetWrapper.get_name(): AlexNetWrapper,
                  TimmResnetWrapper.get_name(): TimmResnetWrapper}


class EmbedderPipeline:
    def __init__(self, metadata_file: str, data_dir: str, model_wrapper_name: str, use_cuda=True, **kwargs):
        self.metadata_file = metadata_file
        self.data_dir = data_dir

        if use_cuda:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.__empty_cache()
        else:
            self.device = torch.device('cpu')

        self.model_wrapper = MODEL_WRAPPERS[model_wrapper_name](self.device, **kwargs)
        self.loader = None

        self.model_output_df = pd.read_pickle(self.metadata_file)

    @staticmethod
    def get_supported_embedders():
        return MODEL_WRAPPERS.keys()

    @staticmethod
    def get_model_text(model_name: str):
        return MODEL_WRAPPERS[model_name].get_model_text()

    @staticmethod
    def get_input_hint(model_name: str):
        return MODEL_WRAPPERS[model_name].get_input_hint()

    def get_model_labels(self):
        return self.model_wrapper.get_model_labels()

    def get_model_output(self) -> pd.DataFrame:
        return self.model_output_df

    def predict(self, callback: Callable[[List[int]], None]):
        self.__setup()

        model_output = []
        with tqdm(self.loader) as pbar:
            for img, _, _ in pbar:
                callback([pbar.n, pbar.total])
                model_output.append(self.__forward(img))

        self.__postprocessing(model_output)

    def __empty_cache(self):
        if self.device == torch.device('cuda'):
            # Cuda maintenance
            gc.collect()
            torch.cuda.empty_cache()

    def __setup(self):
        self.__empty_cache()
        image_transformation = self.model_wrapper.image_transformation_pipeline()
        dataset = JpegDataset(self.metadata_file, self.data_dir, image_transformation)
        # Do not change shuffle to True, otherwise other fields in df wouldn't match
        self.loader = DataLoader(dataset, batch_size=self.model_wrapper.get_batchsize(), shuffle=False)

    def __forward(self, img):
        img = img.to(self.device)
        with torch.no_grad():
            model_output = self.model_wrapper.predict(img)
        return model_output

    def __map_model_classes_to_database_classes(self, probabilities, database_classes):
        model_labels = self.model_wrapper.get_model_labels()
        model_labels_list = list(model_labels.values())

        try:
            target_indices = [model_labels_list.index(database_class) for database_class in database_classes]
        except ValueError:
            print("Databases classes don't match with model classes. Returning empty probabilities.")
            return np.zeros(shape=(probabilities.shape[0], len(database_classes) + 1))

        def map_to_database_classes(probs, indices):
            max_prob_idx = np.argmax(probs, axis=0)
            if max_prob_idx in indices:
                additional_prob = 0.0
            else:
                additional_prob = probs[max_prob_idx]
            return np.append(np.take(probs, indices, axis=0), [additional_prob], axis=0)

        return np.apply_along_axis(func1d=map_to_database_classes, axis=1, arr=probabilities, indices=target_indices)

    def __postprocessing(self, model_output):

        if len(model_output) == 0:
            return

        embeddings = torch.empty((self.model_output_df.shape[0], model_output[0]["embeddings"].shape[1]),
                                 dtype=torch.float)

        probabilities = torch.empty((self.model_output_df.shape[0], model_output[0]["probabilities"].shape[1]),
                                    dtype=torch.float)

        current_idx = 0
        for batch_result in model_output:
            batch_size = len(batch_result["probabilities"])
            embeddings[current_idx:current_idx + batch_size, :] = batch_result["embeddings"]
            probabilities[current_idx:current_idx + batch_size, :] = batch_result["probabilities"]
            current_idx += batch_size

        self.model_output_df[types.EmbeddingsType.name()] = list(embeddings.numpy())

        database_classes = self.model_output_df[types.LabelsType.name()][0]
        probabilities = self.__map_model_classes_to_database_classes(probabilities.numpy(), database_classes)
        self.model_output_df[types.ClassProbabilitiesType.name()] = list(probabilities)
