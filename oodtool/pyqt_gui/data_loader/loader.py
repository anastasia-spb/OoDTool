import os

import logging
from dataclasses import dataclass
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Tuple

from oodtool.core.data_types import types
from oodtool.pyqt_gui.qt_utils.qt_types import ImageInfo
from oodtool.pyqt_gui.ood_images_tab.filters_frame import FilterSettings

from oodtool.pyqt_gui.qt_utils import find_pkl

from oodtool.core.metadata_generator.generator import generate_metadata


class MetadataFiles:

    def __init__(self):
        self.metadata_file = ''
        self.probabilities_file = ''
        self.ood_files = dict()
        self.distance_file = ''
        self.projection_file = ''

    @classmethod
    def __get_method_name_from_ood_file(cls, ood_file: str):
        _, filename = os.path.split(ood_file)
        return filename[: -len(".ood.pkl")]

    def scan_dir(self, metadata_dir: str):
        for file in os.listdir(metadata_dir):
            if file.endswith(".ood.pkl"):
                self.ood_files[self.__get_method_name_from_ood_file(file)] = os.path.join(metadata_dir, file)
            elif file.endswith(".meta.pkl"):
                self.metadata_file = os.path.join(metadata_dir, file)
            elif file.endswith(".clf.pkl"):
                self.probabilities_file = os.path.join(metadata_dir, file)
            elif file.endswith(".dist.pkl"):
                self.distance_file = os.path.join(metadata_dir, file)
            elif file.endswith(".2emb.pkl"):
                self.projection_file = os.path.join(metadata_dir, file)

    def log(self):
        logging.info('Found metadata files: %s, %s, %s, %s, %s', self.metadata_file,
                     self.probabilities_file, self.ood_files, self.distance_file, self.projection_file)


@dataclass(frozen=True)
class DataLoader:
    metadata_df: pd.DataFrame = None
    metadata_dir: str = ''
    dataset_root_dir: str = ''
    status_loaded: bool = False
    selected_ood_method: str = None
    metadata_files: MetadataFiles = None

    def __init__(self,
                 dataset_root_dir: Optional[str] = None,
                 metadata_dir: Optional[str] = None):

        if (dataset_root_dir is None) or (metadata_dir is None):
            return

        self.load_data(dataset_root_dir, metadata_dir)

    def load_data(self, dataset_root_dir: str, metadata_dir: str):
        if (not os.path.exists(metadata_dir)) or (not os.path.exists(dataset_root_dir)):
            return

        object.__setattr__(self, 'metadata_dir', metadata_dir)
        object.__setattr__(self, 'dataset_root_dir', dataset_root_dir)

        metadata_files = MetadataFiles()
        metadata_files.scan_dir(metadata_dir)
        object.__setattr__(self, 'metadata_files', metadata_files)

        dataset_description_json_file = find_pkl.get_description_file(metadata_dir)
        if dataset_description_json_file is None:
            return

        if not os.path.isfile(metadata_files.metadata_file):
            metadata_files.metadata_file = generate_metadata(dataset_description_json_file, metadata_dir,
                                                             self.dataset_root_dir)

        if len(self.metadata_files.ood_files) > 0:
            selected_ood_method = list(self.metadata_files.ood_files.keys())[0]
            object.__setattr__(self, 'selected_ood_method', selected_ood_method)

        try:
            self.__load_data()
            object.__setattr__(self, 'status_loaded', True)
        except Exception as error:
            logging.warning('An exception in DataLoader occurred: {}'.format(error))

    def get_available_ood_methods(self) -> List[str]:
        return list(self.metadata_files.ood_files.keys())

    def __load_ood_score(self, ood_file: str, metadata_df: pd.DataFrame):
        if os.path.isfile(ood_file):
            ood_score_df = pd.read_pickle(ood_file)
            if types.OoDScoreType.name() in metadata_df:
                metadata_df.drop([types.OoDScoreType.name()], axis=1, inplace=True)
            metadata_df = pd.merge(metadata_df, ood_score_df[
                [types.RelativePathType.name(), types.OoDScoreType.name()]],
                                   on=types.RelativePathType.name(), how='inner')
        return metadata_df

    def __load_confidence_info(self, conf_info_file: str, metadata_df: pd.DataFrame):
        if os.path.isfile(conf_info_file):
            probabilities_df = pd.read_pickle(conf_info_file)
            if types.ClassProbabilitiesType.name() in metadata_df:
                metadata_df.drop([types.ClassProbabilitiesType.name()], axis=1, inplace=True)
            if types.PredictedLabelsType.name() in metadata_df:
                metadata_df.drop([types.PredictedLabelsType.name()], axis=1, inplace=True)
            metadata_df = pd.merge(metadata_df, probabilities_df[[types.RelativePathType.name(),
                                                                  types.ClassProbabilitiesType.name(),
                                                                  types.PredictedLabelsType.name()]],
                                   on=types.RelativePathType.name(), how='inner')
        return metadata_df

    def __load_projection_info(self, projection_file, metadata_df: pd.DataFrame):
        if os.path.isfile(projection_file):
            projection_df = pd.read_pickle(projection_file)
            if types.ProjectedEmbeddingsType.name() in metadata_df:
                metadata_df.drop([types.ProjectedEmbeddingsType.name()], axis=1, inplace=True)
            metadata_df = pd.merge(metadata_df, projection_df[
                [types.RelativePathType.name(), types.ProjectedEmbeddingsType.name()]],
                                   on=types.RelativePathType.name(), how='inner')
        return metadata_df

    def get_projected_data_points(self) -> Union[np.ndarray, None]:
        if types.ProjectedEmbeddingsType.name() in self.metadata_df:
            return np.array(self.metadata_df["projected_embedding"].tolist())
        return None

    def get_ood_score(self) -> Union[List[np.ndarray], None]:
        if types.OoDScoreType.name() in self.metadata_df:
            return self.metadata_df[types.OoDScoreType.name()].values
        return None

    def __load_neighbours_info(self, distance_file, metadata_df: pd.DataFrame):
        if os.path.isfile(distance_file):
            knn_df = pd.read_pickle(distance_file)
            if types.NeighboursType.name() in metadata_df:
                metadata_df.drop([types.NeighboursType.name()], axis=1, inplace=True)
            metadata_df = pd.merge(metadata_df, knn_df[
                [types.RelativePathType.name(), types.NeighboursType.name()]],
                                   on=types.RelativePathType.name(), how='inner')
        return metadata_df

    def __load_data(self):
        metadata_df = pd.read_pickle(self.metadata_files.metadata_file)

        if (self.selected_ood_method is not None) and (self.selected_ood_method in self.metadata_files.ood_files):
            metadata_df = self.__load_ood_score(self.metadata_files.ood_files[self.selected_ood_method], metadata_df)

        metadata_df = self.__load_confidence_info(self.metadata_files.probabilities_file, metadata_df)
        metadata_df = self.__load_projection_info(self.metadata_files.projection_file, metadata_df)
        metadata_df = self.__load_neighbours_info(self.metadata_files.distance_file, metadata_df)

        if not metadata_df.empty:
            object.__setattr__(self, 'metadata_df', metadata_df)
        else:
            self.metadata_files.log()
            raise ValueError

    def __convert(self, row):
        relative_path = row[types.RelativePathType.name()]

        ood_score = -1.0
        if types.OoDScoreType.name() in self.metadata_df:
            ood_score = row[types.OoDScoreType.name()]

        confidence = None
        predicted_labels = None
        if types.ClassProbabilitiesType.name() in self.metadata_df:
            confidence = row[types.ClassProbabilitiesType.name()]
            predicted_labels = row[types.PredictedLabelsType.name()]

        return ImageInfo(relative_path=relative_path,
                         dataset_root_dir=self.dataset_root_dir,
                         metadata_dir=self.metadata_dir,
                         ood_score=ood_score,
                         confidence=confidence,
                         labels=self.get_labels(),
                         gt_label=row[types.LabelType.name()],
                         predicted_label=predicted_labels)

    def get_labels(self) -> Union[List[str], None]:
        if self.status_loaded:
            return self.metadata_df[types.LabelsType.name()][0]
        return None

    def get_indices_of_data_with_correct_prediction(self, head_idx: int = 0):
        if self.metadata_df is None:
            return None

        if types.PredictedLabelsType.name() not in self.metadata_df:
            return None

        if "miss" not in self.metadata_df:
            self.metadata_df["miss"] = self.metadata_df.apply(lambda row:
                                                              (row[types.LabelType.name()] ==
                                                               row[types.PredictedLabelsType.name()][head_idx]),
                                                              axis=1).values

        indices = self.metadata_df.index[self.metadata_df["miss"]].values
        return self.metadata_df.index.get_indexer(indices)

    def get_test_indices(self):
        if self.metadata_df is None:
            return []
        indices = self.metadata_df.index[self.metadata_df[types.TestSampleFlagType.name()]].values
        return self.metadata_df.index.get_indexer(indices)

    def get_train_indices(self):
        if self.metadata_df is None:
            return []
        indices = self.metadata_df.index[~self.metadata_df[types.TestSampleFlagType.name()]].values
        return self.metadata_df.index.get_indexer(indices)

    def get_indices(self, labels: List[str]) -> List[int]:
        if self.metadata_df is None:
            return []
            # Convert to global representation
        indices = self.metadata_df.index[self.metadata_df[types.LabelType.name()].isin(labels)].values
        return self.metadata_df.index.get_indexer(indices)

    def get_images_info_at(self, index: int):
        if self.status_loaded:
            return self.metadata_df.iloc[np.array([index])].apply(lambda row: self.__convert(row), axis=1).to_list()[0]
        return []

    def get_images_filtered_indices(self, filter_settings: FilterSettings, restrict_image_num: bool = True) -> List[int]:
        if (filter_settings.ood_method_name is not None) and (filter_settings.ood_method_name !=
                                                              self.selected_ood_method):
            # Reload data if OoD score file was changed
            object.__setattr__(self, 'selected_ood_method', filter_settings.ood_method_name)
            self.__load_data()

        if not self.status_loaded:
            logging.info("DataLoader: image info is empty")
            return []

        if (not filter_settings.show_train) and (not filter_settings.show_test):
            logging.info("DataLoader: neither train nor test data selected")
            return []

        if len(filter_settings.selected_labels) == 0:
            logging.info("DataLoader: no labels selected")
            return []

        if types.OoDScoreType.name() in self.metadata_df:
            self.metadata_df.sort_values(by=[types.OoDScoreType.name()], ascending=filter_settings.sort_ascending,
                                         inplace=True)

        filter_exp = lambda row: row[types.LabelType.name()].isin(filter_settings.selected_labels)
        if filter_settings.show_test and filter_settings.show_train:
            indices = self.metadata_df.loc[filter_exp].index.values
        else:
            indices = self.metadata_df.loc[self.metadata_df[types.TestSampleFlagType.name()] ==
                                           filter_settings.show_test].loc[filter_exp].index.values

        # Convert to global representation
        global_indices = self.metadata_df.index.get_indexer(indices)

        if len(global_indices) == 0:
            logging.info("DataLoader: all data were filtered out")
            return []

        if restrict_image_num:
            return global_indices[:filter_settings.num_images_to_show]
        else:
            return global_indices

    def get_images_filtered_info(self, filter_settings: FilterSettings):
        # Global indices
        indices = self.get_images_filtered_indices(filter_settings)
        if len(indices) > 0:
            return self.metadata_df.iloc[indices].apply(lambda row: self.__convert(row), axis=1).to_list()
        else:
            return []

    def get_k_neighbours(self, info: ImageInfo) -> Union[None, List[ImageInfo]]:
        if types.NeighboursType.name() not in self.metadata_df:
            return None

        img_row = self.metadata_df[self.metadata_df[types.RelativePathType.name()] == info.relative_path]
        neighbours_rel_paths = img_row[types.NeighboursType.name()].tolist()[0]

        neighbours_rows = self.metadata_df.iloc[pd.Index(self.metadata_df[types.RelativePathType.name()]).get_indexer(
            neighbours_rel_paths)]

        return neighbours_rows.apply(lambda row: self.__convert(row), axis=1).to_list()
