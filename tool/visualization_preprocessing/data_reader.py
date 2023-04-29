import os
import pandas as pd
import numpy as np

from tool.data_types import types


class DataReader:
    def __init__(self,
                 dataset_root_dir: str,
                 embeddings_file: str,
                 ood_file: str,
                 use_gt_labels: bool,
                 ood_threshold: float,
                 scale_img_size=(3, 42, 42)):
        self.dataset_root_dir = dataset_root_dir

        self.data_df = pd.read_pickle(embeddings_file)
        self.data_df = self.data_df.filter([types.RelativePathType.name(), types.ClassProbabilitiesType.name(),
                                            types.LabelsType.name(), types.ProjectedEmbeddingsType.name(),
                                            types.LabelType.name(), types.TestSampleFlagType.name()])

        self.ood_score = None
        self.ood_score_cmap = None
        self.probabilities = None
        self.correct_prediction = None

        if os.path.isfile(ood_file):
            ood_df = pd.read_pickle(ood_file)
            self.data_df = pd.merge(self.data_df, ood_df[
                [types.RelativePathType.name(), types.OoDScoreType.name()]],
                                    on=types.RelativePathType.name(), how='inner')

        if os.path.isfile(ood_file):
            ood_score = self.data_df[types.OoDScoreType.name()].tolist()
            # ood_score = [(2**score - 1.0) for score in ood_score]
            ood_score_cmap = [0.1 if score < ood_threshold else 0.9 for score in ood_score]
            self.ood_score = np.array(ood_score, dtype=np.dtype('float64'))
            self.ood_score_cmap = np.array(ood_score_cmap, dtype=np.dtype('float64'))

        self.labels = self.data_df[types.LabelsType.name()][0]

        embeddings = self.data_df[types.ProjectedEmbeddingsType.name()].tolist()
        self.embeddings = np.array(embeddings, dtype=np.dtype('float64'))
        self.y = self.data_df.apply(lambda row: self.labels.index(row[types.LabelType.name()]), axis=1).values

        if not use_gt_labels:
            probabilities = self.data_df[types.ClassProbabilitiesType.name()].tolist()
            self.probabilities = np.array(probabilities, dtype=np.dtype('float64'))
            y_pred = np.argmax(self.probabilities, axis=1).tolist()
            y_true = self.data_df.apply(lambda row: self.labels.index(row[types.LabelType.name()]), axis=1).tolist()
            self.correct_prediction = [y_t == y_p for y_p, y_t in zip(y_pred, y_true)]
        else:
            self.correct_prediction = [True] * self.embeddings.shape[0]

        self.img_size = scale_img_size

    def get_label(self):
        return self.labels

    def get_indices_of_data_with_label(self, idx: int):
        return self.data_df.index[self.data_df[types.LabelType.name()] == self.labels[idx]].tolist()

    def get_indices_of_data_with_correct_prediction(self):
        if self.correct_prediction is not None:
            return self.data_df.index[self.correct_prediction].tolist()
        else:
            return None

    def get_indices_of_test_samples(self):
        return self.data_df.index[self.data_df[types.TestSampleFlagType.name()]].tolist()


