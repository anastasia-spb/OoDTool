import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Optional

from tool.core import data_types
from tool.core.utils import data_helpers
from tool.core.classifier_wrappers.classifiers.logistic_regression.lr_wrapper import LogisticRegressionWrapper
from tool.core.classifier_wrappers.classifiers.linear_classifier.linear_classifier_wrapper import \
    LinearClassifierWrapper

CLASSIFIER_WRAPPERS = {LogisticRegressionWrapper.tag: LogisticRegressionWrapper,
                       LinearClassifierWrapper.tag: LinearClassifierWrapper}


class ClassifierPipeline:
    def __init__(self, classifier_tag: str):
        self.probabilities_df = pd.DataFrame()
        self.classifier = CLASSIFIER_WRAPPERS[classifier_tag]()

    def input_hint(self):
        return self.classifier.input_hint()

    def parameters_hint(self):
        return self.classifier.parameters_hint()

    def check_input_kwargs(self, kwargs):
        return self.classifier.check_input_kwargs(kwargs)

    @staticmethod
    def prepare_data(embeddings_files, use_gt_for_training, probabilities_file, inference_mode):

        data_df, embeddings_columns = data_helpers.merge_data_files(embeddings_files, data_types.EmbeddingsType.name())
        num_classes = data_helpers.get_number_of_classes(data_df)

        def join_features(features_row):
            return np.concatenate(features_row, axis=None, dtype=np.dtype('float64'))

        embeddings_mat = data_df[embeddings_columns].values
        X = np.apply_along_axis(join_features, axis=1, arr=embeddings_mat)

        if inference_mode:
            return None, None, X, num_classes

        if use_gt_for_training:
            labels = data_df[data_types.LabelsType.name()][0]
            y_true = data_df.apply(lambda row: labels.index(row[data_types.LabelType.name()]), axis=1).values
            y = np.array(y_true, dtype=np.dtype('float64'))
            train_indices = data_df.index[data_df[data_types.TestSampleFlagType.name()] == False].tolist()
        else:
            y = data_helpers.get_predictions(probabilities_file)
            # For every sample embedder wrapper returns predictions of dim K+1, where K in number of classes.
            # We don't want to use samples for classifier training, which were classified into last (unknown) category
            valid_indices = np.asarray(y < num_classes).nonzero()
            # now we shall choose train samples from valid
            train_indices, _ = train_test_split(valid_indices[0], test_size=0.7, random_state=42)

        if len(train_indices) == 0:
            return None, None, X, num_classes

        X_train, y_train = X[train_indices, :], y[train_indices]
        num_classes = len(data_df[data_types.LabelsType.name()][0])
        return X_train, y_train, X, num_classes, data_df[data_types.RelativePathType.name()]

    def __store(self, output_folder: str) -> str:
        timestamp_str = datetime.now().strftime("%y%m%d_%H%M%S")
        name = "".join((self.classifier.tag, "_", timestamp_str, '.clf.pkl'))
        file = os.path.join(output_folder, name)
        self.probabilities_df.to_pickle(file)
        return file

    def get_probabilities_df(self) -> pd.DataFrame:
        return self.probabilities_df

    def run(self, embeddings_files: List[str], output_dir: str, use_gt_for_training: bool,
            probabilities_file: Optional[str], kwargs: List[dict]) -> str:
        """Walks through dataset directory and stores metadata information about images into <dataset_name>.meta.pkl file.
            Args:
                embeddings_files: List of all files with embeddings
                output_dir: Folder where will be stored output files and model weights if necessary
                use_gt_for_training: If GT labels shall be used for training or predictions from probabilities_file.
                probabilities_file: File with probabilities, which shall be used for training. Required only if
                                    use_gt_for_training is set to False
                kwargs: Classifier arguments
            Returns:
                Absolute path to <dataset_name>.clf.pkl file or None if input data are invalid.
            """
        X_train, y_train, X, num_classes, relative_paths = self.prepare_data(embeddings_files=embeddings_files,
                                                                             use_gt_for_training=use_gt_for_training,
                                                                             probabilities_file=probabilities_file,
                                                                             inference_mode=
                                                                             self.classifier.inference_mode())
        self.probabilities_df[data_types.RelativePathType.name()] = relative_paths
        for i, params in enumerate(kwargs):
            probabilities = self.classifier.run(X_train, y_train, X, params, num_classes, output_dir)
            self.probabilities_df["".join((data_types.ClassProbabilitiesType.name(), "_", str(i)))] = \
                probabilities.tolist()

        return self.__store(output_dir)
