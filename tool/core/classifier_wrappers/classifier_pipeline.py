import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from typing import List, Optional

from tool.core import data_types
from tool.core.utils import data_helpers
from tool.core.classifier_wrappers.classifiers.lr_wrapper import LogisticRegressionWrapper
from tool.core.classifier_wrappers.classifier_pipeline_config import store_pipeline_config, ClfConfig, \
    unite_configurations


class ClassifierPipeline:
    def __init__(self, classifier_tag: str):
        self.probabilities_df = pd.DataFrame()
        self.classifier = LogisticRegressionWrapper(classifier_tag)

    @classmethod
    def input_hint(cls):
        return LogisticRegressionWrapper.input_hint()

    @classmethod
    def parameters_hint(cls):
        return LogisticRegressionWrapper.parameters_hint()

    @staticmethod
    def prepare_data(embedding_file, use_gt_for_training, probabilities_file, inference_mode):
        data_df = pd.read_pickle(embedding_file)
        num_classes = data_helpers.get_number_of_classes(data_df)

        embeddings = data_df[data_types.EmbeddingsType.name()].tolist()
        embeddings = np.array(embeddings, dtype=np.dtype('float64'))

        if inference_mode:
            return None, None, embeddings, num_classes, data_df[data_types.RelativePathType.name()]

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
            return None, None, embeddings, num_classes, data_df[data_types.RelativePathType.name()]

        X_train, y_train = embeddings[train_indices, :], y[train_indices]
        return X_train, y_train, embeddings, num_classes, data_df[data_types.RelativePathType.name()]

    def __store(self, output_folder: str, store_config_file: bool,
                checkpoints: Optional[List[str]] = None, embeddings_file: Optional[str] = None,
                probabilities_file: Optional[str] = None) -> (str, str):
        timestamp_str = datetime.utcnow().strftime("%y%m%d_%H%M%S.%f")[:-3]
        # Store clf file
        name = "".join((self.classifier.selected_model, "_", timestamp_str, '.clf.pkl'))
        file = os.path.join(output_folder, name)
        self.probabilities_df.to_pickle(file)

        # Store configuration
        if store_config_file:
            config_file = store_pipeline_config(embeddings_file, probabilities_file, self.classifier.selected_model, checkpoints,
                                                output_folder)
        else:
            config_file = None

        return file, config_file

    def get_probabilities_df(self) -> pd.DataFrame:
        return self.probabilities_df

    def classify(self, embeddings_file: str, output_dir: str, use_gt_for_training: bool,
                 probabilities_file: Optional[str], weight_decay: List[float], checkpoint: Optional[str] = None) -> (str, str):
        """Walks through dataset directory and stores metadata information about images into <dataset_name>.meta.pkl file.
            Args:
                embeddings_file: List of all files with embeddings
                output_dir: Folder where will be stored output files and model weights if necessary
                use_gt_for_training: If GT labels shall be used for training or predictions from probabilities_file.
                probabilities_file: File with probabilities, which shall be used for training. Required only if
                                    use_gt_for_training is set to False
                weight_decay: Classifier arguments
                checkpoint: If checkpoint file is valid, then train part will be skipped
            Returns:
                Absolute path to <dataset_name>.clf.pkl file or None if input data are invalid.
            """

        if not os.path.isfile(embeddings_file):
            return ''

        inference_mode = (checkpoint is not None) and os.path.isfile(checkpoint) and checkpoint.endswith('.joblib.pkl')
        model_weights = None
        if inference_mode:
            model_weights = checkpoint

        X_train, y_train, X, num_classes, relative_paths = self.prepare_data(embedding_file=embeddings_file,
                                                                             use_gt_for_training=use_gt_for_training,
                                                                             probabilities_file=probabilities_file,
                                                                             inference_mode=inference_mode)
        self.probabilities_df[data_types.RelativePathType.name()] = relative_paths
        checkpoints = []
        for i, wd in enumerate(weight_decay):
            probabilities = self.classifier.run(X_train, y_train, X, wd, output_dir, model_weights)
            pretrained_model = self.classifier.get_checkpoint()
            if pretrained_model is not None:
                checkpoints.append(pretrained_model)

            self.probabilities_df["".join((data_types.ClassProbabilitiesType.name(), "_", str(i)))] = \
                probabilities.tolist()

        return self.__store(output_dir, True, checkpoints, embeddings_file, probabilities_file)

    def train_and_classify(self, embeddings_files: List[str], output_dir: str, use_gt_for_training: bool,
                           probabilities_file: Optional[str], weight_decays: List[float]) -> List[str]:

        if len(embeddings_files) < 1:
            return []

        output_files = []
        config_files = []

        for file in embeddings_files:
            output_file, config_file = self.classify(file, output_dir, use_gt_for_training, probabilities_file,
                                                     weight_decays)
            output_files.append(output_file)
            config_files.append(config_file)

        unite_configurations(output_dir, config_files)
        return output_files

    def classify_from_config(self, embeddings_file: str, clf_config: dict, output_dir: str) -> str:
        config = ClfConfig(clf_config)
        self.classifier = LogisticRegressionWrapper(config.get_tag())
        weights = config.get_weights()

        if not os.path.isfile(embeddings_file):
            return ''

        _, _, X, num_classes, relative_paths = self.prepare_data(embedding_file=embeddings_file,
                                                                 use_gt_for_training=False,
                                                                 probabilities_file=None,
                                                                 inference_mode=True)

        self.probabilities_df[data_types.RelativePathType.name()] = relative_paths
        checkpoints = []
        for i, checkpoint in enumerate(weights):
            probabilities = self.classifier.run(None, None, X, 0.0, output_dir, checkpoint)
            pretrained_model = self.classifier.get_checkpoint()
            if pretrained_model is not None:
                checkpoints.append(pretrained_model)

            self.probabilities_df["".join((data_types.ClassProbabilitiesType.name(), "_", str(i)))] = \
                probabilities.tolist()

        probs_file, _ = self.__store(output_dir, store_config_file=False)
        return probs_file
