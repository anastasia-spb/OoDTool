import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from tool.core import data_types
from tool.core.classifier_wrappers.classifiers.logistic_regression.lr_wrapper import LogisticRegressionWrapper
from tool.core.classifier_wrappers.classifiers.svm_svc.smv_svc_wrapper import SVCWrapper
from tool.core.classifier_wrappers.classifiers.linear_classifier.linear_classifier_wrapper import LinearClassifierWrapper

CLASSIFIER_WRAPPERS = {LogisticRegressionWrapper.tag: LogisticRegressionWrapper,
                       SVCWrapper.tag: SVCWrapper,
                       LinearClassifierWrapper.tag: LinearClassifierWrapper}


class ClassifierPipeline:
    def __init__(self, classifier_tag: str):
        self.probabilities = {data_types.RelativePathType.name(): [],
                              data_types.ClassProbabilitiesType.name(): np.ndarray}
        self.classifier = CLASSIFIER_WRAPPERS[classifier_tag]()

    def get_tag(self):
        return self.classifier.tag

    def input_hint(self):
        return self.classifier.input_hint()

    def parameters_hint(self):
        return self.classifier.parameters_hint()

    def check_input_kwargs(self, kwargs):
        return self.classifier.check_input_kwargs(kwargs)

    def __prepare_data(self, embeddings_file, use_gt_for_training, inference_mode):
        data_df = pd.read_pickle(embeddings_file)
        self.probabilities[data_types.RelativePathType.name()] = data_df[data_types.RelativePathType.name()].copy()
        num_classes = len(data_df[data_types.LabelsType.name()][0])

        X = data_df[data_types.EmbeddingsType.name()].tolist()
        X = np.array(X, dtype=np.dtype('float64'))

        if inference_mode:
            return None, None, X, num_classes

        if use_gt_for_training:
            labels = data_df[data_types.LabelsType.name()][0]
            y_true = data_df.apply(lambda row: labels.index(row[data_types.LabelType.name()]), axis=1).values
            y = np.array(y_true, dtype=np.dtype('float64'))
            train_indices = data_df.index[data_df[data_types.TestSampleFlagType.name()] == False].tolist()
        else:
            predictions = data_df[data_types.ClassProbabilitiesType.name()].tolist()
            y = np.argmax(predictions, axis=1)
            # For every sample embedder wrapper returns predictions of dim K+1, where K in number of classes.
            # We don't want to use samples for classifier training, which were classified into last (unknown) category
            valid_indices = np.asarray(y < num_classes).nonzero()
            # now we shall choose train samples from valid
            train_indices, _ = train_test_split(valid_indices[0], test_size=0.7, random_state=42)

        if len(train_indices) == 0:
            return None, None, X, num_classes

        X_train, y_train = X[train_indices, :], y[train_indices]
        self.probabilities[data_types.RelativePathType.name()] = data_df[data_types.RelativePathType.name()].copy()
        num_classes = len(data_df[data_types.LabelsType.name()][0])
        return X_train, y_train, X, num_classes

    def __store(self, probabilities_df, embeddings_file: str, kwargs: dict, store_in_folder):
        base = os.path.splitext(os.path.basename(embeddings_file))[0]
        timestamp_str = datetime.now().strftime("%y%m%d_%H%M%S")
        name = "".join((base, timestamp_str, '.clf.pkl'))
        if store_in_folder:
            output_pkl_dir = os.path.join(self.settings.metadata_folder,
                                          string_from_kwargs(self.classifier.get_tag(), kwargs))
            if not os.path.exists(output_pkl_dir):
                os.makedirs(output_pkl_dir)
            file = os.path.join(output_pkl_dir, name)
        else:
            name = "".join((string_from_kwargs(self.classifier.get_tag(), kwargs), name))
            file = os.path.join(self.settings.metadata_folder, name)

        probabilities_df.to_pickle(file)
        self.output_file.append(file)
        self.probabilities_pkl_file.append(file)


    def run(self, embeddings_file, output_dir, use_gt_for_training, kwargs):
        X_train, y_train, X, num_classes = self.__prepare_data(embeddings_file, use_gt_for_training,
                                                               self.classifier.inference_mode())
        probabilities = self.classifier.run(X_train, y_train, X, kwargs, num_classes, output_dir)
        self.probabilities[data_types.ClassProbabilitiesType.name()] = probabilities.tolist()
        return pd.DataFrame.from_dict(self.probabilities)
