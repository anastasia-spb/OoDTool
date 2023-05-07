import pandas as pd
import numpy as np
import os

from sklearn.metrics import accuracy_score, confusion_matrix

from tool.core.classifier_wrappers import classifier_pipeline
from tool.core.utils import data_helpers

from tool.core import data_types
from tool.core.classifier_wrappers.test import test_data_type, linear_clf_test_data, logistic_regression_test_cases


def get_y_actual(metadata_file: str):
    meta_df = pd.read_pickle(metadata_file)
    labels = meta_df[data_types.LabelsType.name()][0]
    return meta_df.apply(lambda row: labels.index(row[data_types.LabelType.name()]), axis=1).values


def test(test_data: test_data_type.TestData):
    metadata_folder = './tmp'
    if not os.path.exists(metadata_folder):
        os.makedirs(metadata_folder)

    clf = classifier_pipeline.ClassifierPipeline(test_data.classifier_tag)
    _ = clf.classify(test_data.embeddings_pkl, metadata_folder, test_data.use_gt,
                     test_data.pkl_with_probabilities, test_data.weight_decays)
    predictions_df = clf.get_probabilities_df()

    predictions_columns = data_helpers.get_columns_which_start_with(predictions_df,
                                                                    data_types.ClassProbabilitiesType.name())

    y_actual = get_y_actual(test_data.embeddings_pkl)

    for column_name in predictions_columns:
        print(column_name)
        predictions = predictions_df[column_name].tolist()
        predicted_classes = np.argmax(predictions, axis=1)
        score = accuracy_score(y_actual, predicted_classes)
        print("Test Accuracy : {}".format(score))
        print("\nConfusion Matrix : ")
        print(confusion_matrix(y_actual, predicted_classes))


if __name__ == "__main__":
    linear_clf_test_data.linear_classifier_test(test)
    # logistic_regression_test_cases.lr_classifier_test(test)
