from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np
import math

from tool.models.alexnet.alexnet_wrapper import AlexNetWrapper
from tool.models.timm_resnet.timm_resnet_wrapper import TimmResnetWrapper
from tool.models.embedderpipeline import EmbedderPipeline
from sklearn.metrics import accuracy_score, confusion_matrix

from tool import data_types


@dataclass
class TestData:
    data_dir: str
    metadata_file: str
    embedder_name: str
    wrapper_parameters: dict
    expected_accuracy: float


def get_y_actual(metadata_file: str):
    meta_df = pd.read_pickle(metadata_file)
    labels = meta_df[data_types.LabelsType.name()][0]
    return meta_df.apply(lambda row: labels.index(row[data_types.LabelType.name()]), axis=1).values


def test(test_data: TestData, store_embeddings: bool):
    pipeline = EmbedderPipeline(test_data.metadata_file, test_data.data_dir, test_data.embedder_name,
                                use_cuda=True, **test_data.wrapper_parameters)

    def callback(progress_info: List[int]):
        pass

    pipeline.predict(callback)
    model_output_df = pipeline.get_model_output()

    y_preds = model_output_df[data_types.ClassProbabilitiesType.name()].tolist()
    y_preds = np.array(y_preds, dtype=np.dtype('float64'))

    y_actual = get_y_actual(test_data.metadata_file)
    predicted_classes = np.argmax(y_preds, axis=1)

    database_labels = model_output_df[data_types.LabelsType.name()][0]
    model_labels = pipeline.get_model_labels()

    print("Database labels: {}".format(database_labels))
    # print("Model labels:    {}".format(model_labels))
    score = accuracy_score(y_actual, predicted_classes)
    print("Test Accuracy : {}".format(score))
    print("\nConfusion Matrix : ")
    print(confusion_matrix(y_actual, predicted_classes))

    assert math.isclose(score, test_data.expected_accuracy, abs_tol=0.1)

    if store_embeddings:
        model_output_df.to_pickle(''.join((test_data.embedder_name, '_', 'test_pipeline.emb.pkl')))


ALEXNET_TEST_PARAMS = TestData('/home/vlasova/datasets', 'test_data/SummerWinter.meta.pkl', AlexNetWrapper.get_name(),
                               {"weights_path": '/home/vlasova/Desktop/gitlab/oodtool/tool/models/alexnet/alexnet_0.9805023923444977_12-Apr-2023_16-16-06_SummerWinter.pth'},
                               0.84)

TIMM_DENSNET_TEST_PARAMS = TestData('/home/vlasova/datasets', 'test_data/ImageNetVegetables.meta.pkl',
                                    TimmResnetWrapper.get_name(),
                                    {"model_checkpoint": 'densenet121'},
                                    0.89)

TIMM_RESNET_TEST_PARAMS = TestData('/home/vlasova/datasets', 'test_data/ImageNetVegetables.meta.pkl',
                                   TimmResnetWrapper.get_name(),
                                   {"model_checkpoint": 'resnet34'},
                                   0.88)

TIMM_RESNET_ON_UNKNOWN_CLASSES_TEST_PARAMS = TestData('/home/vlasova/datasets', 'test_data/DroneBird.meta.pkl',
                                                      TimmResnetWrapper.get_name(),
                                                      {"model_checkpoint": 'resnet34'},
                                                      0.4)


def test_pipeline(store_embeddings: bool):
    testdata = [
        # REGNET_TEST_PARAMS,
        # ALEXNET_TEST_PARAMS,
        TIMM_DENSNET_TEST_PARAMS,
        # TIMM_RESNET_TEST_PARAMS,
        # TIMM_RESNET_ON_UNKNOWN_CLASSES_TEST_PARAMS,
    ]

    for test_data in testdata:
        print("Testing {0}".format(test_data.embedder_name))
        test(test_data, store_embeddings)
        print("===========================================")

    print("RESULT ARE APPROXIMATE IF DATABASE CLASSES DON'T MATCH WITH MODEL CLASSES")


if __name__ == "__main__":
    test_pipeline(store_embeddings=False)
