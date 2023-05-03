import os
from dataclasses import dataclass
from typing import List
import pandas as pd
import numpy as np
import math

from sklearn.metrics import accuracy_score, confusion_matrix

from tool.core.model_wrappers.models.alexnet.alexnet_wrapper import AlexNetWrapper
from tool.core.model_wrappers.models.timm_resnet.timm_resnet_wrapper import TimmResnetWrapper
from tool.core.model_wrappers.models.regnet.regnet_wrapper import RegnetWrapper
from tool.core.model_wrappers.embedder_pipeline import EmbedderPipeline
from tool.core import data_types


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

    pipeline.predict(callback, requires_grad=True, metadata_folder='', dataset_root_dir=test_data.data_dir)
    model_output_df = pipeline.get_model_output()

    y_preds = model_output_df[data_types.ClassProbabilitiesType.name()].tolist()
    y_preds = np.array(y_preds, dtype=np.dtype('float64'))

    y_actual = get_y_actual(test_data.metadata_file)
    predicted_classes = np.argmax(y_preds, axis=1)

    database_labels = model_output_df[data_types.LabelsType.name()][0]

    print("Database labels: {}".format(database_labels))
    score = accuracy_score(y_actual, predicted_classes)
    print("Test Accuracy : {}".format(score))
    print("\nConfusion Matrix : ")
    print(confusion_matrix(y_actual, predicted_classes))

    assert math.isclose(score, test_data.expected_accuracy, abs_tol=0.1)

    if store_embeddings:
        model_output_df.to_pickle(''.join((test_data.embedder_name, '_', 'test_pipeline.emb.pkl')))


BUBBLE_BALLOONS_DATASET_ROOT = '../../../../example_data/datasets/BalloonsBubbles'
WORKING_DIR = '../../../../example_data/tool_working_dir'

ALEXNET_TEST_PARAMS = TestData(BUBBLE_BALLOONS_DATASET_ROOT,
                               os.path.join(WORKING_DIR, 'BalloonsBubbles/BalloonsBubbles.meta.pkl'),
                               AlexNetWrapper.get_name(),
                               {"weights_path": '../../../../pretrained_weights/embedders/AlexNet_BalloonsBubbles.pth',
                                "model_labels": "[bubble, balloon]"},
                               0.63)

TIMM_DENSNET_TEST_PARAMS = TestData(BUBBLE_BALLOONS_DATASET_ROOT,
                                    os.path.join(WORKING_DIR, 'BalloonsBubbles/BalloonsBubbles.meta.pkl'),
                                    TimmResnetWrapper.get_name(),
                                    {"model_checkpoint": 'densenet121'},
                                    0.68)

TIMM_RESNET_TEST_PARAMS = TestData(BUBBLE_BALLOONS_DATASET_ROOT,
                                   os.path.join(WORKING_DIR, 'BalloonsBubbles/BalloonsBubbles.meta.pkl'),
                                   TimmResnetWrapper.get_name(),
                                   {"model_checkpoint": 'resnet34'},
                                   0.68)

DOGS_CATS_DATASET_ROOT = '../../../../example_data/datasets/DogsCats'

TIMM_RESNET_ON_UNKNOWN_CLASSES_TEST_PARAMS = TestData(DOGS_CATS_DATASET_ROOT,
                                                      os.path.join(WORKING_DIR, 'DogsCats/DogsCats.meta.pkl'),
                                                      TimmResnetWrapper.get_name(),
                                                      {"model_checkpoint": 'resnet34'},
                                                      0.46)


def test_pipeline(store_embeddings: bool):
    testdata = [
        # ALEXNET_TEST_PARAMS,
        # TIMM_DENSNET_TEST_PARAMS,
        # TIMM_RESNET_TEST_PARAMS,
        TIMM_RESNET_ON_UNKNOWN_CLASSES_TEST_PARAMS,
    ]

    for test_data in testdata:
        print("Testing {0}".format(test_data.embedder_name))
        test(test_data, store_embeddings)
        print("===========================================")


if __name__ == "__main__":
    test_pipeline(store_embeddings=False)
