import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.metrics import accuracy_score, confusion_matrix

from tool.classifiers.classifier import Classifier
from tool.classifiers.linear_classifier.linear_classifier_wrapper import LinearClassifierWrapper

from tool import data_types


@dataclass
class TestData:
    embeddings_pkl: str
    kwargs: dict
    use_gt: bool


def get_y_actual(metadata_file: str):
    meta_df = pd.read_pickle(metadata_file)
    labels = meta_df[data_types.LabelsType.name()][0]
    return meta_df.apply(lambda row: labels.index(row[data_types.LabelType.name()]), axis=1).values


def test(test_data: TestData):
    metadata_folder = './'
    clf = Classifier(LinearClassifierWrapper.tag)
    predictions_df = clf.run(test_data.embeddings_pkl, metadata_folder, test_data.use_gt, test_data.kwargs)
    predictions = predictions_df[data_types.ClassProbabilitiesType.name()].tolist()
    predicted_classes = np.argmax(predictions, axis=1)

    y_actual = get_y_actual(test_data.embeddings_pkl)

    score = accuracy_score(y_actual, predicted_classes)
    print("Test Accuracy : {}".format(score))
    print("\nConfusion Matrix : ")
    print(confusion_matrix(y_actual, predicted_classes))


TEST_TRAIN_WD_SMALL = TestData('test_data/ResNetDroneBird230424_191030.emb.pkl',
                               {"weight_decay": '0.000001', "checkpoint": ''},
                               use_gt=False)

TEST_TRAIN_WD_MIDDLE = TestData('test_data/ResNetDroneBird230424_191030.emb.pkl',
                                {"weight_decay": '0.004', "checkpoint": ''},
                                use_gt=False)

TEST_TRAIN_WD_LARGE = TestData('test_data/TimmResnetWrapperImageNetVegetables230427_145051.emb.pkl',
                               {"weight_decay": '1.0', "checkpoint": ''},
                               use_gt=False)

TEST_TRAIN_WD_LARGE_GT = TestData('test_data/TimmResnetWrapperImageNetVegetables230427_145051.emb.pkl',
                                  {"weight_decay": '1.0', "checkpoint": ''},
                                  use_gt=True)

# weight_decay and use_gt values are not important when checkpoint file is valid
TEST_EVAL = TestData('test_data/ResNetDroneBird230424_191030.emb.pkl',
                     {"weight_decay": '0.0', "checkpoint": 'test_data/epoch=39-step=160.ckpt'},
                     use_gt=False)


def test_pipeline():
    testdata = [
        # TEST_TRAIN_WD_SMALL,
        # TEST_TRAIN_WD_MIDDLE,
        TEST_TRAIN_WD_LARGE,
        # TEST_EVAL,
        TEST_TRAIN_WD_LARGE_GT,
    ]

    for test_data in testdata:
        print("Testing {0}".format(test_data.kwargs))
        test(test_data)
        print("===========================================")


if __name__ == "__main__":
    test_pipeline()
