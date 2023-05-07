from typing import Callable

from tool.core.classifier_wrappers.test import test_data_type
from tool.core.classifier_wrappers.classifiers.logistic_regression.lr_wrapper import LogisticRegressionWrapper

TEST_DATA = test_data_type.TestData(LogisticRegressionWrapper.tag,
                                    'test_data/ResNetDroneBird230424_191030.emb.pkl',
                                    None,
                                    [0.00001, 10000.0, 1.0],
                                    use_gt=True,
                                    checkpoint=None)


def lr_classifier_test(test_func: Callable[[test_data_type.TestData], None]):
    testdata = [
        TEST_DATA,
    ]

    for test_data in testdata:
        print("Testing {0}".format(test_data.kwargs))
        test_func(test_data)
        print("===========================================")
