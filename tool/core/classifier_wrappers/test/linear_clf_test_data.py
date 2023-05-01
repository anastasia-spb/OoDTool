from typing import Callable

from tool.core.classifier_wrappers.test import test_data_type
from tool.core.classifier_wrappers.classifiers.linear_classifier.linear_classifier_wrapper import \
    LinearClassifierWrapper

TEST_TRAIN_USE_GT = test_data_type.TestData(LinearClassifierWrapper.tag,
                                            ['test_data/ResNetDroneBird230424_191030.emb.pkl'],
                                            None,
                                            [{"weight_decay": '0.000001', "checkpoint": ''},
                                             {"weight_decay": '0.004', "checkpoint": ''},
                                             {"weight_decay": '1.0', "checkpoint": ''}],
                                            use_gt=True)

TEST_TRAIN_WD_LARGE = test_data_type.TestData(LinearClassifierWrapper.tag,
                                              ['test_data/TimmResnetWrapperImageNetVegetables230427_145051.emb.pkl'],
                                              'test_data/TimmResnetWrapperImageNetVegetables230427_145051.emb.pkl',
                                              [{"weight_decay": '1.0', "checkpoint": ''}],
                                              use_gt=False)

TEST_TRAIN_MULTIPLE_FEATURES = test_data_type.TestData(LinearClassifierWrapper.tag,
                                                       ['test_data/TimmResnetWrapperImageNetVegetables230427_145051.emb.pkl',
                                                        'test_data/TimmResnetWrapperImageNetVegetables230427_145051.emb.pkl'],
                                                       'test_data/TimmResnetWrapperImageNetVegetables230427_145051.emb.pkl',
                                                       [{"weight_decay": '1.0', "checkpoint": ''}],
                                                       use_gt=False)

TEST_TRAIN_WD_LARGE_GT = test_data_type.TestData(LinearClassifierWrapper.tag,
                                                 ['test_data/TimmResnetWrapperImageNetVegetables230427_145051.emb.pkl'],
                                                 None,
                                                 [{"weight_decay": '1.0', "checkpoint": ''}],
                                                 use_gt=True)

# weight_decay and use_gt values are not important when checkpoint file is valid
TEST_EVAL = test_data_type.TestData(LinearClassifierWrapper.tag,
                                    ['test_data/ResNetDroneBird230424_191030.emb.pkl'],
                                    None,
                                    [{"weight_decay": '0.0', "checkpoint": 'test_data/epoch=39-step=160.ckpt'}],
                                    use_gt=True)


def linear_classifier_test(test_func: Callable[[test_data_type.TestData], None]):
    testdata = [
        TEST_TRAIN_USE_GT,
        TEST_TRAIN_WD_LARGE,
        TEST_EVAL,
        TEST_TRAIN_MULTIPLE_FEATURES,
        TEST_TRAIN_WD_LARGE_GT,
    ]

    for test_data in testdata:
        print("Testing {0}".format(test_data.kwargs))
        test_func(test_data)
        print("===========================================")
