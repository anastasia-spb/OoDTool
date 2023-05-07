from typing import Callable

from tool.core.classifier_wrappers.test import test_data_type
from tool.core.classifier_wrappers.classifiers.linear_classifier.linear_classifier_wrapper import \
    LinearClassifierWrapper

TEST_TRAIN_USE_GT = test_data_type.TestData(LinearClassifierWrapper.tag,
                                            'test_data/ResNetDroneBird230424_191030.emb.pkl',
                                            None,
                                            [0.000001, 0.004, 1.0],
                                            use_gt=True,
                                            checkpoint=None)

TEST_TRAIN_WD_LARGE = test_data_type.TestData(LinearClassifierWrapper.tag,
                                              'test_data/TimmResnetWrapperImageNetVegetables230427_145051.emb.pkl',
                                              'test_data/TimmResnetWrapperImageNetVegetables230427_145051.emb.pkl',
                                              [1.0],
                                              use_gt=False,
                                              checkpoint=None)

TEST_TRAIN_MULTIPLE_FEATURES = test_data_type.TestData(LinearClassifierWrapper.tag,
                                                       'test_data/TimmResnetWrapperImageNetVegetables230427_145051.emb.pkl',
                                                       'test_data/TimmResnetWrapperImageNetVegetables230427_145051.emb.pkl',
                                                       [1.0],
                                                       use_gt=False,
                                                       checkpoint=None)

TEST_TRAIN_WD_LARGE_GT = test_data_type.TestData(LinearClassifierWrapper.tag,
                                                 'test_data/TimmResnetWrapperImageNetVegetables230427_145051.emb.pkl',
                                                 None,
                                                 [1.0],
                                                 use_gt=True,
                                                 checkpoint=None)

# weight_decay and use_gt values are not important when checkpoint file is valid
TEST_EVAL = test_data_type.TestData(LinearClassifierWrapper.tag,
                                    'test_data/ResNetDroneBird230424_191030.emb.pkl',
                                    None,
                                    [0.0],
                                    use_gt=True,
                                    checkpoint='test_data/epoch=39-step=160.ckpt')


def linear_classifier_test(test_func: Callable[[test_data_type.TestData], None]):
    testdata = [
        TEST_TRAIN_USE_GT,
        TEST_TRAIN_WD_LARGE,
        TEST_EVAL,
        TEST_TRAIN_MULTIPLE_FEATURES,
        TEST_TRAIN_WD_LARGE_GT,
    ]

    for test_data in testdata:
        print("Testing {0}".format(test_data.weight_decays))
        test_func(test_data)
        print("===========================================")
