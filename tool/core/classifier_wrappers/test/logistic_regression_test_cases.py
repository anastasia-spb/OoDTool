from typing import Callable

from tool.core.classifier_wrappers.test import test_data_type

TEST_DATA = test_data_type.TestData("LogisticRegression_liblinear",
                                    'test_data/ResNetDroneBird230424_191030.emb.pkl',
                                    None,
                                    [0.00001, 10000.0, 1.0],
                                    use_gt=True,
                                    checkpoint=None)

TEST_LR_NEWTON_CG = test_data_type.TestData("LogisticRegression_newton-cg",
                                            'test_data/ResNetDroneBird230424_191030.emb.pkl',
                                            None,
                                            [0.00001, 10000.0, 1.0],
                                            use_gt=True,
                                            checkpoint=None)

TEST_LR_lbfgs = test_data_type.TestData("LogisticRegression_lbfgs",
                                        'test_data/ResNetDroneBird230424_191030.emb.pkl',
                                        None,
                                        [0.00001, 10000.0, 1.0],
                                        use_gt=True,
                                        checkpoint=None)

TEST_PRETRAINED = test_data_type.TestData("LogisticRegression_lbfgs",
                                          'test_data/ResNetDroneBird230424_191030.emb.pkl',
                                          None,
                                          [0],
                                          use_gt=False,
                                          checkpoint="/home/nastya/Desktop/OoDTool/tool/core/classifier_wrappers/test/test_data/230507_200528.863.joblib.pkl")

TEST_SGD = test_data_type.TestData("SGDClassifier",
                                   '/home/nastya/Desktop/OoDTool/tool/core/classifier_wrappers/test/test_data/TimmResnetWrapper_densenet121_DroneBird_1024_230507_112742.580.sampled.emb.pkl',
                                   '/home/nastya/Desktop/OoDTool/tool/core/classifier_wrappers/test/test_data/TimmResnetWrapper_densenet121_DroneBird_1024_230507_112742.580.sampled.emb.pkl',
                                   [0.1],
                                   use_gt=False,
                                   checkpoint="")


def lr_classifier_test(test_func: Callable[[test_data_type.TestData], None]):
    testdata = [
        TEST_DATA,
        TEST_PRETRAINED,
        TEST_SGD,
        TEST_LR_NEWTON_CG,
        TEST_LR_lbfgs,
    ]

    for test_data in testdata:
        print("Testing {0}".format(test_data.weight_decays))
        test_func(test_data)
        print("===========================================")
