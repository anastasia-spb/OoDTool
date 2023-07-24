import os.path
from parameterized import parameterized
import unittest
import pandas as pd
import logging
from datetime import datetime

from oodtool.core.ood_score.metrics import run_metrics
from oodtool.core.ood_score import ood_score_to_df
from oodtool.core.ood_score import score_by_ood
from oodtool.core.ood_score import features_selector

TL_METADATA_FOLDER = "./test_data/pedestrian_tl/ood_session_test/"
TL_METADATA_FILE = os.path.join(TL_METADATA_FOLDER, "DatasetDescription.meta.pkl")
TL_OOD_FOLDERS = ['images/test/trash']

DOGS_CATS_METADATA_FOLDER = "../../../example_data/DogsCats/oodsession_0"
DOGS_CATS_METADATA_FILE = os.path.join(DOGS_CATS_METADATA_FOLDER, "DatasetDescription.meta.pkl")
DOGS_CATS_OOD_FOLDERS = ['test/cat/cats_on_leash', 'test/ood_samples']

logger = logging.getLogger()
timestamp_str = datetime.utcnow().strftime("%y%m%d_%H%M%S.%f")[:-3]
fhandler = logging.FileHandler(filename="".join(('ood_metrics_', timestamp_str, '.log')), mode='a')
formatter = logging.Formatter('%(asctime)s - %(message)s')
fhandler.setFormatter(formatter)
logger.addHandler(fhandler)
logger.setLevel(logging.DEBUG)


class TestPipeline(unittest.TestCase):
    @parameterized.expand([
        ["DogsCats Entropy", DOGS_CATS_METADATA_FILE,
         [os.path.join(DOGS_CATS_METADATA_FOLDER, "torch_embedder_towheeresnet50_v2.emb.pkl")],
         [None], features_selector.OOD_ENTROPY, DOGS_CATS_OOD_FOLDERS],
        ["DogsCats KNN Dist", DOGS_CATS_METADATA_FILE,
         [os.path.join(DOGS_CATS_METADATA_FOLDER, "torch_embedder_towheeresnet50_v2.emb.pkl")],
         [None], features_selector.OOD_KNN_DIST, DOGS_CATS_OOD_FOLDERS],
        ["TrafficLights Entropy", TL_METADATA_FILE,
         [os.path.join(TL_METADATA_FOLDER, "torch_shared-regnet_trafficlights_v12.emb.pkl"),
          os.path.join(TL_METADATA_FOLDER, "torch_embedder_towheeresnet50_v2.emb.pkl")],
         [os.path.join(TL_METADATA_FOLDER, "torch_shared-regnet_trafficlights_v12.clf.pkl"), None],
         features_selector.OOD_ENTROPY, TL_OOD_FOLDERS],
        ["TrafficLights KNN Dist Trained", TL_METADATA_FILE,
         [os.path.join(TL_METADATA_FOLDER, "torch_shared-regnet_trafficlights_v12.emb.pkl")],
         [os.path.join(TL_METADATA_FOLDER, "torch_shared-regnet_trafficlights_v12.clf.pkl")],
         features_selector.OOD_KNN_DIST, TL_OOD_FOLDERS],
        ["TrafficLights KNN Dist Generic", TL_METADATA_FILE,
         [os.path.join(TL_METADATA_FOLDER, "torch_embedder_towheeresnet50_v2.emb.pkl")],
         [os.path.join(TL_METADATA_FOLDER, "torch_shared-regnet_trafficlights_v12.clf.pkl")],
         features_selector.OOD_KNN_DIST, TL_OOD_FOLDERS],
        ["TrafficLights Confident Learning", TL_METADATA_FILE,
         [],
         [os.path.join(TL_METADATA_FOLDER, "torch_shared-regnet_trafficlights_v12.clf.pkl")],
         features_selector.OOD_CONFIDENT_LEARNING, TL_OOD_FOLDERS],
    ])
    def test_sequence(self, name, metadata_file, embeddings_files, probabilities_files, ood_method, ood_folders):
        metadata_df = pd.read_pickle(metadata_file)
        score = score_by_ood(ood_method, metadata_df, embeddings_files=embeddings_files,
                             probabilities_files=probabilities_files, head_idx=0)

        assert score is not None

        ood_df = ood_score_to_df(score, metadata_df)
        logger.info("=" * 80)
        logger.info(name)
        run_metrics(ood_df, metadata_df, ood_folders, logger.info, probabilities_files[0])


# Select folder with OoD files
METADATA_FOLDER = '/home/vlasova/datasets/ood_datasets/Letters/oodsession_1'
METADATA_FILE = os.path.join(METADATA_FOLDER, "DatasetDescription.meta.pkl")
OOD_FOLDERS = ['test/others', 'test/MNIST', 'test/a/diff_background', 'test/b/diff_background',
               'test/c/diff_background', 'test/d/diff_background', 'test/e/diff_background', 'test/f/diff_background',
               'test/g/diff_background', 'test/h/diff_background', 'test/i/diff_background',
               'test/j/diff_background']


class GetMetrics(unittest.TestCase):
    @parameterized.expand([
        ["OoD Entropy", METADATA_FILE,
         [os.path.join(METADATA_FOLDER, "torch_embedder_towheeresnet50_v2.emb.pkl")],
         [None], features_selector.OOD_ENTROPY, OOD_FOLDERS],
        ["OoD KNN Dist", METADATA_FILE,
         [os.path.join(METADATA_FOLDER, "torch_embedder_towheeresnet50_v2.emb.pkl")],
         [None], features_selector.OOD_KNN_DIST, OOD_FOLDERS],
    ])
    def test_sequence(self, name, metadata_file, embeddings_files, probabilities_files, ood_method, ood_folders):
        metadata_df = pd.read_pickle(metadata_file)
        score = score_by_ood(ood_method, metadata_df, embeddings_files=embeddings_files,
                             probabilities_files=probabilities_files, head_idx=0)

        assert score is not None

        ood_df = ood_score_to_df(score, metadata_df)
        logger.info("=" * 80)
        logger.info(name)
        run_metrics(ood_df, metadata_df, ood_folders, logger.info, probabilities_files[0])
