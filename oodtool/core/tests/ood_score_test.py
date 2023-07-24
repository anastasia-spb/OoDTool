import unittest
import os
import numpy as np
import pandas as pd
from parameterized import parameterized
from oodtool.core.ood_score import features_selector
from oodtool.core.ood_score import score_by_ood
from oodtool.core import data_types


class TestOoDMethods(unittest.TestCase):
    metadata_folder = './test_data/pedestrian_tl/ood_session_test/'
    metadata_file = os.path.join(metadata_folder, 'DatasetDescription.meta.pkl')
    embeddings_files = [os.path.join(metadata_folder, 'torch_shared-regnet_trafficlights_v12.emb.pkl'),
                        os.path.join(metadata_folder, 'torch_embedder_towheeresnet50_v2.emb.pkl')]
    probabilities_files = [os.path.join(metadata_folder, 'torch_shared-regnet_trafficlights_v12.clf.pkl'),
                           None]

    @parameterized.expand([
        ["entropy", features_selector.OOD_ENTROPY,
         ['images/test/trash/136.8_get.410.230.left.000100.x596_y96_w28_h18.png',
          'images/test/pedestrian_tl_01_stop/179.7_get.543.132.left.000068.x787_y96_w86_h183.png',
          'images/test/trash/131.5_get.410.153.left.000100.x520_y20_w30_h14.png',
          'images/test/pedestrian_tl_10_forward/253.4_get.542.316.left.000039.x849_y245_w11_h22.png',
          'images/test/trash/129.1_get.410.092.left.000031.x874_y49_w45_h105.png']],
        ["knn_dist", features_selector.OOD_KNN_DIST,
         ['images/test/trash/131.5_get.410.153.left.000100.x520_y20_w30_h14.png',
          'images/test/trash/146.1_get.410.270.left.000068.x556_y178_w12_h17.png',
          'images/test/trash/136.8_get.410.230.left.000100.x596_y96_w28_h18.png',
          'images/test/trash/140.8_get.410.040.left.000038.x733_y98_w42_h50.png',
          'images/test/trash/130.9_get.543.150.left.000063.x322_y260_w7_h17.png']],
        ["confident_learning for head 0", features_selector.OOD_CONFIDENT_LEARNING,
         ['images/test/trash/146.8_get.410.328.left.000110.x6_y128_w30_h67.png',
          'images/test/trash/136.8_get.410.230.left.000100.x596_y96_w28_h18.png',
          'images/test/pedestrian_tl_10_forward/144.6_get.620.178.left.000008.x46_y145_w29_h44.png',
          'images/train/pedestrian_tl_01_stop/0.522989988_487463955931896_y50_x665_w42_h57.png',
          'images/test/pedestrian_tl_01_stop/141.0_get.410.213.left.000115.x47_y218_w10_h14.png'], 0],
        ["confident_learning for head 1", features_selector.OOD_CONFIDENT_LEARNING, None, 1],
    ])
    def test_sequence(self, name, method, expected_top_images, head_idx=0):
        metadata_df = pd.read_pickle(self.metadata_file)
        score = score_by_ood(method, metadata_df, embeddings_files=self.embeddings_files,
                             probabilities_files=self.probabilities_files, head_idx=head_idx)

        if expected_top_images is None:
            assert score is None
        else:
            assert score is not None
            inverted_score = np.subtract(1.0, score)
            sorted_ood_indices = np.argsort(inverted_score, axis=0)[:5]
            top_images = metadata_df.iloc[sorted_ood_indices][data_types.RelativePathType.name()].tolist()
            assert top_images == expected_top_images
