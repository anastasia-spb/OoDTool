import unittest
from parameterized import parameterized
import os
import pandas as pd
from oodtool.core import data_types
from oodtool.core.data_projectors.data_projector import DataProjector


class TestEmbedder(unittest.TestCase):
    @parameterized.expand([
        [method_name] for method_name in DataProjector.methods
    ])
    def test_sequence(self, name):
        projector = DataProjector(name)
        oodsession_dir = './test_data/pedestrian_tl/ood_session_test/'
        embeddings_file = os.path.join(oodsession_dir, 'torch_shared-regnet_trafficlights_v12.emb.pkl')
        output_file = projector.project(metadata_folder=oodsession_dir, embeddings_file=embeddings_file)
        print(output_file)

        df = pd.read_pickle(output_file)
        emb_df = pd.read_pickle(embeddings_file)

        self.assertTrue(df[data_types.RelativePathType.name()].equals(emb_df[data_types.RelativePathType.name()]))
        self.assertIn(data_types.ProjectedEmbeddingsType.name(), df)
