import unittest
from parameterized import parameterized
import os
import pandas as pd
from oodtool.core import data_types
from oodtool.core.czebra_adapter import CZebraAdapter, determine_dataset_usecase, usecase
from oodtool.core.ood_score import features_selector
from oodtool.core.czebra_adapter.czebra_saliency_map import get_saliency_map


class TestEmbedder(unittest.TestCase):
    @parameterized.expand([
        # ["with_classifier", "./test_data/pedestrian_tl/ood_session_test", "DatasetDescription.meta.pkl",
        #  "./test_data/pedestrian_tl", features_selector.OOD_ENTROPY, usecase.TRAFFICLIGHTS, 2],
        ["only_embedder", "../../../example_data/DogsCats/oodsession_0", "DatasetDescription.meta.pkl",
         "../../../example_data/DogsCats", features_selector.OOD_ENTROPY_SWIN, usecase.OTHER, 1],
    ])
    def test_sequence(self, name, ood_session_dir, metadata_file, data_dir, tested_ood_method: str, expected_usecase,
                      expected_num_of_features):

        metadata_file_path = os.path.join(ood_session_dir, metadata_file)
        metadata_df = pd.read_pickle(metadata_file_path)
        labels = metadata_df[data_types.LabelsType.name()][0]

        embedder_wrapper = CZebraAdapter(metadata_df, data_dir, output_dir=ood_session_dir)

        result_files = []

        usecase_name = determine_dataset_usecase(labels)
        self.assertEqual(usecase_name, expected_usecase)

        embedders_ids = features_selector.OOD_METHOD_FEATURES[tested_ood_method][usecase_name]
        for emb_id in embedders_ids:
            output_file, output_probabilities_file = embedder_wrapper.predict(model_id=emb_id)
            result_files.append((output_file, output_probabilities_file))
        self.assertEqual(len(result_files), expected_num_of_features)

        for files in result_files:
            self.assertEqual(len(files), 2)

        print(result_files)

        self.__check_columns(result_files, metadata_df)

    def __check_columns(self, result_files, metadata_df):
        for emb_file, clf_file in result_files:
            emb_df = pd.read_pickle(emb_file)
            self.assertIn(data_types.EmbeddingsType.name(), emb_df)
            self.assertTrue(emb_df[data_types.RelativePathType.name()].equals(
                metadata_df[data_types.RelativePathType.name()]))
            if clf_file is not None:
                clf_df = pd.read_pickle(clf_file)
                self.assertTrue(clf_df[data_types.RelativePathType.name()].equals(
                    metadata_df[data_types.RelativePathType.name()]))
                self.assertIn(data_types.PredictedLabelsType.name(), clf_df)
                self.assertIn(data_types.PredictedProbabilitiesType.name(), clf_df)
                self.assertIn(data_types.LabelsForPredictedProbabilitiesType.name(), clf_df)
                self.assertIn(data_types.ClassProbabilitiesType.name(), clf_df)


def test_saliency_map():
    maps = get_saliency_map(
        './test_data/pedestrian_tl/images/test/pedestrian_tl_10_forward/132.4_get.546.229.left.000063.x833_y247_w11_h20.png',
        'torch_shared-regnet_trafficlights_v11')
    assert len(maps) == 2
    assert maps[0][1] == 'pedestrian_tl_10_forward'
    assert maps[1][1] == 'ROTATED'


