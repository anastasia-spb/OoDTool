import unittest
from parameterized import parameterized
import os
from oodtool.core.metadata_generator.generator import generate_metadata
from oodtool.core.czebra_adapter import CZebraAdapter, determine_dataset_usecase
from oodtool.pyqt_gui.data_loader.loader import DataLoader
from oodtool.core.ood_score import features_selector
from oodtool.core.ood_score import score_by_ood
from oodtool.core.distance_wrapper.calculate_distances import DistanceCalculator
from oodtool.core.data_projectors.data_projector import DataProjector


class TestPipeline(unittest.TestCase):
    @parameterized.expand([
        ["TrafficLights Entropy Pipeline", "./test_data/pedestrian_tl", "./test_data/pedestrian_tl/ood_session_test/",
         features_selector.OOD_ENTROPY],
        ["DogsCats Entropy Pipeline", "../../../example_data/DogsCats", "../../../example_data/DogsCats/oodsession_0/",
         features_selector.OOD_ENTROPY],
        ["DogsCats KNN DIST Pipeline", "../../../example_data/DogsCats", "../../../example_data/DogsCats/oodsession_0/",
         features_selector.OOD_KNN_DIST],
    ])
    def test_sequence(self, name, dataset_root_path, metadata_folder, ood_method):
        json_file = os.path.join(metadata_folder, "description.desc.json")

        _ = generate_metadata(json_file, output_folder=metadata_folder, data_root_dir=dataset_root_path)

        data_loader = DataLoader()
        data_loader.load_data(dataset_root_path, metadata_folder)

        embedder_wrapper = CZebraAdapter(data_loader.metadata_df, dataset_root_path, output_dir=metadata_folder)
        usecase_name = determine_dataset_usecase(data_loader.get_labels())
        embedders_ids = features_selector.OOD_METHOD_FEATURES[features_selector.OOD_ENTROPY][usecase_name]
        result_files = []
        for emb_id in embedders_ids:
            output_file, output_probabilities_file = embedder_wrapper.predict(model_id=emb_id)
            result_files.append((output_file, output_probabilities_file))

        _ = score_by_ood(ood_method,
                         data_loader.metadata_df, embeddings_files=[emb_file[0] for emb_file in result_files],
                         probabilities_files=[prob_file[1] for prob_file in result_files], head_idx=0)

        selected_embeddings_file = result_files[0][0]
        distance_calculator = DistanceCalculator(selected_embeddings_file, output_folder=metadata_folder,
                                                 num_neighbours=5)
        distance_calculator.get_pdist()

        projector = DataProjector(DataProjector.method_name)
        projector.project(metadata_folder=metadata_folder, embeddings_file=selected_embeddings_file)

        data_loader.load_data(dataset_root_path, metadata_folder)
