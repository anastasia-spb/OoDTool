import os
from oodtool.core.distance_wrapper.calculate_distances import DistanceCalculator
from oodtool.pyqt_gui.data_loader.loader import DataLoader
from oodtool.pyqt_gui.qt_utils.qt_types import ImageInfo


def test_tl_distance_calculation():
    ood_session_folder = './test_data/pedestrian_tl/ood_session_test/'
    embeddings_file = os.path.join(ood_session_folder, 'torch_shared-regnet_trafficlights_v12.emb.pkl')
    distance_calculator = DistanceCalculator(embeddings_file, output_folder=ood_session_folder, num_neighbours=5)
    output_file = distance_calculator.get_pdist()
    print(output_file)
    assert os.path.isfile(output_file)


def test_dogs_cats_distance_calculation():
    ood_session_folder = "../../../example_data/DogsCats/oodsession_0/"
    embeddings_file = os.path.join(ood_session_folder, 'torch_embedder_towheeresnet50_v2.emb.pkl')
    distance_calculator = DistanceCalculator(embeddings_file, output_folder=ood_session_folder, num_neighbours=3)
    output_file = distance_calculator.get_pdist()
    print(output_file)
    assert os.path.isfile(output_file)


def test_tl():
    dataset_root_path = './test_data/pedestrian_tl'
    metadata_folder = './test_data/pedestrian_tl/ood_session_test/'

    data_loader = DataLoader()
    data_loader.load_data(dataset_root_path, metadata_folder)

    assert data_loader.status_loaded

    info = ImageInfo(
        relative_path='images/test/pedestrian_tl_10_forward/132.4_get.546.229.left.000063.x833_y247_w11_h20.png',
        ood_score=0.0,
        labels=[],
        dataset_root_dir=dataset_root_path,
        metadata_dir=metadata_folder)

    images_meta = data_loader.get_k_neighbours(info)

    expected_neighbours = ['images/test/pedestrian_tl_10_forward/132.4_get.546.229.left.000063.x833_y247_w11_h20.png',
                           'images/test/pedestrian_tl_10_forward/221.9_get.410.023.left.000077.x870_y153_w18_h33.png',
                           'images/test/pedestrian_tl_10_forward/212.3_get.410.023.left.000059.x616_y187_w9_h17.png',
                           'images/test/pedestrian_tl_10_forward/236.0_get.410.093.left.000032.x801_y176_w12_h26.png',
                           'images/test/pedestrian_tl_10_forward/184.8_get.546.023.left.000051.x715_y256_w7_h17.png']

    assert len(images_meta) == len(expected_neighbours)

    for neighbour_meta, expected in zip(images_meta, expected_neighbours):
        assert neighbour_meta.relative_path == expected


def test_dogs_cats():
    dataset_root_path = "../../../example_data/DogsCats"
    metadata_folder = "../../../example_data/DogsCats/oodsession_0/"

    data_loader = DataLoader()
    data_loader.load_data(dataset_root_path, metadata_folder)

    assert data_loader.status_loaded

    info = ImageInfo(
        relative_path='test/ood_samples/img_0_68.jpg',
        ood_score=0.0,
        labels=[],
        dataset_root_dir=dataset_root_path,
        metadata_dir=metadata_folder)

    images_meta = data_loader.get_k_neighbours(info)

    expected_neighbours = ['test/ood_samples/img_0_68.jpg',
                           'test/ood_samples/img_0_78.jpg',
                           'train/cat/cat.9.jpg']

    assert len(images_meta) == len(expected_neighbours)

    for neighbour_meta, expected in zip(images_meta, expected_neighbours):
        assert neighbour_meta.relative_path == expected
