import os
import json
from tool.core.metadata_generator.src import parse_datasets


def get_datasets_names(datasets_description):
    with open(datasets_description, 'r') as json_file:
        datasets_data = json.load(json_file)

    root, _ = os.path.split(datasets_description)
    datasets = []
    for dataset in datasets_data["Datasets"]:
        datasets.append(dataset["name"])
    return datasets


def generate_metadata(path, dataset_name, output_folder):
    """Walks through dataset directory and stores metadata information about images into <dataset_name>.meta.pkl file.
        Args:
            path: absolute path to dataset root folder where datasets.json shall be placed
            dataset_name: name of dataset. Shall be present in datasets.json file
            output_folder: folder where <dataset_name>.meta.pkl will be stored
        Returns:
            Absolute path to <dataset_name>.meta.pkl file or None if input data are invalid.
        """
    dataset = parse_datasets(path, dataset_name)
    if dataset is not None:
        file_name = "".join((dataset_name, '.meta.pkl'))
        full_path = os.path.join(output_folder, file_name)
        dataset.to_pickle(full_path)
        return full_path
    return None
