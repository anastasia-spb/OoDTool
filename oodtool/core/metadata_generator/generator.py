import os
from oodtool.core.metadata_generator.src import parse_dataset


def generate_metadata(description_json, output_folder, data_root_dir):
    """Walks through dataset directory and stores metadata information about images into <dataset_name>.meta.pkl file.
        Args:
            description_json: json file with dataset description
            output_folder: folder where <dataset_name>.meta.pkl will be stored
            data_root_dir: prefix for relative folders paths
        Returns:
            Absolute path to <dataset_name>.meta.pkl file or None if input data are invalid.
        """
    dataset, dataset_name = parse_dataset(description_json, data_root_dir)
    if dataset is not None:
        file_name = "".join((dataset_name, '.meta.pkl'))
        full_path = os.path.join(output_folder, file_name)
        dataset.to_pickle(full_path)
        return full_path
    return None
