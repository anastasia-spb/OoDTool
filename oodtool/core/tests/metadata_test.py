import os.path

import pandas as pd
from oodtool.core.metadata_generator.generator import generate_metadata


def test():
    dataset_root_path = './test_data/pedestrian_tl'
    metadata_folder = './test_data/pedestrian_tl/ood_session_test/'
    json_file = os.path.join(metadata_folder, "description.desc.json")

    meta_file_path = generate_metadata(json_file, output_folder=metadata_folder, data_root_dir=dataset_root_path)

    data_df = pd.read_pickle(meta_file_path)
    print(data_df.to_string(index=False))
