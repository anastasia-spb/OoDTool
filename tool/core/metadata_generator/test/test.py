import pandas as pd

from tool.core.metadata_generator.generator import generate_metadata


def test():
    path = '../../../../example_data/datasets/DogsCats/description.json'
    output_folder = '.'

    meta_file_path = generate_metadata(path, output_folder)

    data_df = pd.read_pickle(meta_file_path)
    print(data_df.to_string(index=False))


if __name__ == "__main__":
    test()
