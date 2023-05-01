import pandas as pd

from tool.core.metadata_generator.generator import generate_metadata, get_datasets_names


def test():
    path = '../../../../example_data/datasets/datasets.json'
    output_folder = '.'

    dataset_name = 'DogsCats'
    meta_file_path = generate_metadata(path, dataset_name, output_folder)

    data_df = pd.read_pickle(meta_file_path)
    print(data_df.to_string(index=False))

    datasets = get_datasets_names(path)
    print(datasets)


if __name__ == "__main__":
    test()
