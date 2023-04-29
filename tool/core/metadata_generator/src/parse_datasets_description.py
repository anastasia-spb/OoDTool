import json
import os


def get_datasets_names(datasets_description):
    with open(datasets_description, 'r') as json_file:
        datasets_data = json.load(json_file)

    root, _ = os.path.split(datasets_description)
    datasets = []
    for dataset in datasets_data["Datasets"]:
        datasets.append(dataset["name"])
    return datasets


def test():
    path = '/home/nastya/Desktop/NIR/project/NIR/data/datasets/datasets.json'
    datasets = get_datasets_names(path)
    print(datasets)


if __name__ == "__main__":
    test()
