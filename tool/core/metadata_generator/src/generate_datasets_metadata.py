import json
import os
import pandas as pd
from tool.core import data_types


def parse_dataset_description(description_json: str, dataset_folder, dataset_labels):
    with open(description_json, 'r') as json_file:
        dataset_description = json.load(json_file)
    dataset_data = dataset_description["Dataset"]
    root, _ = os.path.split(description_json)
    samples = []
    for folder in dataset_data["folders"]:
        images_path = os.path.join(root, folder["path"])
        for root_dir, _, filenames in os.walk(images_path):
            for name in filenames:
                filename, file_extension = os.path.splitext(name)
                if file_extension not in {".jpg", ".png", ".jpeg"}:
                    continue
                rel_path = os.path.relpath(os.path.join(root_dir, name), images_path)
                samples.append(data_types.MetadataSampleType(
                    data_types.RelativePathType(os.path.join(dataset_folder, folder["path"], rel_path)),
                    data_types.LabelsType(dataset_labels),
                    data_types.TestSampleFlagType(folder["test"]),
                    data_types.LabelType(folder["label"])))
    return pd.DataFrame.from_records([s.to_dict() for s in samples])


def parse_dataset(root, dataset):
    samples = []
    for description in dataset["descriptions"]:
        path_file = os.path.split(description)
        samples.append(parse_dataset_description(os.path.join(root, description), path_file[0], dataset["labels"]))
    return pd.concat(samples, ignore_index=True, sort=False)


def parse_datasets(datasets_description, dataset_name):
    with open(datasets_description, 'r') as json_file:
        datasets_data = json.load(json_file)

    root, _ = os.path.split(datasets_description)
    for dataset in datasets_data["Datasets"]:
        if dataset["name"] == dataset_name:
            return parse_dataset(root, dataset)
    return None




