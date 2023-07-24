import json
import os
import pandas as pd
from oodtool.core import data_types


def parse_dataset_description(description_json: str, root: str):
    with open(description_json, 'r') as json_file:
        dataset_description = json.load(json_file)
    dataset_data = dataset_description["Dataset"]
    dataset_labels = dataset_data["labels"]
    samples = []
    for folder in dataset_data["folders"]:
        images_path = os.path.join(root, folder["path"])
        for root_dir, _, filenames in os.walk(images_path):
            for name in filenames:
                filename, file_extension = os.path.splitext(name)
                if file_extension.lower() not in {".jpg", ".png", ".jpeg"}:
                    continue
                rel_path = os.path.relpath(os.path.join(root_dir, name), images_path)
                samples.append(data_types.MetadataSampleType(
                    data_types.RelativePathType(os.path.join(folder["path"], rel_path)),
                    data_types.LabelsType(dataset_labels),
                    data_types.TestSampleFlagType(folder["test"]),
                    data_types.LabelType(folder["label"])))
    return pd.DataFrame.from_records([s.to_dict() for s in samples]), dataset_data["name"]


def parse_dataset(description_json, root: str):
    return parse_dataset_description(description_json, root)




