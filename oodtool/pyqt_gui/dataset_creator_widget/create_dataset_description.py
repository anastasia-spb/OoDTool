import os.path
from typing import List, Optional
import json


def get_folders_desc(folders, data_dir: str, test: bool):
    folders_desc = []
    labels = set()
    for folder in folders:
        _, label = os.path.split(folder)
        folder_desc = {"path": os.path.relpath(folder, data_dir),
                       "label": label,
                       "test": test}
        folders_desc.append(folder_desc)
        labels.add(label)
    return folders_desc, labels


def create_dataset_description(train_folders: List[str], test_folders: List[str], output_dir: str,
                               data_dir: str, name: Optional[str] = None):
    if name is None:
        name = "DatasetDescription"

    dataset = dict()
    dataset["name"] = name

    train_desc, train_labels = get_folders_desc(train_folders, data_dir, test=False)
    test_desc, test_labels = get_folders_desc(test_folders, data_dir, test=True)

    dataset["folders"] = train_desc + test_desc
    dataset["labels"] = list(train_labels | test_labels)

    description = dict()
    description["Dataset"] = dataset
    description_file = os.path.join(output_dir, 'description.desc.json')
    with open(description_file, 'w', encoding='utf8') as json_file:
        json.dump(description, json_file, ensure_ascii=False)

    return description_file


def find_all_subfolders(input_folder: str):
    folders = []

    for root_dir, subdirectories, filenames in os.walk(input_folder):
        if len(subdirectories) == 0:
            folders.append(root_dir)

    return folders
