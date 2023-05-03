import os
import json
from datetime import datetime
from typing import List


def store_pipeline_config(embeddings_file: str, probabilities_file: str, tag: str, weights: List[str],
                          metadata_folder: str):
    config = {"training_embeddings_file": embeddings_file,
              "training_probabilities_file": probabilities_file,
              "tag": tag,
              "classifiers": []}

    for weight in weights:
        config["classifiers"].append({"weights": weight})

    timestamp_str = datetime.utcnow().strftime("%y%m%d_%H%M%S.%f")[:-3]
    config_file = os.path.join(metadata_folder, "".join((timestamp_str, ".config.json")))
    with open(config_file, 'w', encoding='utf-8') as outfile:
        json.dump(config, outfile, ensure_ascii=False, indent=4)

    return config_file


class ClfConfig:
    def __init__(self, json_object: dict):
        self.tag = json_object["tag"]
        classifiers = json_object["classifiers"]
        self.weights = [w["weights"] for w in classifiers]

    def get_tag(self):
        return self.tag

    def get_weights(self):
        return self.weights


def unite_configurations(root_dir: str, config_files: List[str]) -> str:
    if len(config_files) < 1:
        return ''

    ood_configs = []
    for config_path in config_files:
        with open(config_path, 'r') as openfile:
            json_object = json.load(openfile)
        ood_configs.append({"embeddings_file": '', "clf_config": json_object})
        os.remove(config_path)

    ood_config = {"OoDConfig": ood_configs}

    output_file = os.path.join(root_dir, "".join(("ood_pipeline", ".config.json")))
    with open(output_file, 'w', encoding='utf-8') as outfile:
        json.dump(ood_config, outfile, ensure_ascii=False, indent=4)

    return output_file
