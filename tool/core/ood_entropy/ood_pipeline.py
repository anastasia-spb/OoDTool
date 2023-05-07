import json
import os.path
import pandas as pd
import matplotlib.pyplot as plt

from tool.core.classifier_wrappers.classifier_pipeline import ClassifierPipeline
from tool.core.ood_entropy.ood_score import OoDScore
from tool.core import data_types


def run_ood_pipeline(output_folder: str, config_file: str) -> str:
    if not os.path.isfile(config_file):
        return ''

    with open(config_file, 'r') as openfile:
        json_object = json.load(openfile)

    ood_configs = json_object["OoDConfig"]

    probabilities = []
    for config in ood_configs:
        embeddings_file = config["embeddings_file"]
        clf_config = config["clf_config"]
        classifier_pipeline = ClassifierPipeline(clf_config["tag"])
        output = classifier_pipeline.classify_from_config(embeddings_file, clf_config, output_folder)
        probabilities.append(output)

    pipeline = OoDScore()
    return pipeline.run(probabilities, output_folder)


def test_ood_pipeline():
    output_folder = "./"
    config_file = './test/test_data/ood_pipeline.config.json'
    resulting_file = run_ood_pipeline(output_folder, config_file)
    ood_df = pd.read_pickle(resulting_file)
    plt.hist(ood_df[data_types.types.OoDScoreType.name()], density=True, bins=30)
    plt.show()


if __name__ == "__main__":
    test_ood_pipeline()
