import argparse
import os
import pandas as pd

from tool.core.metadata_generator.generator import generate_metadata


def train_classifiers():
    parser = argparse.ArgumentParser(description='Generate metadata')
    parser.add_argument('-d', '--embeddings', default='', required=True)
    parser.add_argument('-probs', '--probabilities', default='', required=False)
    args = parser.parse_args()

    use_gt_for_training = not os.path.isfile(args.probabilities)

    path = '/home/vlasova/datasets/TrafficLightsDVC/description_pedestrian.json'
    output_folder = '/home/vlasova/datasets/TrafficLightsDVC'

    meta_file_path = generate_metadata(path, output_folder)
    print(meta_file_path)


if __name__ == "__main__":
    train_classifiers()
