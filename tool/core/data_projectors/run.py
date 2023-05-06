import argparse
import os

from tool.core.data_projectors.data_projector import DataProjector


def get_ood():
    parser = argparse.ArgumentParser(description='Project data to 2D')
    parser.add_argument('-emb', '--embeddings', default='', required=True)
    parser.add_argument('-m', '--method', default='trimap', required=False, choices=DataProjector.methods)
    args = parser.parse_args()

    embeddings = args.embeddings
    output_folder, _ = os.path.split(embeddings)

    projector = DataProjector(args.method)
    result = projector.project(metadata_folder=output_folder,
                               embeddings_file=embeddings)
    print(result)


if __name__ == "__main__":
    get_ood()
