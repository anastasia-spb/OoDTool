import argparse
import os

from tool.core.distance_wrapper.calculate_distances import DistanceCalculator


def get_ood():
    parser = argparse.ArgumentParser(description='Calculate Distances based on embeddings')
    parser.add_argument('-emb', '--embeddings', default='', required=True)
    parser.add_argument('-m', '--method', default='cosine', required=False, choices=DistanceCalculator.methods)
    args = parser.parse_args()

    embeddings = args.embeddings
    output_folder, _ = os.path.split(embeddings)

    calculator = DistanceCalculator(method_name=args.method,
                                    embeddings_pkl=embeddings,
                                    output_folder=output_folder)
    result = calculator.run()
    print(result)


if __name__ == "__main__":
    get_ood()
