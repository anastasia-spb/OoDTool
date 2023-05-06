import argparse
import os

from tool.core.ood_entropy.ood_score import OoDScore


def get_ood():
    parser = argparse.ArgumentParser(description='Calculate OoD score')
    parser.add_argument('-p', '--probabilities', nargs='+', default=[], required=True)
    args = parser.parse_args()

    probabilities = args.probabilities
    if len(probabilities) < 1:
        raise RuntimeError('No inputs are provided')

    output_folder, _ = os.path.split(probabilities[0])

    pipeline = OoDScore()
    result = pipeline.run(probabilities, output_folder)
    print(result)


if __name__ == "__main__":
    get_ood()
