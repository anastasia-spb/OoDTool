import argparse
import os
import ast

from tool.core.classifier_wrappers import classifier_pipeline
from tool.core.classifier_wrappers.classifier_pipeline import CLASSIFIER_WRAPPERS


def train_classifiers():
    parser = argparse.ArgumentParser(description='Train Classifiers Ensemble')
    parser.add_argument('-emb', '--embeddings', nargs='+', default=[], required=True)
    parser.add_argument('-probs', '--probabilities', default='', required=False)
    args = parser.parse_args()

    use_gt_for_training = not os.path.isfile(args.probabilities)
    embeddings = args.embeddings
    if len(embeddings) < 1:
        raise RuntimeError('No inputs are provided')

    output_dir, _ = os.path.split(embeddings[0])

    print("Choose classifier type from: {0}".format(CLASSIFIER_WRAPPERS.keys()))
    classifier_tag = input()

    clf = classifier_pipeline.ClassifierPipeline(classifier_tag)

    kwargs = []

    print("Input ensemble size: ")
    ensemble_size = int(input())

    i = 0
    print("Enter input arguments in format: {0}".format(clf.input_hint()))
    while i < ensemble_size:
        kwargs_line = input()
        try:
            kwargs.append(ast.literal_eval(kwargs_line))
            i += 1
        except SyntaxError:
            print("Parameters couldn't be parsed")
            continue

    result = clf.train_and_classify(embeddings_files=args.embeddings,
                                    output_dir=output_dir,
                                    use_gt_for_training=use_gt_for_training,
                                    probabilities_file=args.probabilities,
                                    kwargs=kwargs)

    print(result)


if __name__ == "__main__":
    train_classifiers()
