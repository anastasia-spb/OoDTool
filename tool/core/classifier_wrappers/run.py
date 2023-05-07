import argparse
import os

from tool.core.classifier_wrappers import classifier_pipeline
from tool.core.classifier_wrappers.classifiers.lr_wrapper import SUPPORTED_CLASSIFIERS


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

    print("Choose classifier type from: {0}".format(SUPPORTED_CLASSIFIERS))
    classifier_tag = input()

    clf = classifier_pipeline.ClassifierPipeline(classifier_tag)

    print("Enter weight decay values separated by space: ")
    weight_decays = input()
    weight_decays = weight_decays.split(' ')
    weight_decays_values = [float(value) for value in weight_decays]

    result = clf.train_and_classify(embeddings_files=args.embeddings,
                                    output_dir=output_dir,
                                    use_gt_for_training=use_gt_for_training,
                                    probabilities_file=args.probabilities,
                                    weight_decays=weight_decays_values)

    print(result)


if __name__ == "__main__":
    train_classifiers()
