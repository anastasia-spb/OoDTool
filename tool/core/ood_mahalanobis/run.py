import argparse
import os

from tool.core.ood_mahalanobis.ood_score import OoDMahalanobisScore


def get_score():
    parser = argparse.ArgumentParser(description='Calculate ood score based on mahalanobis dist')
    parser.add_argument('-emb', '--embeddings', default='', required=True)
    parser.add_argument('-probs', '--probabilities', default='', required=False)
    args = parser.parse_args()

    use_gt_for_training = not os.path.isfile(args.probabilities)
    output_dir, _ = os.path.split(args.embeddings)

    pipeline = OoDMahalanobisScore()
    result = pipeline.run(args.embeddings, use_gt_for_training=use_gt_for_training, output_dir=output_dir,
                          probabilities_file=args.probabilities)

    print(result)


if __name__ == "__main__":
    get_score()
