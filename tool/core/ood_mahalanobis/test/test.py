import os
import matplotlib.pyplot as plt

from tool.core import data_types
from tool.core.ood_mahalanobis.ood_score import OoDMahalanobisScore


def test_ood_mahalanobis_pipeline(use_gt_for_training):
    metadata_folder = './tmp'
    if not os.path.exists(metadata_folder):
        os.makedirs(metadata_folder)

    pipeline = OoDMahalanobisScore()
    test_file = 'test_data/TimmResnetWrapperImageNetVegetables230427_145051.emb.pkl'
    _ = pipeline.run([test_file], use_gt_for_training=use_gt_for_training, output_dir=metadata_folder,
                     probabilities_file=test_file)
    ood_df = pipeline.get_ood_df()
    plt.hist(ood_df[data_types.types.OoDScoreType.name()], density=True, bins=30)
    plt.show(block=False)


if __name__ == "__main__":
    test_ood_mahalanobis_pipeline(use_gt_for_training=True)
    test_ood_mahalanobis_pipeline(use_gt_for_training=False)
    plt.show()

