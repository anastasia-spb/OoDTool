import os
import matplotlib.pyplot as plt

from tool.core import data_types
from tool.core.ood_entropy.ood_score import OoDScore


def test_ood_entropy_pipeline():
    metadata_folder = './tmp'
    if not os.path.exists(metadata_folder):
        os.makedirs(metadata_folder)

    pipeline = OoDScore()
    test_file = 'test_data/LogisticRegression_230501_191751.clf.pkl'
    _ = pipeline.run([test_file, test_file], metadata_folder)
    ood_df = pipeline.get_ood_df()
    plt.hist(ood_df[data_types.types.OoDScoreType.name()], density=True, bins=30)
    plt.show()


if __name__ == "__main__":
    test_ood_entropy_pipeline()
