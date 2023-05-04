import unittest
import pandas as pd

from tool.core.density_based_sampler.sampler import DensityBasedSampler
from tool.core import data_types


def test_sampler():
    embeddings_file = \
        '/home/vlasova/datasets/DroneBird/oodsession_1/AlexNetWrapper_DroneBird_256_230503_121854.emb.pkl'
    ood_score_file = \
        '/home/vlasova/datasets/DroneBird/oodsession_1/ood_score_230503_164449.ood.pkl'
    n_samples = 100

    def get_n_samples_with_highest_score(n=n_samples) -> list:
        ood_df = pd.read_pickle(ood_score_file)
        ood_df.sort_values(by=[data_types.OoDScoreType.name()], inplace=True, ascending=False)
        return ood_df[data_types.RelativePathType.name()][:n]

    sampler = DensityBasedSampler(embeddings_file, ood_score_file)
    sampler.fit(n_samples=n_samples)
    sampled_df = sampler.get_sampled_data()
    sampled_df.sort_values(by=[data_types.OoDScoreType.name()], inplace=True, ascending=False)
    selected_files = set(sampled_df[data_types.RelativePathType.name()][:n_samples])

    original_files = set(get_n_samples_with_highest_score(n_samples))

    assert len(selected_files) == len(original_files)

    score = len(selected_files.intersection(original_files)) / n_samples
    print(score)


if __name__ == '__main__':
    test_sampler()
