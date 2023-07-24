import os
import numpy as np
import pandas as pd

from oodtool.core.ood_score.ood_relative_mahalanobis_distance import score
from oodtool.core import data_types


def test_ood_rmd_score():
    metadata_folder = '../../../example_data/DogsCats/oodsession_0'
    metadata_file = os.path.join(metadata_folder, 'DatasetDescription.meta.pkl')
    embeddings_file = os.path.join(metadata_folder, 'torch_embedder_towheeresnet50_v2.emb.pkl')

    metadata_df = pd.read_pickle(metadata_file)
    ood_score = score(embeddings_file, metadata_df)

    inverted_score = np.subtract(1.0, ood_score)
    sorted_ood_indices = np.argsort(inverted_score, axis=0)[:5]
    top_images = metadata_df.iloc[sorted_ood_indices][data_types.RelativePathType.name()].tolist()
    print(top_images)



