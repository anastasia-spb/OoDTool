import os
import time
import pandas as pd
import numpy as np
from scipy.spatial import distance

from tool.core import data_types


def store_result(data_df, distance_mat: np.ndarray):
    index_values = range(0, distance_mat.shape[0])
    matrix_df = pd.DataFrame(data=distance_mat,
                             index=index_values,
                             columns=index_values)
    matrix_df[data_types.RelativePathType.name()] = data_df[data_types.RelativePathType.name()].tolist()
    return matrix_df


def cosine_distances(embeddings_pkl):
    # Read embeddings
    data_df = pd.read_pickle(embeddings_pkl)
    embeddings = data_df[data_types.EmbeddingsType.name()].tolist()
    X = np.array(embeddings, dtype=np.dtype('float64'))
    # Calculate cosine distance
    m = X.shape[0]
    distance_mat = np.zeros(shape=(m, m))
    start_time = time.perf_counter()
    condensed_distance_matrix = distance.pdist(X, 'cosine')
    end_time = time.perf_counter()
    print(f"Cosine distances calculated in {end_time - start_time:0.4f} seconds")

    for i in range(m):
        for j in range(i + 1, m):
            idx = m * i + j - ((i + 2) * (i + 1)) // 2
            distance_mat[i, j] = condensed_distance_matrix[idx]
            distance_mat[j, i] = condensed_distance_matrix[idx]

    postprocessing_time = time.perf_counter()
    print(f"Postprocessing in {postprocessing_time - end_time:0.4f} seconds")

    result = store_result(data_df, distance_mat)
    result.to_pickle('cosine_distances.dist.pkl')


if __name__ == "__main__":
    input_data = '../../../example_data/tool_working_dir/BalloonsBubbles/TimmResnetWrapper_BalloonsBubbles_1024_230430_001343.emb.pkl'
    cosine_distances(input_data)
