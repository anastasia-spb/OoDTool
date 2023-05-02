import os
import time
import pandas as pd
import numpy as np
from scipy.spatial import distance

from tool.core import data_types


class DistanceCalculator:
    methods = ['cosine', 'euclidian']
    method_name = methods[0]

    def __init__(self, method_name, embeddings_pkl, output_folder, verbose=0):
        self.method_name = method_name
        self.distance_mat = None
        self.relative_paths = None
        self.output_file = None
        self.output_folder = output_folder
        self.data = self.__prepare_data(embeddings_pkl)

    def run(self):
        if self.method_name == 'cosine':
            self.__cosine_distance()
        elif self.method_name == 'euclidian':
            self.__euclidian_distance()
        self.output_file = self.__store(self.output_folder)

    def get_output_file(self):
        return self.output_file

    def __prepare_data(self, embeddings_pkl):
        # Read embeddings
        data_df = pd.read_pickle(embeddings_pkl)
        embeddings = data_df[data_types.EmbeddingsType.name()].tolist()
        self.relative_paths = data_df[data_types.RelativePathType.name()].tolist()
        return np.array(embeddings, dtype=np.dtype('float64'))

    def __euclidian_distance(self):
        m = self.data.shape[0]
        self.distance_mat = np.zeros(shape=(m, m))

    def __cosine_distance(self):
        # Calculate cosine distance
        m = self.data.shape[0]
        self.distance_mat = np.zeros(shape=(m, m))

        start_time = time.perf_counter()
        condensed_distance_matrix = distance.pdist(self.data, 'cosine')
        end_time = time.perf_counter()
        print(f"Cosine distances calculated in {end_time - start_time:0.4f} seconds")

        for i in range(m):
            for j in range(i + 1, m):
                idx = m * i + j - ((i + 2) * (i + 1)) // 2
                self.distance_mat[i, j] = condensed_distance_matrix[idx]
                self.distance_mat[j, i] = condensed_distance_matrix[idx]

        postprocessing_time = time.perf_counter()
        print(f"Postprocessing in {postprocessing_time - end_time:0.4f} seconds")

    @classmethod
    def __store_result(cls, relative_paths, distance_mat: np.ndarray):
        index_values = range(0, distance_mat.shape[0])
        matrix_df = pd.DataFrame(data=distance_mat,
                                 index=index_values,
                                 columns=index_values)
        matrix_df[data_types.RelativePathType.name()] = relative_paths
        return matrix_df

    def __store(self, output_folder):
        result = self.__store_result(self.relative_paths, self.distance_mat)
        output_file = os.path.join(output_folder, self.method_name + '.dist.pkl')
        result.to_pickle(output_file)
        return output_file


if __name__ == "__main__":
    input_data = '../../../example_data/tool_working_dir/BalloonsBubbles/TimmResnetWrapper_BalloonsBubbles_1024_230430_001343.emb.pkl'
    calculator = DistanceCalculator(method_name='cosine',
                                    embeddings_pkl=input_data,
                                    output_folder="./")
    calculator.run()
