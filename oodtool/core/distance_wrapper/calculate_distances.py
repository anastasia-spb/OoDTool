from typing import Callable, List
import os
from tqdm import tqdm
import gc
import time
import pandas as pd
import numpy as np
import torch

from oodtool.core import data_types


class DistanceCalculator:
    def __init__(self, embeddings_pkl, output_folder, num_neighbours: int, device=None):
        self.distance_mat = None
        self.relative_paths = None
        self.output_file = None
        self.output_folder = output_folder
        _, output_file_name = os.path.split(embeddings_pkl)
        self.output_file_name = output_file_name[: -len(".emb.pkl")]
        self.data = self.__prepare_data(embeddings_pkl)
        self.num_neighbours = num_neighbours

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.device = device
        if self.device == torch.device('cuda'):
            # Cuda maintenance
            gc.collect()
            torch.cuda.empty_cache()

    def get_pdist(self, progress_callback: Callable[[List[int]], None] = None) -> str:

        self.distance_mat = self.__minkowski_distance(self.num_neighbours, self.device, progress_callback)
        self.output_file = self.__store()
        return self.output_file

    def get_output_file(self):
        return self.output_file

    def __prepare_data(self, embeddings_pkl):
        # Read embeddings
        data_df = pd.read_pickle(embeddings_pkl)
        embeddings = data_df[data_types.EmbeddingsType.name()].tolist()
        self.relative_paths = data_df[data_types.RelativePathType.name()].tolist()
        return np.array(embeddings, dtype=np.dtype('float64'))

    def __minkowski_distance(self, k: int, device: torch.device, progress_callback: Callable[[List[int]], None] = None):
        n = self.data.shape[0]
        torch_data = torch.from_numpy(self.data)
        results_matrix = torch.from_numpy(np.zeros(shape=(n, k)))
        if n < 50000:
            # Optimize calculations
            distance_mat = torch.from_numpy(np.zeros(shape=(n, n)))
            torch_data = torch_data.to(device, non_blocking=True)
            start_time = time.perf_counter()
            for i in tqdm(range(n - 1)):
                from_idx = i + 1
                distance_mat[i, from_idx:] = torch.cdist(torch_data[i, :][None, :], torch_data[from_idx:, :])
                distance_mat[from_idx:, i] = distance_mat[i, from_idx:]
                _, results_matrix[i, :] = torch.topk(distance_mat[i, :], k=k, largest=False, sorted=True)

                if progress_callback is not None:
                    progress_callback([i, n])

            _, results_matrix[n - 1, :] = torch.topk(distance_mat[n - 1, :], k=k, largest=False, sorted=True)

        else:
            # Large distance matrix doesn't fit into memory
            start_time = time.perf_counter()
            for i in tqdm(range(n)):
                distance_row = torch.cdist(torch_data[i, :][None, :].to(device), torch_data.to(device))
                _, results_matrix[i, :] = torch.topk(distance_row, k=k, largest=False, sorted=True)

                if progress_callback is not None:
                    progress_callback([i, n])

        if progress_callback is not None:
            progress_callback([n, n])

        end_time = time.perf_counter()
        print(f"Distances calculated in {end_time - start_time:0.4f} seconds")

        return results_matrix.detach().cpu()

    @classmethod
    def __store_result(cls, relative_paths, neighbours: List[List[str]]):
        matrix_df = pd.DataFrame()
        matrix_df[data_types.RelativePathType.name()] = relative_paths
        matrix_df[data_types.NeighboursType.name()] = neighbours
        return matrix_df

    def __store(self):
        # Convert indices to paths
        def indices_to_paths(row):
            return [self.relative_paths[int(value)] for value in row]

        neighbours = [indices_to_paths(row) for row in self.distance_mat]
        result = self.__store_result(self.relative_paths, neighbours)
        output_file = os.path.join(self.output_folder, self.output_file_name + '.dist.pkl')
        result.to_pickle(output_file)
        return output_file
