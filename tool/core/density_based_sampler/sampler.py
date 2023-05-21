import os
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.neighbors import BallTree
from sklearn.preprocessing import MinMaxScaler

from tool.core import data_types


class DensityBasedSampler:
    def __init__(self, embeddings_file, ood_score_file):
        self.output_dir, file_name = os.path.split(embeddings_file)
        self.file_name = "".join((file_name.strip(".emb.pkl"), ".sampled.emb.pkl"))

        embeddings_df = pd.read_pickle(embeddings_file)
        ood_df = pd.read_pickle(ood_score_file)
        self.data_df = pd.merge(embeddings_df, ood_df[[data_types.RelativePathType.name(),
                                                       data_types.OoDScoreType.name()]],
                                on=data_types.RelativePathType.name(), how='inner')

    def __save(self) -> str:
        self.data_df.drop(columns=[data_types.OoDScoreType.name()])
        full_path = os.path.join(self.output_dir, self.file_name)
        self.data_df.to_pickle(full_path)
        return full_path

    def fit(self, n_samples=300, with_random_select=True) -> str:
        embeddings = self.data_df[data_types.EmbeddingsType.name()].tolist()
        embeddings = np.array(embeddings, dtype=np.dtype('float64'))
        ood_score = self.data_df[data_types.OoDScoreType.name()].tolist()
        selected_indices = self._fit_samples(embeddings, entropy=ood_score, n_samples=n_samples,
                                             with_random_select=with_random_select)
        self.data_df = self.data_df.iloc[selected_indices]
        return self.__save()

    def get_sampled_data(self) -> pd.DataFrame:
        return self.data_df

    @classmethod
    def __sample_probability(cls, density: float, ood_score: float) -> float:
        # We want to select samples with higher radius, ood_score and confidence
        return ood_score * density  # *confidence

    def _fit_samples(self, embeddings: np.ndarray, entropy: np.ndarray, n_samples, with_random_select) -> np.ndarray:
        """
        exact density biased sampling:
        under-sample dense regions and over-sample light regions.

        Ref: Palmer et al., Density Biased Sampling: An Improved Method for Data Mining and Clustering ,SIGMOD 2000
        """
        n_input_samples = embeddings.shape[0]
        if n_samples >= n_input_samples:
            # Sampling is not required. Return all samples
            return np.arange(0, n_input_samples, 1.0, dtype=int)

        tree = BallTree(embeddings, leaf_size=2)
        density = tree.kernel_density(embeddings, h=2.0, kernel='gaussian')
        normalized_density = MinMaxScaler().fit_transform(density.reshape(-1, 1))
        normalized_density = normalized_density.flatten()

        scores = np.zeros(shape=n_input_samples)
        for i in range(embeddings.shape[0]):
            # Normalize radius for each sample
            scores[i] = self.__sample_probability(density=normalized_density[i], ood_score=entropy[i])

        probabilities = softmax(scores)

        if with_random_select:
            np.random.seed(seed=42)
            # replace shall be set to False, since we want to select each sample only once
            selected_indices = np.random.choice(n_input_samples, n_samples, replace=False, p=probabilities)
        else:
            selected_indices = np.argpartition(probabilities, -n_samples)[-n_samples:]

        return selected_indices
