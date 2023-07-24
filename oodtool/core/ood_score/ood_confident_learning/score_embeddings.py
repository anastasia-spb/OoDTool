import pandas as pd
import numpy as np
import time
from oodtool.core.data_types import types
from typing import Callable, Union
from cleanlab.outlier import OutOfDistribution
from oodtool.core.utils import data_helpers


def score_embeddings(embedding_file: str, metadata_df: pd.DataFrame,
                     logs_callback: Callable[[str], None] = None) -> Union[None, np.ndarray]:
    start_time = time.perf_counter()
    train_indices = metadata_df.index[~metadata_df[types.TestSampleFlagType.name()]].tolist()

    emb_df = pd.read_pickle(embedding_file)
    assert emb_df[types.RelativePathType.name()].equals(metadata_df[types.RelativePathType.name()])
    embeddings = data_helpers.preprocess_embeddings(emb_df)
    train_embeddings = embeddings[train_indices, :]
    end_time = time.perf_counter()
    if logs_callback is not None:
        logs_callback(f"OoD preprocessing finished in {end_time - start_time:0.4f} seconds")
        logs_callback(f"Starting OoD model fitting on train subset...")

    ood = OutOfDistribution()
    start_time = time.perf_counter()
    ood.fit_score(features=train_embeddings)
    end_time = time.perf_counter()
    if logs_callback is not None:
        logs_callback(f"OoD model fit finished in {end_time - start_time:0.4f} seconds")
        logs_callback(f"Starting scoring images...")

    start_time = time.perf_counter()
    ood_score = ood.score(features=embeddings)
    ood_score_inv = np.subtract(1.0, ood_score)
    end_time = time.perf_counter()
    if logs_callback is not None:
        logs_callback(f"OoD scoring finished in {end_time - start_time:0.4f} seconds")

    return ood_score_inv
