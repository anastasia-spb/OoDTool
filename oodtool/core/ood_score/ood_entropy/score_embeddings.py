import time
import pandas as pd
import numpy as np
from typing import List, Callable, Optional, Union
from oodtool.core.ood_score.ood_entropy.ood import OoD
from oodtool.core.data_types import types
from oodtool.core.utils import data_helpers
from sklearn.decomposition import PCA


def extract_features(X_train: np.ndarray, X_test: np.ndarray, n_components: int,
                     logs_callback: Callable[[str], None] = None):
    start_time = time.perf_counter()
    n_components = min(n_components, X_test.shape[1])

    projector = PCA(n_components=n_components, svd_solver='auto', random_state=42)
    projector.fit(X_train)
    reduced_embedding = projector.transform(X_test)
    end_time = time.perf_counter()

    if logs_callback is not None:
        logs_callback(f"PCA took {end_time - start_time:0.4f} seconds")
        logs_callback(f"Original features length: {X_test.shape[1]}. Reduced: {reduced_embedding.shape[1]}")

    return reduced_embedding


def score_embeddings(embeddings_files: List[str], metadata_df: pd.DataFrame,
                     progress_callback: Callable[[List[int]], None] = None,
                     logs_callback: Callable[[str], None] = None,
                     checkpoints: Optional[List[List[str]]] = None,
                     save_model=False, output_dir: Optional[str] = None,
                     checkpoint_tag: Optional[str] = None,
                     regularization_coefficients=OoD.default_regularization_coefficients,
                     classifier_type=OoD.default_classifier_type,
                     reduce_dim: bool = False, n_components: Optional[List[int]] = None) -> Union[None, np.ndarray]:
    train_indices = metadata_df.index[~metadata_df[types.TestSampleFlagType.name()]].tolist()
    y_train = metadata_df[types.LabelType.name()].to_numpy()

    embeddings = []
    for embeddings_file in embeddings_files:
        emb_df = pd.read_pickle(embeddings_file)
        assert emb_df[types.RelativePathType.name()].equals(metadata_df[types.RelativePathType.name()])
        embeddings.append(data_helpers.preprocess_embeddings(emb_df))

    if reduce_dim and (n_components is not None):
        assert len(n_components) == len(embeddings_files)
        reduced_embeddings = [extract_features(emb[train_indices, :], emb, n, logs_callback)
                              for emb, n in zip(embeddings, n_components)]
        embeddings = reduced_embeddings

    train_embeddings = [emb[train_indices, :] for emb in embeddings]

    if checkpoints is not None:
        assert len(checkpoints) == len(embeddings)

    ood = OoD(regularization_coefficients, classifier_type)
    start_time = time.perf_counter()
    score = ood.fit_and_score(train_embeddings, y_train[train_indices], embeddings, progress_callback, logs_callback,
                              checkpoints=checkpoints, save_model=save_model, output_dir=output_dir,
                              checkpoint_tag=checkpoint_tag)
    end_time = time.perf_counter()
    if logs_callback is not None:
        logs_callback(f"OoD Entropy overall time: {end_time - start_time:0.4f} seconds")

    return score
