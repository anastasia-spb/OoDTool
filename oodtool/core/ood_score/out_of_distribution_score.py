import numpy as np
import os
import pandas as pd
from typing import Callable, List, Union, Optional
from oodtool.core.ood_score import features_selector
from oodtool.core.ood_score import ood_entropy
from oodtool.core.ood_score import ood_confident_learning
from oodtool.core.data_types import types
import pickle


def store_ood(score_df, selected_method, output_dir) -> str:
    name = "".join((selected_method, '.ood.pkl'))
    output_file = os.path.join(output_dir, name)
    with open(output_file, 'wb') as handle:
        pickle.dump(score_df, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return output_file


def ood_score_to_df(score: np.ndarray, metadata_df: pd.DataFrame):
    ood_df = pd.DataFrame()
    ood_df[types.RelativePathType.name()] = metadata_df[types.RelativePathType.name()]
    ood_df[types.OoDScoreType.name()] = score
    return ood_df


def score_by_ood(method: str,
                 metadata_df: pd.DataFrame,
                 embeddings_files: Optional[List[str]] = None,
                 probabilities_files: Optional[List[str]] = None,
                 head_idx: Optional[int] = None,
                 progress_callback: Callable[[List[int]], None] = None,
                 logs_callback: Callable[[str], None] = None) -> Union[np.ndarray, None]:
    ood_score = None
    if (method == features_selector.OOD_ENTROPY) or (method == features_selector.OOD_ENTROPY_SWIN):
        assert len(embeddings_files) > 0
        assert types.TestSampleFlagType.name() in metadata_df
        assert types.LabelType.name() in metadata_df
        ood_score = ood_entropy.score_embeddings(embeddings_files=embeddings_files,
                                                 metadata_df=metadata_df,
                                                 progress_callback=progress_callback,
                                                 logs_callback=logs_callback)
    elif (method == features_selector.OOD_KNN_DIST) or (method == features_selector.OOD_KNN_DIST_SWIN):
        assert len(embeddings_files) > 0
        assert types.TestSampleFlagType.name() in metadata_df
        # method doesn't provide progress output
        if progress_callback is not None:
            progress_callback([-1, -1])
        ood_score = ood_confident_learning.score_embeddings(embedding_file=embeddings_files[0],
                                                            metadata_df=metadata_df,
                                                            logs_callback=logs_callback)
    elif method == features_selector.OOD_CONFIDENT_LEARNING:
        assert len(probabilities_files) > 0
        assert types.TestSampleFlagType.name() in metadata_df
        assert types.LabelType.name() in metadata_df
        assert head_idx is not None
        # method doesn't provide progress output
        if progress_callback is not None:
            progress_callback([-1, -1])
        ood_score = ood_confident_learning.score_predicted_probabilities(
            probabilities_file=probabilities_files[0],
            metadata_df=metadata_df,
            head_idx=head_idx,
            logs_callback=logs_callback)
    else:
        logs_callback(f"Selected ood method {method} isn't supported.")

    return ood_score
