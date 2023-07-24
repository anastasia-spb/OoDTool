import numpy as np
import pandas as pd
from typing import List, Callable

from oodtool.core import data_types


def label_to_idx(labels: List[str], label: str, logs_callback: Callable[[str], None] = None) -> int:
    try:
        return labels.index(label)
    except ValueError:
        if logs_callback is not None:
            logs_callback(f"Label \"{label}\" isn't found in \"{labels}\"")
        return -1


def get_columns_which_start_with(df, column: str) -> List[str]:
    return [col for col in df.columns if col.startswith(column)]


def get_labels(data_df: pd.DataFrame) -> List[str]:
    if data_df.empty:
        return []
    return data_df[data_types.LabelsType.name()].iloc[0]


def preprocess_embeddings(embeddings_df: pd.DataFrame) -> np.ndarray:
    embeddings = embeddings_df[data_types.EmbeddingsType.name()].tolist()
    return np.array(embeddings, dtype=np.dtype('float64'))
