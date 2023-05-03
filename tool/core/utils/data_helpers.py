import pandas as pd
from typing import List
import numpy as np

from tool.core import data_types


def get_columns_which_start_with(df, column: str) -> List[str]:
    return [col for col in df.columns if col.startswith(column)]


def merge_data_files(files: List[str]) -> (pd.DataFrame, List[str]):
    united_df = pd.DataFrame()
    for i, file in enumerate(files):
        df = pd.read_pickle(file)
        if united_df.empty:
            united_df = df
        else:
            suffix = '_' + str(i)
            united_df = pd.merge(united_df, df,
                                 on=data_types.RelativePathType.name(), how='inner',
                                 suffixes=('', suffix))
    return united_df


def get_labels(data_df: pd.DataFrame) -> List[str]:
    if data_df.empty:
        return []
    return data_df[data_types.LabelsType.name()][0]


def get_predictions(probabilities_file: str) -> np.ndarray:
    data_df = pd.read_pickle(probabilities_file)
    probabilities = data_df[data_types.ClassProbabilitiesType.name()].tolist()
    return np.argmax(probabilities, axis=1)


def get_number_of_classes(data_df: pd.DataFrame) -> int:
    if data_df.empty:
        return 0
    return len(data_df[data_types.LabelsType.name()][0])


def string_from_kwargs(tag, kwargs: dict):
    name = tag
    for value in kwargs.values():
        name += "_" + str(value)
    return name
