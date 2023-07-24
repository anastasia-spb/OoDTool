import numpy as np
import time
import os
import pandas as pd
from oodtool.core import data_types
from typing import List, Callable, Tuple, Union
from cleanlab.outlier import OutOfDistribution
from oodtool.core.utils import data_helpers


def preprocess_probabilities(data_df: pd.DataFrame, head_idx: int) -> Tuple[np.ndarray, List[str]]:
    predicted_probabilities = data_df[data_types.PredictedProbabilitiesType.name()].tolist()
    head_model_labels = data_df[data_types.LabelsForPredictedProbabilitiesType.name()][0][head_idx]
    head_predicted_probabilities = np.stack([prob[head_idx] for prob in predicted_probabilities], axis=0)

    return head_predicted_probabilities, head_model_labels


def score_predicted_probabilities(probabilities_file: str, metadata_df: pd.DataFrame, head_idx: int,
                                  logs_callback: Callable[[str], None] = None) -> Union[None, np.ndarray]:
    start_time = time.perf_counter()
    train_indices = metadata_df.index[~metadata_df[data_types.TestSampleFlagType.name()]].tolist()

    prob_df = pd.read_pickle(probabilities_file)
    assert prob_df[data_types.RelativePathType.name()].equals(metadata_df[data_types.RelativePathType.name()])
    head_predicted_probabilities, head_model_labels = preprocess_probabilities(prob_df, head_idx)

    gt_labels = metadata_df[data_types.LabelType.name()].to_list()
    gt_labels_idx = np.array([data_helpers.label_to_idx(head_model_labels, gt_label, logs_callback)
                              for gt_label in gt_labels])
    end_time = time.perf_counter()

    if logs_callback is not None:
        logs_callback(f"Preprocessing finished in {end_time - start_time:0.4f} seconds")
        logs_callback(f"Starting ood model parameters fit...")

    start_time = time.perf_counter()
    ood = OutOfDistribution()
    try:
        ood.fit_score(pred_probs=head_predicted_probabilities[train_indices, :], labels=gt_labels_idx[train_indices])
    except ValueError:
        if logs_callback is not None:
            logs_callback(f"Fitting failed...")
        return None
    end_time = time.perf_counter()
    if logs_callback is not None:
        logs_callback(f"OoD model parameters fit finished in {end_time - start_time:0.4f} seconds")
        logs_callback(f"Starting scoring images...")

    start_time = time.perf_counter()
    ood_predictions_scores = ood.score(pred_probs=head_predicted_probabilities)
    ood_score_inv = np.subtract(1.0, ood_predictions_scores)
    end_time = time.perf_counter()
    if logs_callback is not None:
        logs_callback(f"Images scoring finished in {end_time - start_time:0.4f} seconds")

    return ood_score_inv


if __name__ == '__main__':
    ood_session_dir = '/home/vlasova/datasets/ood_datasets/Office-31/Office_31/oodsession_1'
    metadata_file_path = os.path.join(ood_session_dir, 'DatasetDescription.meta.pkl')
    metadata_df = pd.read_pickle(metadata_file_path)
    data_dir = 'home/vlasova/datasets/ood_datasets/Office-31/Office_31'
    output_dir = ood_session_dir

    probabilities_file = os.path.join(ood_session_dir, 'log_regression.clf.pkl')

    ood_score = score_predicted_probabilities(
        probabilities_file=probabilities_file,
        metadata_df=metadata_df, head_idx=0)
