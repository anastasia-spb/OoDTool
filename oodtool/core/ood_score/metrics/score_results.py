import numpy as np
import pandas as pd
from typing import List, Optional, Callable
from pathlib import Path
from oodtool.core import data_types
from sklearn.metrics import roc_auc_score

import optuna


def high_conf_miss_metric(ood_score: np.ndarray, gt_labels: List[str],
                          top_score: np.ndarray, top_tags: List[List[str]],
                          threshold: float, logs_callback: Callable[[str], None],
                          conf_threshold: float = 0.75, head_idx: int = 0):
    misclassified = [gt != tags[head_idx] for gt, tags in zip(gt_labels, top_tags)]
    with_high_conf = [score[head_idx] > conf_threshold for score in top_score]
    with_high_ood_score = [score > threshold for score in ood_score]

    misclassified_with_high_confidence = [miss and high_conf for miss, high_conf in
                                          zip(misclassified, with_high_conf)]

    resulting_metric_state = [miss and high_ood for miss, high_ood in
                              zip(misclassified_with_high_confidence, with_high_ood_score)]

    num_misclassified_samples = sum(misclassified)
    num_misclassified_samples_with_high_conf = sum(misclassified_with_high_confidence)
    num_miss_with_high_conf_and_ood_score = sum(resulting_metric_state)

    logs_callback(f"Number of misclassified samples: {num_misclassified_samples}.")
    logs_callback(f"Number of misclassified samples with confidence > {conf_threshold}:"
                  f" {num_misclassified_samples_with_high_conf}.")

    return num_miss_with_high_conf_and_ood_score


def first_k(ood_score: np.ndarray, ood_gt: List[bool], k=50):
    inverted_score = np.subtract(1.0, ood_score)
    sorted_gt = [gt for _, gt in sorted(zip(inverted_score, ood_gt))]
    k = min(k, len(sorted_gt))
    return sum(sorted_gt[:k])


def load_test_data(ood_df: pd.DataFrame, metadata_df: pd.DataFrame, ood_folders: List[str],
                   logs_callback: Callable[[str], None], probabilities_file: Optional[str] = None):
    probabilities_df = None
    if probabilities_file is not None:
        probabilities_df = pd.read_pickle(probabilities_file)

    data_df = pd.merge(metadata_df, ood_df[
        [data_types.RelativePathType.name(), data_types.OoDScoreType.name()]],
                       on=data_types.RelativePathType.name(), how='inner')

    if probabilities_df is not None:
        data_df = pd.merge(data_df, probabilities_df[
            [data_types.RelativePathType.name(), data_types.PredictedLabelsType.name(),
             data_types.ClassProbabilitiesType.name()]], on=data_types.RelativePathType.name(), how='inner')

    data_df["ood_gt"] = data_df.apply(lambda row: in_ood_path(row[data_types.RelativePathType.name()], ood_folders),
                                      axis=1).values
    num_of_ood = sum(data_df["ood_gt"])

    logs_callback(f"Number of ood gt samples: {num_of_ood}.")

    return data_df, num_of_ood


def predict_ood_state(ood_score: np.ndarray, threshold: float):
    def predict(score):
        return score >= threshold

    return np.apply_along_axis(predict, axis=0, arr=ood_score)


def roc_auc_score_ood(ood_score: np.ndarray, ood_gt: List[bool], threshold: float):
    assert 0.0 < threshold < 1.0
    return roc_auc_score(ood_gt, predict_ood_state(ood_score, threshold))


def objective(trial, ood_score: np.ndarray, ood_gt: List[bool]):
    threshold = trial.suggest_float("threshold", 0.1, 0.9, step=0.01)
    return roc_auc_score_ood(ood_score, ood_gt, threshold)


def find_threshold_for_roc_auc_score_ood(ood_score: np.ndarray, ood_gt: List[bool],
                                         logs_callback: Callable[[str], None], n_trials=1000):
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    func = lambda trial: objective(trial, ood_score, ood_gt)

    study = optuna.create_study(direction="maximize")
    study.optimize(func, n_trials=n_trials, timeout=10000)

    logs_callback("Study statistics: ")
    logs_callback(f"  Number of finished trials: {len(study.trials)}")

    logs_callback("Best trial: ")
    best_trial = study.best_trial
    logs_callback(f"Trial number: {best_trial.number}")
    logs_callback(f"  Value: {best_trial.value}")
    threshold = best_trial.params["threshold"]
    logs_callback(f"  Threshold: {threshold}")
    return threshold


def in_ood_path(file_path: str, ood_folders: List[str]):
    for folder in ood_folders:
        if Path(folder) in Path(file_path).parents:
            return True
    return False


def run_metrics(ood_df: pd.DataFrame, metadata_df: pd.DataFrame, ood_folders: List[str],
                logs_callback: Callable[[str], None], probabilities_file: Optional[str] = None,
                k: int = 100):
    data_df, num_of_ood = load_test_data(ood_df, metadata_df, ood_folders, logs_callback, probabilities_file)

    logs_callback(f"{data_df.shape[0]} samples found.")

    k_metric = first_k(data_df[data_types.OoDScoreType.name()].values,
                       data_df["ood_gt"].tolist(), k=k)

    logs_callback(f"Found {k_metric} ood samples in first {k}.")

    threshold = find_threshold_for_roc_auc_score_ood(data_df[data_types.OoDScoreType.name()].values,
                                                     data_df["ood_gt"].tolist(), logs_callback, n_trials=1000)

    roc_auc_score_value = roc_auc_score_ood(data_df[data_types.OoDScoreType.name()].values,
                                            data_df["ood_gt"].tolist(), threshold=threshold)

    logs_callback(f"ROC AUC score for threshold {threshold}: {roc_auc_score_value}.")

    if probabilities_file is not None:
        conf_threshold = 0.8
        value = high_conf_miss_metric(data_df[data_types.OoDScoreType.name()].values,
                                      data_df[data_types.LabelType.name()].values,
                                      data_df[data_types.ClassProbabilitiesType.name()].values,
                                      data_df[data_types.PredictedLabelsType.name()].values,
                                      threshold, logs_callback, conf_threshold=conf_threshold, head_idx=0)
        logs_callback(
            f"Found {value} misclassified images with confidence > {conf_threshold} and ood score > {threshold}.")
