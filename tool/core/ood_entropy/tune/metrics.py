import pandas as pd
import numpy as np
import os
from tool.core import data_types


def ood_metrics(embeddings_metric_file, ood_file_path, ood_folders):
    ood_df = pd.read_pickle(ood_file_path)
    emb_df = pd.read_pickle(embeddings_metric_file)

    data_df = pd.merge(emb_df, ood_df[
        [data_types.RelativePathType.name(), data_types.OoDScoreType.name()]],
                       on=data_types.RelativePathType.name(), how='inner')

    def get_predicted_label(probs, labels):
        return labels[np.argmax(probs)]

    data_df["pred_label"] = data_df.apply(lambda row: get_predicted_label(row[data_types.ClassProbabilitiesType.name()],
                                                                          row[data_types.LabelsType.name()]),
                                          axis=1).values

    data_df["miss"] = data_df.apply(lambda row: row[data_types.LabelType.name()] != row["pred_label"], axis=1).values

    data_df["conf"] = data_df.apply(lambda row: max(row[data_types.ClassProbabilitiesType.name()]), axis=1).values

    def in_ood(img_path: str) -> bool:
        p, _ = os.path.split(img_path)
        return p in ood_folders

    data_df["ood_flag"] = data_df.apply(lambda row: in_ood(row[data_types.RelativePathType.name()]), axis=1).values

    n_samples = data_df.shape[0]
    data_df.sort_values(by=[data_types.OoDScoreType.name()], inplace=True, ascending=False)
    total_ood = data_df["ood_flag"][:n_samples].values.sum()
    k50 = data_df["ood_flag"][:50].values.sum()
    k100 = data_df["ood_flag"][:100].values.sum()

    total_miss = data_df["miss"][:n_samples].values.sum()
    miss_with_high_ood = (data_df[data_df["miss"]]["ood_score"].values > 0.75).sum()
    miss_with_high_ood_and_conf = \
        data_df.loc[(data_df['miss']) & (data_df['ood_score'] > 0.75) & (data_df['conf'] > 0.75)].shape[0]

    metric = (miss_with_high_ood_and_conf / total_miss) + (1.0 * miss_with_high_ood / total_miss) + \
             (k100 / 100.0)

    metrics = {"n_samples": [n_samples], "total_ood": [total_ood], "k50": [k50], "k100": [k100],
               "total_miss": [total_miss], "miss_with_high_ood": [miss_with_high_ood],
               "miss_with_high_ood_and_conf": [miss_with_high_ood_and_conf], "metric": [metric]}

    return metrics


if __name__ == "__main__":
    ood_metrics(embeddings_metric_file='',
                ood_file_path='',
                ood_folders=[])
