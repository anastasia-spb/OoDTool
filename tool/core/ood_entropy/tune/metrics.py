import pandas as pd
import numpy as np
import os
from sklearn.metrics import roc_auc_score
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

    data_df["conf"] = data_df.apply(lambda row: max(row[data_types.ClassProbabilitiesType.name()]), axis=1).values

    def in_ood(img_path: str) -> bool:
        p, _ = os.path.split(img_path)
        for f in ood_folders:
            if f == os.path.commonprefix([p, f]):
                return True
        return False

    data_df["ood_flag"] = data_df.apply(lambda row: in_ood(row[data_types.RelativePathType.name()]), axis=1).values
    # There is no correct classification for OoD samples
    data_df["miss"] = data_df.apply(lambda row: (row[data_types.LabelType.name()] != row["pred_label"] and
                                                 (not row["ood_flag"]) and
                                                 (row["conf"] > 0.75)), axis=1).values

    n_samples = data_df.shape[0]
    data_df.sort_values(by=[data_types.OoDScoreType.name()], inplace=True, ascending=False)
    total_ood = data_df["ood_flag"][:n_samples].values.sum()
    k50 = data_df["ood_flag"][:50].values.sum()
    k100 = data_df["ood_flag"][:100].values.sum()

    total_miss = data_df["miss"][:n_samples].values.sum()

    treshs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    def is_ood(score: float, threshold: float) -> bool:
        return score > threshold

    max_idx = 0
    max_val = 0.0
    for i, val in enumerate(treshs):
        ood_pred = data_df.apply(lambda row: is_ood(row[data_types.OoDScoreType.name()], val), axis=1).values
        ood_true = data_df["ood_flag"].values
        roc_auc = roc_auc_score(ood_true, ood_pred)
        if roc_auc >= max_val:
            max_val = roc_auc
            max_idx = i

    roc_auc_theshold = treshs[max_idx]

    if total_miss > 0.0:
        miss_with_high_ood = (data_df[data_df["miss"]]["ood_score"].values > roc_auc_theshold).sum()
        miss_with_high_ood_and_conf = \
            data_df.loc[(data_df['miss']) & (data_df['ood_score'] > roc_auc_theshold) & (data_df['conf'] > 0.75)].shape[0]

        metric = (1.0 * miss_with_high_ood_and_conf / total_miss) + (k50 / 50.0)

    else:
        miss_with_high_ood = 0.0
        miss_with_high_ood_and_conf = 0.0
        metric = k50 / 50.0

    metrics = {"n_samples": [n_samples], "total_ood": [total_ood], "k50": [k50], "k100": [k100],
               "total_miss": [total_miss], "miss_with_high_ood": [miss_with_high_ood],
               "miss_with_high_ood_and_conf": [miss_with_high_ood_and_conf],
               "roc_auc": [max_val], "roc_auc_theshold": [roc_auc_theshold],
               "metric": [metric]}

    return metrics


if __name__ == "__main__":
    results = ood_metrics(
        embeddings_metric_file='/home/nastya/Desktop/ood_datasets/PedestrianTrafficLights/oodsession_0/TimmResnetWrapper_densenet121_PedestrianTrafficLights_1024_230521_191905.811.emb.pkl',
        ood_file_path='/home/nastya/Desktop/OoDTool/tool/core/ood_entropy/tune/tmp2/trial_71/ood_score_230521_222728.515.ood.pkl',
        ood_folders=['ood_samples'])#, 'ood_samples/EMNIST', 'another_background']) #, 'ood_samples/EMNIST', 'another_background'])
        # ood_folders = ['ood_samples/EMNIST'])
        # ood_folders = ['another_background'])

    print(results)
