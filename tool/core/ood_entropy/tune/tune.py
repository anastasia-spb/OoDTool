import pandas as pd
import os
from datetime import datetime
import math
from typing import List
from tool.core.ood_entropy.ood_score import OoDScore

import optuna
from optuna.trial import TrialState
from tool.core.classifier_wrappers.classifier_pipeline import ClassifierPipeline
from tool.core.ood_entropy.tune.metrics import ood_metrics

from tool.core import data_types
from tool.core.utils import data_helpers


def run_with_trial(tag: str, classifications_dist: List[dict], output_dir: str,
                   weight_decays_exp: List[float], embeddings_metric_file: str,
                   ood_folders: List[str]):

    probabilities_files = []

    for emb_result in classifications_dist:
        for wd in weight_decays_exp:
            probabilities_files.append(emb_result[wd])

    pipeline = OoDScore()
    ood_score_file = pipeline.run(probabilities_files, output_dir)
    trial_metrics = ood_metrics(embeddings_metric_file, ood_score_file, ood_folders)
    trial_metrics["tags"] = tag

    trial_metrics_file = os.path.join(output_dir, "metrics.csv")
    trial_metrics_df = pd.DataFrame.from_dict(trial_metrics)
    trial_metrics_df.to_csv(trial_metrics_file, index=False, header=True)

    return trial_metrics


def objective(trial, tag: str, output_dir: str, classifications_dist: List[dict], embeddings_metric_file, ood_folders,
              n_classifiers, weight_decays_exp_min, weight_decays_exp_max):

    emb_weight_decays = []
    for i in range(n_classifiers):
        param_name = "".join(("c_exp_", str(i)))
        exp = trial.suggest_int(param_name, weight_decays_exp_min, weight_decays_exp_max)
        if exp in emb_weight_decays:
            # No sense to test classifiers with same weight decay
            raise optuna.exceptions.TrialPruned()
        emb_weight_decays.append(exp)

    trial_dir = os.path.join(output_dir, "".join(("trial_", str(trial.number))))
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)

    trial_metrics = run_with_trial(tag, classifications_dist, trial_dir, emb_weight_decays,
                                   embeddings_metric_file, ood_folders)

    trial.set_user_attr("metric", trial_metrics["metric"][0])
    trial.set_user_attr("k50", trial_metrics["k50"][0])
    trial.set_user_attr("k100", trial_metrics["k100"][0])
    trial.set_user_attr("total_miss", trial_metrics["total_miss"][0])
    trial.set_user_attr("miss_with_high_ood", trial_metrics["miss_with_high_ood"][0])
    trial.set_user_attr("miss_with_high_ood_and_conf", trial_metrics["miss_with_high_ood_and_conf"][0])
    trial.set_user_attr("tag", tag)

    return trial_metrics["metric"][0]


def split_to_files(probabilities_df, weight_decays_exp, output_folder, tag, emb_num):
    result = dict()

    probabilities_columns = \
        data_helpers.get_columns_which_start_with(probabilities_df, data_types.ClassProbabilitiesType.name())

    for wd_exp, prob_column in zip(weight_decays_exp, probabilities_columns):
        timestamp_str = datetime.utcnow().strftime("%y%m%d_%H%M%S.%f")[:-3]
        name = "".join((tag, "_", str(emb_num), "_", str(wd_exp), "_", timestamp_str, ".clf.pkl"))
        output_file = os.path.join(output_folder, name)
        probabilities_df[[data_types.RelativePathType.name(), prob_column]].to_pickle(output_file)
        result[wd_exp] = output_file

    return result


def run_classifiers(tag: str, output_dir: str, embeddings_files: List[str], weight_decays_exp_min,
                    weight_decays_exp_max):
    weight_decays_range = list(range(weight_decays_exp_min, weight_decays_exp_max + 1))
    weight_decays = [math.pow(10, wd_exp) for wd_exp in weight_decays_range]
    classifier_pipeline = ClassifierPipeline(tag)

    result = []
    for i, emb in enumerate(embeddings_files):
        classifier_pipeline.classify(emb, output_dir, use_gt_for_training=True, probabilities_file=None,
                                     weight_decay=weight_decays)
        probabilities_df = classifier_pipeline.get_probabilities_df()
        results_dict = split_to_files(probabilities_df, weight_decays_range, output_dir, tag, i)
        result.append(results_dict)

    return result


def start_optimization():
    # ALl parameters
    wd = '/home/rpc/Desktop/ood_datasets/DroneBird/oodsession_9'
    embeddings = [
        os.path.join(wd, 'TimmResnetWrapper_densenet121_DroneBird_1024_230509_181421.718.emb.pkl'),
        os.path.join(wd, 'TimmResnetWrapper_resnet50_DroneBird_2048_230509_181143.997.emb.pkl')]
    embeddings_metric_file = \
        os.path.join(wd, 'TimmResnetWrapper_densenet121_DroneBird_1024_230509_181421.718.emb.pkl')
    ood_folders = ['ood_samples']
    # We shouldn't optimize number of classifiers in ensemble at the same time with hyperparameters,
    # since it's not documented how optuna will treat other parameters which are suggested,
    # but not used: https://github.com/optuna/optuna/issues/1459
    n_classifiers = 3

    tag = 'Torch_LR'

    output_dir = './tmp'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    wd_min = -12
    wd_max = 12

    result = run_classifiers(tag=tag, output_dir=output_dir, embeddings_files=embeddings,
                             weight_decays_exp_min=wd_min, weight_decays_exp_max=wd_max)

    func = lambda trial: objective(trial, tag=tag, output_dir=output_dir, classifications_dist=result,
                                   embeddings_metric_file=embeddings_metric_file, ood_folders=ood_folders,
                                   n_classifiers=n_classifiers, weight_decays_exp_min=wd_min,
                                   weight_decays_exp_max=wd_max)

    study = optuna.create_study(direction="maximize")
    study.optimize(func, n_trials=10000, timeout=10000)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial: ")
    best_trial = study.best_trial
    print("Trial number: ", best_trial.number)

    print("  Value: ", best_trial.value)

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))

    trials_metrics_df = study.trials_dataframe()
    trials_metrics_file = os.path.join(output_dir, "trials_metrics.csv")
    trials_metrics_df.to_csv(trials_metrics_file, index=False, header=True)


if __name__ == "__main__":
    start_optimization()
