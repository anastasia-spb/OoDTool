import pandas as pd
import os
from typing import List
import itertools
from tool.core.ood_entropy.ood_score import OoDScore

import optuna
from optuna.trial import TrialState
from tool.core.classifier_wrappers.classifier_pipeline import ClassifierPipeline
from tool.core.classifier_wrappers.classifiers.lr_wrapper import SUPPORTED_CLASSIFIERS
from tool.core.ood_entropy.tune.metrics import ood_metrics


def run_with_trial(tags: List[str], embeddings_files: List[str], output_dir: str,
                   weight_decays: List[List[float]], embeddings_metric_file: str,
                   ood_folders: List[str]):
    assert len(tags) == len(embeddings_files)
    assert len(embeddings_files) == len(weight_decays)

    probabilities = []
    for emb, tag, wds in zip(embeddings_files, tags, weight_decays):
        classifier_pipeline = ClassifierPipeline(tag)
        output = classifier_pipeline.train_and_classify([emb], output_dir, use_gt_for_training=True,
                                                        probabilities_file=None, weight_decays=wds)
        probabilities.append(output[0])

    pipeline = OoDScore()
    ood_score_file = pipeline.run(probabilities, output_dir)
    trial_metrics = ood_metrics(embeddings_metric_file, ood_score_file, ood_folders)
    trial_metrics["tags"] = [tags]

    trial_metrics_file = os.path.join(output_dir, "metrics.csv")
    trial_metrics_df = pd.DataFrame.from_dict(trial_metrics)
    trial_metrics_df.to_csv(trial_metrics_file, index=False, header=True)

    return trial_metrics


def objective(trial, output_dir: str, embeddings_files, embeddings_metric_file, ood_folders, tags_combinations,
              n_classifiers):
    # Generate the optimizers.
    tags_idx = trial.suggest_int("tags", 0, len(tags_combinations) - 1)
    tags = tags_combinations[tags_idx]

    emb_weight_decays = []
    for i in range(n_classifiers):
        param_name = "".join(("c_exp_", str(i)))
        exp = trial.suggest_int(param_name, -9, 9)
        c = 10 ** exp
        if c in emb_weight_decays:
            # No sense to test classifiers with same weight decay
            raise optuna.exceptions.TrialPruned()
        emb_weight_decays.append(c)

    weight_decays = []
    for k in range(len(tags)):
        weight_decays.append(emb_weight_decays)

    trial_dir = os.path.join(output_dir, "".join(("trial_", str(trial.number))))
    if not os.path.exists(trial_dir):
        os.makedirs(trial_dir)

    trial_metrics = run_with_trial(tags, embeddings_files, trial_dir, weight_decays,
                                   embeddings_metric_file, ood_folders)

    trial.set_user_attr("metric", trial_metrics["metric"][0])
    trial.set_user_attr("k50", trial_metrics["k50"][0])
    trial.set_user_attr("k100", trial_metrics["k100"][0])
    trial.set_user_attr("total_miss", trial_metrics["total_miss"][0])
    trial.set_user_attr("miss_with_high_ood", trial_metrics["miss_with_high_ood"][0])
    trial.set_user_attr("miss_with_high_ood_and_conf", trial_metrics["miss_with_high_ood_and_conf"][0])
    trial.set_user_attr("tags", tags)

    return trial_metrics["metric"][0]


def start_optimization():
    # ALl parameters
    wd = '/home/vlasova/datasets/ood_datasets/PedestrianTrafficLights/oodsession_0'
    embeddings = [
        os.path.join(wd, 'TimmResnetWrapper_densenet121_PedestrianTrafficLights_1024_230510_154849.143.emb.pkl'),
        os.path.join(wd, 'TimmResnetWrapper_resnet50_PedestrianTrafficLights_2048_230510_153037.455.emb.pkl')]
    embeddings_metric_file = \
        os.path.join(wd, 'TimmResnetWrapper_densenet121_PedestrianTrafficLights_1024_230510_154849.143.emb.pkl')
    ood_folders = ['ood_samples']
    # We shouldn't optimize number of classifiers in ensemble at the same time with hyperparameters,
    # since it's not documented how optuna will treat other parameters which are suggested,
    # but not used: https://github.com/optuna/optuna/issues/1459
    n_classifiers = 5

    tags_combinations = []

    for item in SUPPORTED_CLASSIFIERS:
        # For simplicity use same classifier type for each embedding
        tags_combinations.append([item, item])

    output_dir = './tmp'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    func = lambda trial: objective(trial, output_dir=output_dir, embeddings_files=embeddings,
                                   embeddings_metric_file=embeddings_metric_file, ood_folders=ood_folders,
                                   tags_combinations=tags_combinations, n_classifiers=n_classifiers)

    study = optuna.create_study(direction="maximize")
    study.optimize(func, n_trials=100, timeout=None)

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


def run_trial_manually():
    wd = '/home/vlasova/datasets/all/TrafficLightsDVC/oodsession_3'
    embeddings = [
        os.path.join(wd, 'TimmResnetWrapper_resnet50_CarLightsDVC_2048_230510_081343.026.emb.pkl'),
        os.path.join(wd, 'RegnetWrapper_CarLightsDVC_784_230510_092138.871.emb.pkl')]
    embeddings_metric_file = \
        os.path.join(wd, 'RegnetWrapper_CarLightsDVC_784_230510_092138.871.emb.pkl')
    ood_folders = ['']

    output_dir = './tmp'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    run_with_trial(tags=["LogisticRegression_lbfgs", "LogisticRegression_lbfgs"],
                   embeddings_files=embeddings, output_dir=output_dir,
                   weight_decays=[[10e-5, 1.0, 10e5], [10e-5, 1.0, 10e5]],
                   embeddings_metric_file=embeddings_metric_file,
                   ood_folders=ood_folders)


if __name__ == "__main__":
    start_optimization()
    # run_trial_manually()
