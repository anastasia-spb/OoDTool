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


def objective(trial, output_dir: str, embeddings_files, embeddings_metric_file, ood_folders, tags_combinations):
    # Generate the optimizers.
    tags_idx = trial.suggest_int("tags", 0, len(tags_combinations) - 1)
    tags = tags_combinations[tags_idx]
    max_clf_number = 5

    weight_decays = []
    for k in range(len(tags)):
        emb_weight_decays = []
        param_name = "".join(("n_classifiers_", str(k)))
        n_classifiers = trial.suggest_int(param_name, 3, max_clf_number)
        for i in range(n_classifiers):
            param_name = "".join(("c_exp_", str(k), "_", str(i)))
            exp = trial.suggest_int(param_name, -12, 12)
            c = 10 ** exp
            # if c in emb_weight_decays:
            #    raise optuna.exceptions.TrialPruned()
            emb_weight_decays.append(c)
        for j in range(n_classifiers, max_clf_number):
            #  be sure that any relative sampler will work correctly
            param_name = "".join(("c_exp_", str(k), "_", str(j)))
            empty_param = trial.suggest_int(param_name, -12, 12)
        emb_weight_decays = set(emb_weight_decays)
        weight_decays.append(list(emb_weight_decays))

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

    return trial_metrics["k50"][0] # trial_metrics["metric"][0]


def start_optimization():
    # ALl parameters
    embeddings = [
        '/home/nastya/Desktop/ood_datasets/DroneBird/oodsession_7/TimmResnetWrapper_densenet121_DroneBird_1024_230507_134347.030.emb.pkl',
        '/home/nastya/Desktop/ood_datasets/DroneBird/oodsession_7/TimmResnetWrapper_resnet34_DroneBird_512_230507_134006.988.emb.pkl']
    embeddings_metric_file = \
        '/home/nastya/Desktop/ood_datasets/DroneBird/oodsession_7/TimmResnetWrapper_densenet121_DroneBird_1024_230507_134347.030.emb.pkl'
    ood_folders = ['drones/close_to_ood', 'birds/close_to_ood', 'bird_drone',
                   'ood_samples']

    tags_combinations = []

    for item in itertools.product(SUPPORTED_CLASSIFIERS, repeat=len(embeddings)):
        tags_combinations.append(item)

    output_dir = './tmp'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    func = lambda trial: objective(trial, output_dir=output_dir, embeddings_files=embeddings,
                                   embeddings_metric_file=embeddings_metric_file, ood_folders=ood_folders,
                                   tags_combinations=tags_combinations)

    study = optuna.create_study(direction="maximize")
    study.optimize(func, n_trials=300, timeout=None)

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
