import gc
from datetime import datetime

import optuna
from optuna.trial import TrialState

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torch.utils.data import random_split

from tool.core.model_wrappers.models.alexnet.alexnet_module import AlexNet, AlexNetTransforms
from tool.core.model_wrappers.models.alexnet.train_and_test.utils import epoch_train
from tool.core.model_wrappers.models.utils.jpeg_dataset import JpegTrainDataset

DATA_ROOT_DIR = '/home/vlasova/datasets/'
METADATA_FILE = '/home/vlasova/datasets/0metadata/SummerWinter/SummerWinter.meta.pkl'
BATCHSIZE = 128
CLASSES = 2  # Winter and Summer
EPOCHS = 30


def train_with_trial(model, trial, loss_func, optimizer, train_loader, val_loader, device, epochs: int,
                     save_model=False):
    metrics = {"train loss": [], "valid loss": [], "valid acc": []}
    for epoch in range(1, epochs + 1):
        epoch_train(model, loss_func, optimizer, train_loader, val_loader, device, metrics)
        trial.report(metrics["valid acc"][-1], epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    if save_model:
        timestamp_str = datetime.now().strftime("%d-%b-%Y_%H-%M-%S")
        model_parameters_file = "".join(('./model_', timestamp_str, '.pth'))
        torch.save(model.state_dict(), model_parameters_file)

    return metrics


def objective(trial):
    # Cuda maintenance
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Torch device: ", device)

    # Generate the model.
    dropout = trial.suggest_float("dr", 0.2, 0.5, log=True)
    model = AlexNet(CLASSES, dropout).to(device)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_float("lr", 1e-5, 1e-1, log=True)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    train_dataset = JpegTrainDataset(METADATA_FILE, DATA_ROOT_DIR, AlexNetTransforms())

    validation_set_size = int(0.2 * len(train_dataset))
    train_set_size = len(train_dataset) - validation_set_size

    train_subset, validation_subset = random_split(train_dataset, [train_set_size, validation_set_size],
                                                   generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_subset, batch_size=BATCHSIZE, shuffle=True)
    val_loader = DataLoader(validation_subset, batch_size=BATCHSIZE, shuffle=False)

    # Training of the model
    loss_fn = nn.CrossEntropyLoss()
    metrics = train_with_trial(model, trial, loss_fn, optimizer, train_loader, val_loader, device, EPOCHS)

    return metrics["valid acc"][-1]


def start_optimization():
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100, timeout=None)

    pruned_trials = study.get_trials(deepcopy=False, states=[TrialState.PRUNED])
    complete_trials = study.get_trials(deepcopy=False, states=[TrialState.COMPLETE])

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    best_trial = study.best_trial

    print("  Value: ", best_trial.value)

    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    start_optimization()
