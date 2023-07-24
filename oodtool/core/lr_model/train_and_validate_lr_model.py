import os
import numpy as np
from typing import Optional
import torch
import pandas as pd
import pytorch_lightning as pl
from torch.utils.data import random_split, TensorDataset
from oodtool.core import data_types
from oodtool.core.lr_model.logistic_regression_model import LinearClassifier
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from oodtool.core.utils import data_helpers

NUM_WORKERS = os.cpu_count()


def prepare_data(embeddings_file: str, metadata_file: str):
    embeddings_df = pd.read_pickle(embeddings_file)
    metadata_df = pd.read_pickle(metadata_file)

    data_df = pd.merge(metadata_df, embeddings_df[[data_types.RelativePathType.name(),
                                                   data_types.EmbeddingsType.name()]],
                       on=data_types.RelativePathType.name(), how='inner')

    embeddings = data_df[data_types.EmbeddingsType.name()].tolist()
    embeddings = np.array(embeddings, dtype=np.dtype('float64'))

    labels = labels = ["dogs", "cats", "ood"]  # data_df[data_types.LabelsType.name()][0]
    y_true = data_df.apply(lambda row: data_helpers.label_to_idx(labels, row[data_types.LabelType.name()]),
                           axis=1).values
    y = np.array(y_true, dtype=np.dtype('float64'))
    train_indices = data_df.index[data_df[data_types.TestSampleFlagType.name()] == False].tolist()

    return embeddings[train_indices, :], y[train_indices], embeddings, labels, \
        data_df[data_types.RelativePathType.name()]


class LinearClassifierWrapper:
    def __init__(self):
        super().__init__()
        self.weight_decay = 0.1
        self.batch_size = 32
        self.checkpoint = None

    @classmethod
    def _train(cls, model, batch_size, train_dataset, valid_dataset, device, output_dir: str, max_epochs=300):
        trainer = pl.Trainer(default_root_dir=os.path.join(output_dir, "Torch_LR_Train"),
                             accelerator="gpu" if str(device).startswith("cuda") else "cpu",
                             devices="auto",
                             max_epochs=max_epochs,
                             callbacks=[ModelCheckpoint(save_weights_only=True, mode='max', monitor='val_acc'),
                                        LearningRateMonitor("epoch")],
                             enable_progress_bar=True,
                             check_val_every_n_epoch=10)
        trainer.logger._default_hp_metric = None

        # Data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  drop_last=False, pin_memory=True, num_workers=0)
        valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                  drop_last=False, pin_memory=True, num_workers=0)
        trainer.fit(model, train_loader, valid_loader)
        return trainer

    def __train_run(self, model, X_train, y_train, device, output_dir):
        dataset = TensorDataset(torch.from_numpy(X_train).type(torch.float),
                                torch.LongTensor(y_train))

        train_lengths = int(len(dataset) * 0.8)
        lengths = [train_lengths, len(dataset) - train_lengths]
        train_set, val_set = random_split(dataset, lengths, generator=torch.Generator().manual_seed(42))

        trainer = self._train(model=model, batch_size=self.batch_size, train_dataset=train_set,
                              valid_dataset=val_set, output_dir=output_dir, device=device)
        return trainer.checkpoint_callback.best_model_path

    def __eval_run(self, model, X_test, device):
        test_dataset = TensorDataset(torch.from_numpy(X_test).type(torch.float))
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False,
                                 drop_last=False, pin_memory=True, num_workers=0)

        model.eval()

        predictions = []
        for idx, img in enumerate(test_loader):
            img = img[0].to(device)
            with torch.no_grad():
                predictions.append(model(img))

        predictions = torch.cat(predictions)
        softmax = torch.nn.Softmax(dim=1)
        predictions = softmax(predictions)
        return predictions.detach().cpu().numpy()

    def get_checkpoint(self):
        return self.checkpoint

    def run(self, X_train: Optional[np.ndarray], y_train: Optional[np.ndarray],
            X_test: np.ndarray, weight_decay: float, output_dir: str,
            checkpoint: Optional[str] = None, num_classes: int = None) -> np.ndarray:

        if (num_classes is None) and (y_train is None):
            raise RuntimeError('Neither number of classes nor y_train data are provided.')

        if num_classes is None:
            num_classes = np.unique(y_train).shape[0]

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        feature_dim = X_test.shape[1]

        inference_mode = (checkpoint is not None) and os.path.isfile(checkpoint)

        if inference_mode:
            self.checkpoint = checkpoint
            weight_decay = 0.0
        else:
            assert weight_decay >= 0.0
            weight_decay = weight_decay

        model = LinearClassifier(feature_dim=feature_dim, num_classes=num_classes,
                                 weight_decay=weight_decay)
        model.to(device)

        if not inference_mode:
            self.checkpoint = self.__train_run(model, X_train, y_train, device, output_dir=output_dir)

        checkpoint = torch.load(self.checkpoint)
        model.load_state_dict(checkpoint["state_dict"])
        model.to(device)
        return self.__eval_run(model, X_test, device)


def train_logreg(embeddings_file: str, metadata_file: str, checkpoint_path: str):
    classifier = LinearClassifierWrapper()
    X_train, y_train, X, labels, relative_paths = prepare_data(embeddings_file, metadata_file)

    model_weights = None  # provide path to checkpoint
    wd = 0.1
    probabilities = classifier.run(X_train, y_train, X, wd, checkpoint_path, model_weights)
    predicted_classes_id = np.argmax(probabilities, axis=1)
    confidence = np.max(probabilities, axis=1)

    result_df = pd.DataFrame()
    result_df[data_types.RelativePathType.name()] = relative_paths
    result_df[data_types.ClassProbabilitiesType.name()] = [[i] for i in confidence]
    result_df[data_types.PredictedLabelsType.name()] = [[labels[i]] for i in predicted_classes_id]
    result_df[data_types.PredictedProbabilitiesType.name()] = [[i] for i in probabilities]
    result_df[data_types.LabelsForPredictedProbabilitiesType.name()] = [[labels] for i in predicted_classes_id]

    return result_df


def run():
    session_path = '/home/vlasova/datasets/ood_datasets/CatsDogs/oodsession_4'
    embeddings_file = os.path.join(session_path, 'torch_embedder_towheeresnet50_v2.emb.pkl')
    metadata_file = os.path.join(session_path, 'DatasetDescription.meta.pkl')
    checkpoint_path = os.path.join(session_path, 'checkpoint')

    result_df = train_logreg(embeddings_file, metadata_file, checkpoint_path)
    output_file = os.path.join(session_path, "torch_embedder_towheeresnet50log_regression" + '.clf.pkl')
    result_df.to_pickle(output_file)


if __name__ == "__main__":
    run()
