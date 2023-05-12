import os
import numpy as np
from typing import Optional
import torch
import torch.nn as nn
from torch.utils.data import random_split, DataLoader, TensorDataset
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint

from tool.core.classifier_wrappers.classifiers.i_classifier import IClassifier


class LinearClassifier(pl.LightningModule):
    def __init__(self, feature_dim, num_classes, weight_decay, lr=0.001, max_epochs=300):
        super(LinearClassifier, self).__init__()
        self.save_hyperparameters()
        self.model = torch.nn.Linear(self.hparams.feature_dim, self.hparams.num_classes)

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=self.hparams.lr,
                                    weight_decay=self.hparams.weight_decay)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=[int(self.hparams.max_epochs * 0.6),
                                                                        int(self.hparams.max_epochs * 0.8)],
                                                            gamma=0.1)
        return [optimizer], [lr_scheduler]

    def _calculate_loss(self, batch, mode='train'):
        feats, labels = batch
        predictions = self.model(feats)
        loss = torch.nn.functional.cross_entropy(predictions, labels)
        accuracy = (predictions.argmax(dim=-1) == labels).float().mean()

        self.log(mode + '_loss', loss)
        self.log(mode + '_acc', accuracy)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode='train')

    def validation_step(self, batch, batch_idx):
        self._calculate_loss(batch, mode='val')

    def forward(self, batch):
        return self.model(batch)


class LinearClassifierWrapper(IClassifier):
    tag = 'Torch_LR'

    def __init__(self, selected_model: str):
        super().__init__()
        self.weight_decay = 0.1
        self.batch_size = 32
        self.checkpoint = None
        self.selected_model = selected_model

    @classmethod
    def _train(cls, model, batch_size, train_dataset, valid_dataset, device, output_dir: str, max_epochs=100):
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
        train_set, val_set = random_split(dataset, [0.7, 0.3], generator=torch.Generator().manual_seed(42))

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
        softmax = nn.Softmax(dim=1)
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
