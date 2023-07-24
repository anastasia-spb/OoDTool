import pytorch_lightning as pl
import torch


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

