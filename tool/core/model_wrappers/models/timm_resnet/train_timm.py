import gc
import pandas as pd
import torch
import timm
import os
from timm.optim import create_optimizer_v2, optimizer_kwargs
from timm.scheduler import create_scheduler
from timm.data.transforms_factory import create_transform
from timm.data import resolve_data_config
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.utils.data import random_split

from tool.core.model_wrappers.models.utils.jpeg_dataset import JpegTrainDataset, JpegTestDataset
from tool.core.model_wrappers.models.alexnet.train_and_test.utils import train_model


def train_timm(
        checkpoint_path='',
        finetune=False):
    data_dir = '/home/nastya/Desktop/OoDTool/DroneBird'
    metadata_file = '/home/nastya/Desktop/OoDTool/DroneBird/oodsession_4/DroneBird.meta.pkl'
    classes = ["Drone", "Bird"]

    # Cuda maintenance
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if finetune:
        checkpoint_path = ''

    model = timm.create_model(model_name='vgg16_bn', pretrained=finetune, num_classes=len(classes),
                              checkpoint_path=checkpoint_path).to(device)
    config = resolve_data_config({}, model=model)
    image_transformation = create_transform(**config)

    train_dataset = JpegTrainDataset(metadata_file, data_dir, image_transformation)
    validation_set_size = int(0.3 * len(train_dataset))
    train_set_size = len(train_dataset) - validation_set_size
    train_subset, validation_subset = random_split(train_dataset, [train_set_size, validation_set_size],
                                                   generator=torch.Generator().manual_seed(42))

    batch_size = 16
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(validation_subset, batch_size=batch_size, shuffle=False)

    training_epochs = 1
    cooldown_epochs = 10
    num_epochs = training_epochs + cooldown_epochs

    lr = 5e-3
    optimizer = timm.optim.AdamP(model.parameters(), lr=lr)
    scheduler = timm.scheduler.CosineLRScheduler(optimizer,
                                                 t_initial=num_epochs,
                                                 cycle_decay=0.5,
                                                 lr_min=1e-6,
                                                 t_in_epochs=True,
                                                 warmup_t=3,
                                                 warmup_lr_init=1e-4,
                                                 cycle_limit=1, )
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    metrics = train_model(model, loss_fn, optimizer, train_loader, val_loader,
                          device=device, epochs=num_epochs, save_model=True,
                          name='timm_vgg16_bn', scheduler=scheduler)
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.to_csv("vgg16_bn_eval_metrics.csv")


if __name__ == "__main__":
    train_timm()