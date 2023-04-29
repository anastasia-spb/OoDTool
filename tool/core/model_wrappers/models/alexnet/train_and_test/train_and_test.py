import gc
import numpy as np

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from tool.models.alexnet.alexnet_module import AlexNet
from tool.models.utils.jpeg_dataset import JpegDataset, JpegTrainDataset, JpegTestDataset
from tool.models.alexnet.utils import train_model, predict
from tool.models.alexnet.alexnet_wrapper import AlexNetWrapper

BATCHSIZE = 128
CLASSES = 2  # Winter and Summer
EPOCHS = 20


def load_and_run(weights_path):
    data_dir = '/home/vlasova/Desktop/NIR/NIR/data/datasets'
    metadata_file = '/home/vlasova/Desktop/NIR/NIR/data/datasets_metadata/SummerWinter.pkl'

    # Cuda maintenance
    gc.collect()
    torch.cuda.empty_cache()

    wrapper = AlexNetWrapper(weights_path)
    image_transformation = wrapper.image_transformation_pipeline()

    dataset = JpegDataset(metadata_file, data_dir, image_transformation)
    loader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = wrapper.load_model(device)
    y_actual, y_preds = predict(model, loader, device)
    predicted_classes = np.argmax(y_preds, axis=1)

    print("Test Accuracy : {}".format(accuracy_score(y_actual, predicted_classes)))
    print("\nClassification Report : ")
    print(classification_report(y_actual, predicted_classes))
    print("\nConfusion Matrix : ")
    print(confusion_matrix(y_actual, predicted_classes))


def train_and_test():
    data_dir = '/home/vlasova/datasets'
    metadata_file = '/home/vlasova/OOD_WD/SummerWinter.pkl'

    # Cuda maintenance
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_transformation = AlexNetWrapper.image_transformation_pipeline()

    # Generate the model.
    dropout = 0.4
    model = AlexNet(CLASSES, dropout).to(device)

    optimizer_name = "Adam"
    lr = 4.8e-05
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    train_dataset = JpegTrainDataset(metadata_file, data_dir, image_transformation)
    test_dataset = JpegTestDataset(metadata_file, data_dir, image_transformation)

    validation_set_size = int(0.3 * len(train_dataset))
    train_set_size = len(train_dataset) - validation_set_size

    train_subset, validation_subset = random_split(train_dataset, [train_set_size, validation_set_size],
                                                   generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_subset, batch_size=BATCHSIZE, shuffle=True)
    val_loader = DataLoader(validation_subset, batch_size=BATCHSIZE, shuffle=False)

    # Training of the model
    loss_fn = nn.CrossEntropyLoss()
    metrics = train_model(model, loss_fn, optimizer, train_loader, val_loader, device, EPOCHS, True, 'alexnet')

    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)
    y_actual, y_preds = predict(model, test_loader, device)
    predicted_classes = np.argmax(y_preds, axis=1)

    print("Test Accuracy : {}".format(accuracy_score(y_actual, predicted_classes)))
    print("\nClassification Report : ")
    print(classification_report(y_actual, predicted_classes))
    print("\nConfusion Matrix : ")
    print(confusion_matrix(y_actual, predicted_classes))


if __name__ == "__main__":
    train_and_test()
    # load_and_run('alexnet_0.9781100478468899_10-Apr-2023_14-21-13.pth')
