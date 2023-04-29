import gc
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import random_split

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve

from tool.core.model_wrappers.models.alexnet.alexnet_module import AlexNet, AlexNetTransforms
from tool.core.model_wrappers.models.utils.jpeg_dataset import JpegDataset, JpegTrainDataset, JpegTestDataset
from tool.core.model_wrappers.models.alexnet.train_and_test.utils import train_model, predict
from tool.core.model_wrappers.models.alexnet.alexnet_wrapper import AlexNetWrapper

BATCHSIZE = 128
CLASSES = 2
EPOCHS = 20


def load_and_run(weights_path, model_labels):
    data_dir = '/home/nastya/Desktop/OoDTool/example_data/datasets'
    metadata_file = \
        '/home/nastya/Desktop/OoDTool/example_data/tool_working_dir/BalloonsBubbles/BalloonsBubbles.meta.pkl'

    # Cuda maintenance
    gc.collect()
    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    kwargs = {"weights_path": weights_path, "model_labels": model_labels}
    wrapper = AlexNetWrapper(device=device, **kwargs)
    image_transformation = wrapper.image_transformation_pipeline()

    dataset = JpegDataset(metadata_file, data_dir, image_transformation)
    loader = DataLoader(dataset, batch_size=BATCHSIZE, shuffle=False)

    y_actual, y_preds = predict(wrapper.model, loader, device)
    predicted_classes = np.argmax(y_preds, axis=1)

    show_metrics(y_actual, predicted_classes)


def train_and_test():
    data_dir = '/home/nastya/Desktop/OoDTool/example_data/datasets'
    metadata_file = \
        '/home/nastya/Desktop/OoDTool/example_data/tool_working_dir/BalloonsBubbles/BalloonsBubbles.meta.pkl'

    # Cuda maintenance
    gc.collect()
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Generate the model.
    dropout = 0.4
    model = AlexNet(CLASSES, dropout).to(device)

    optimizer_name = "RMSprop"  # Adam
    lr = 4.8e-05
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    image_transformation = AlexNetTransforms()
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
    metrics_df = pd.DataFrame.from_dict(metrics, orient="index")
    metrics_df.to_csv("AlexNet_eval_metrics.csv")

    test_loader = DataLoader(test_dataset, batch_size=BATCHSIZE, shuffle=False)
    y_actual, y_preds = predict(model, test_loader, device)
    predicted_classes = np.argmax(y_preds, axis=1)

    show_metrics(y_actual, predicted_classes)


def show_metrics(y_actual, y_preds):
    auc = roc_auc_score(y_actual, y_preds)
    fpr, tpr, _ = roc_curve(y_actual, y_preds)

    print("Test Accuracy : {}".format(accuracy_score(y_actual, y_preds)))
    print("\nClassification Report : ")
    print(classification_report(y_actual, y_preds))
    print("\nConfusion Matrix : ")
    print(confusion_matrix(y_actual, y_preds))

    plt.plot(fpr, tpr, label="AUC=" + str(auc))
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc=4)
    plt.show()


if __name__ == "__main__":
    # train_and_test()
    load_and_run('/home/nastya/Desktop/OoDTool/pretrained_weights/embedders/AlexNet_BalloonsBubbles.pth',
                 ["bubble", "balloon"])
