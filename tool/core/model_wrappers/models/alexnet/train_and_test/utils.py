import gc
from datetime import datetime
from torch.nn import functional as F

import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def calculate_val_loss_and_accuracy(model, loss_fn, val_loader, device, metrics):
    model.eval()
    with torch.no_grad():
        y_shuffled, y_preds, losses = [], [], []
        for img, label, _ in val_loader:
            img = img.to(device=device)
            label = label.to(device=device)
            y_predicted = model(img)
            loss = loss_fn(y_predicted, label)
            losses.append(loss.item())

            y_shuffled.append(label)
            y_preds.append(y_predicted.argmax(dim=-1))

        y_shuffled = torch.cat(y_shuffled)
        y_preds = torch.cat(y_preds)

        val_loss = torch.tensor(losses).cpu().mean()
        print("Valid Loss : {:.3f}".format(val_loss))
        metrics["valid loss"].append(val_loss)
        accuracy = accuracy_score(y_shuffled.cpu().detach().numpy(), y_preds.cpu().detach().numpy())
        print("Valid Acc  : {:.3f}".format(accuracy))
        metrics["valid acc"].append(accuracy)


def epoch_train(model, loss_func, optimizer, train_loader, val_loader, device, metrics):
    model.train()
    losses = []
    for img, label, _ in tqdm(train_loader):
        img = img.to(device=device)
        label = label.to(device=device)
        label_predicted = model(img)

        loss = loss_func(label_predicted, label)
        losses.append(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  # Update model weights according to current gradients values
        del img

    gc.collect()
    epoch_loss = torch.tensor(losses).mean()
    print("Train loss : {:.3f}".format(epoch_loss))
    metrics["train loss"].append(epoch_loss)
    calculate_val_loss_and_accuracy(model, loss_func, val_loader, device, metrics)


def train_model(model, loss_func, optimizer, train_loader, val_loader, device, epochs: int, save_model=True, name="",
                scheduler=None):
    metrics = {"train loss": [], "valid loss": [], "valid acc": []}
    best_valid_accuracy = 0.0
    for epoch in range(1, epochs + 1):
        print("Epoch {0}".format(epoch))
        epoch_train(model, loss_func, optimizer, train_loader, val_loader, device, metrics)

        if metrics["valid acc"][-1] > best_valid_accuracy:
            best_valid_accuracy = metrics["valid acc"][-1]

            if save_model:
                timestamp_str = datetime.utcnow().strftime("%y%m%d_%H%M%S.%f")[:-3]
                model_parameters_file = "".join((name, '_', str(best_valid_accuracy), '_', timestamp_str, '.pth'))
                torch.save(model.state_dict(), model_parameters_file)
        if scheduler is not None:
            scheduler.step(epoch + 1)

    return metrics


def predict(model, loader, device):
    y_shuffled, y_preds = [], []
    for img, label, img_paths in tqdm(loader):
        img = img.to(device)
        preds = model(img)
        y_preds.append(preds.detach().cpu())
        y_shuffled.append(label)
        del img
    gc.collect()
    y_preds, y_shuffled = torch.cat(y_preds), torch.cat(y_shuffled)

    return y_shuffled.cpu().detach().numpy(), F.softmax(y_preds, dim=-1).cpu().detach().numpy()

