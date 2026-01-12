from utils import *
from init import *
from NeuralNetwork import NeuralNetwork

import torch
from torch import nn
from torch.utils.data import DataLoader

# from torchvision import datasets
from torchvision.transforms import ToTensor

TRAINING_DATA = ""
TESTING_DATA = ""
BATCH_SIZE = 64

TRAIN_LOADER = DataLoader(TRAINING_DATA, batch_size=BATCH_SIZE)
TEST_LOADER = DataLoader(TESTING_DATA, batch_size=BATCH_SIZE)

DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

MODEL = NeuralNetwork().to(DEVICE)
MODEL_PATH = "models"

EPOCHS = 5


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def main():

    model = load_model(MODEL_PATH, DEVICE)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train(TRAIN_LOADER, model, loss_fn, optimizer)
        test(TEST_LOADER, model, loss_fn)
    print("Done!")

