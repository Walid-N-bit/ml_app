from utils import *
from init import *
from CustomClasses import NeuralNetwork, ImageDataset, CNN

import torch
from torch import nn
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor, Lambda

TRAINING_DATA = datasets.FashionMNIST(
    root="data", train=True, download=False, transform=ToTensor()
)
TESTING_DATA = datasets.FashionMNIST(
    root="data", train=False, download=False, transform=ToTensor()
)

IMAGE, _ = TRAINING_DATA[0]
_, H, W = image_shape(IMAGE)

CLASSES = TRAINING_DATA.classes


BATCH_SIZE = 64

TRAIN_LOADER = DataLoader(TRAINING_DATA, batch_size=BATCH_SIZE)
TEST_LOADER = DataLoader(TESTING_DATA, batch_size=BATCH_SIZE)

DEVICE = (
    torch.accelerator.current_accelerator().type
    if torch.accelerator.is_available()
    else "cpu"
)

# MODEL = NeuralNetwork(input_size=H * W, output_size=len(CLASSES)).to(DEVICE)
MODEL = NeuralNetwork(input_size=H * W, output_size=len(CLASSES)).to(DEVICE)
MODEL_PATH = "models/model.pth"

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


def evaluate_model(model, test_data, device, classes: list):
    """
    evaluate model accuracy against a list of classes

    :param model: PyTorch model
    :param test_data: testing data
    :param device: device that the model runs on
    :param classes: data classes
    :type classes: list
    """
    model.eval()
    sample_idx = torch.randint(len(test_data), size=(1,)).item()
    # x, y = test_data[0][0], test_data[0][1]
    image, label = test_data[sample_idx]
    with torch.no_grad():
        image = image.to(device)
        pred = model(image)
        predicted, actual = classes[pred[0].argmax(0)], classes[label]
        save_img(image, predicted)
        print(f'Predicted: "{predicted}", Actual: "{actual}"')


def main():

    # requirements()

    model = MODEL
    model_exists = file_exists(MODEL_PATH)
    if model_exists:
        model = load_model(MODEL_PATH, DEVICE, model)

    loss_fn = nn.CrossEntropyLoss()  # for single class
    # loss_fn = nn.BCEWithLogitsLoss()  # for multiple classes
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train(TRAIN_LOADER, model, loss_fn, optimizer)
        test(TEST_LOADER, model, loss_fn)
    print("Done!")
    save_model(model, path=MODEL_PATH)
    evaluate_model(model, TESTING_DATA, DEVICE, CLASSES)


if __name__ == "__main__":
    main()
