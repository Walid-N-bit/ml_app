from utils import *
from init import *
from CustomClasses import NeuralNetwork, ImageDataset, CNN
from torchvision import models
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

print("Device: ", DEVICE)

# MODEL = NeuralNetwork(input_size=H * W, output_size=len(CLASSES)).to(DEVICE)
MODEL = models.mobilenet_v3_small().to(DEVICE)
# MODEL = CNN(in_channels=1, out_channels=3, kernel_size=5, out_features=len(CLASSES)).to(
#     DEVICE
# )
MODEL_PATH = "models/model.pth"

EPOCHS = 20


def load_model():

    return


def train(dataloader: DataLoader, model: CNN, loss_fn, optimizer):
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


def evaluate_model(model: CNN, test_data, device, classes: list):
    """
    evaluate model accuracy against a list of classes

    :param model: PyTorch model
    :param test_data: testing data
    :param device: device that the model runs on
    :param classes: data classes
    :type classes: list
    """
    model.eval()
    samples = []
    preds_images = []
    for _ in range(6):
        sample = torch.randint(len(test_data), size=(1,)).item()
        image, label = test_data[sample]
        samples.append((image, label))
    with torch.no_grad():
        for sample in samples:
            image, label = sample
            image = image.to(device).unsqueeze(1)
            pred = model(image)
            predicted, actual = classes[pred[0].argmax(0)], classes[label]
            preds_images.append((image, predicted))
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
    show_predictions(preds_images)


def eval2(testloader, model: CNN, classes):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(
        f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
    )
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    # again no gradients needed
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")


def main():

    # requirements()

    model = MODEL
    model_exists = file_exists(MODEL_PATH)
    if model_exists:
        model = load_model(MODEL_PATH, model)

    loss_fn = nn.CrossEntropyLoss()  # for single class
    # loss_fn = nn.BCEWithLogitsLoss()  # for multiple classes
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        train(TRAIN_LOADER, model, loss_fn, optimizer)
        test(TEST_LOADER, model, loss_fn)
    print("Done!")
    save_model(model, path=MODEL_PATH)
    evaluate_model(model, TESTING_DATA, DEVICE, CLASSES)
    eval2(TEST_LOADER, model, CLASSES)


if __name__ == "__main__":
    main()
