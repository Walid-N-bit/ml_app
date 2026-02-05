from utils import *
from init import *
from CustomClasses import NeuralNetwork, ImageDataset, CNN
from torchvision import models
from torchvision.models import MobileNet_V3_Small_Weights
import torch
from torch.optim import lr_scheduler
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets
from torchvision import transforms

from wheat_data_utils import WheatImgDataset, get_class_weights, oversampler

from torchinfo import summary

from datetime import datetime
from tempfile import TemporaryDirectory


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", DEVICE)

# imagenet images are 224x224 so we resize our custom data to 224
TRANSFORM = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


# TRAINING_DATA = datasets.CIFAR10(
#     root="data", train=True, download=False, transform=TRANSFORM
# )
# TESTING_DATA = datasets.CIFAR10(
#     root="data", train=False, download=False, transform=TRANSFORM
# )

TRAINING_DATA = WheatImgDataset(
    data_file="compressed_images_wheat/train.csv", transform=TRANSFORM
)
TESTING_DATA = WheatImgDataset(
    data_file="compressed_images_wheat/test.csv", transform=TRANSFORM
)

IMAGE, _ = TRAINING_DATA[0]
C, H, W = image_shape(IMAGE)
print(f"Images shape: {C, H, W}")


CLASSES = TRAINING_DATA.classes  # dict of labels to class_names
print(f"\nClasses are:\n{CLASSES}\n")

CLASS_WEIGHTS = get_class_weights("compressed_images_wheat/train.csv").to(DEVICE)

TRAIN_SAMPLER = oversampler(
    data_path="compressed_images_wheat/train.csv", weights=CLASS_WEIGHTS
)
TESTING_SAMPLER = oversampler(
    data_path="compressed_images_wheat/test.csv", weights=CLASS_WEIGHTS
)
print(f"Train sampler:\n{list(TRAIN_SAMPLER.weights)}\n")
print(f"Test sampler:\n{list(TESTING_SAMPLER.weights)}\n")

BATCH_SIZE = 16

# TRAIN_LOADER = DataLoader(TRAINING_DATA, batch_size=BATCH_SIZE, shuffle=True)
# TEST_LOADER = DataLoader(TESTING_DATA, batch_size=BATCH_SIZE, shuffle=True)

TRAIN_LOADER = DataLoader(TRAINING_DATA, batch_size=BATCH_SIZE, sampler=TRAIN_SAMPLER)
TEST_LOADER = DataLoader(TESTING_DATA, batch_size=BATCH_SIZE, sampler=TESTING_SAMPLER)


Net = models.MobileNetV3

# MODEL = NeuralNetwork(input_size=H * W, output_size=len(CLASSES)).to(DEVICE)
# MODEL = CNN(in_channels=1, out_channels=3, kernel_size=5, out_features=len(CLASSES)).to(
#     DEVICE
# )

MODEL = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT).to(DEVICE)

# freeze head for feature extraction
for param in MODEL.parameters():
    param.requires_grad = False

MODEL.classifier[3] = nn.Linear(in_features=1024, out_features=len(CLASSES))


MODEL_PATH = "models/model_3.pth"

EPOCHS = 30

# summary(MODEL, input_size=(1, 3, 32, 32), device="cpu", verbose=1)


def train(dataloader: DataLoader, model: Net, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    val_loss = []

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
            val_loss.append(loss)
    val_loss = sum(val_loss) / len(val_loss)
    return val_loss


def test(dataloader, model: Net, loss_fn):
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


def evaluate_model(model: Net, test_data, device, classes: list):
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
            image = image.to(device).unsqueeze(0)
            pred = model(image)
            predicted, actual = classes[pred[0].argmax(0)], classes[label]
            preds_images.append((image, predicted))
            print(f'Predicted: "{predicted}", Actual: "{actual}"')
    show_predictions(preds_images)


def eval2(testloader, model: Net, classes):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
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
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] == 0:
            print(f"Total predictions for {classname} = {total_pred[classname]}")
        else:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")


def main():

    # requirements()
    t_start = datetime.now()

    model = MODEL
    model_exists = file_exists(MODEL_PATH)
    if model_exists:
        model = load_model(MODEL_PATH, model)

    model.to(DEVICE)
    # loss_fn = nn.CrossEntropyLoss(weight=CLASS_WEIGHTS)  # for single class
    loss_fn = nn.CrossEntropyLoss()  # for single class
    # loss_fn = nn.BCEWithLogitsLoss()  # for multiple classes
    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    # optimizer = torch.optim.SGD(model.classifier.parameters(), lr=0.001, momentum=0.9)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=0.0001)
    # scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=5)

    for t in range(EPOCHS):
        print(f"Epoch {t+1}\n-------------------------------")
        val_loss = train(TRAIN_LOADER, model, loss_fn, optimizer)
        test(TEST_LOADER, model, loss_fn)
        # scheduler.step()
        scheduler.step(val_loss)

    print("\nEnd of Training!")
    t_end = datetime.now() - t_start
    print("##########################\n")
    print(f"# Training time: {t_end} #\n")
    print("##########################\n")
    print("Evaluation...")
    save_model(model, path=MODEL_PATH)
    classes_list = list(CLASSES.values())
    evaluate_model(model, TESTING_DATA, DEVICE, classes_list)
    eval2(TEST_LOADER, model, classes_list)
    print("End of Evaluation!")
    t_eva = datetime.now() - t_end
    t_total = datetime.now() - t_start
    print("##########################\n")
    print(f"# Evaluation time: {t_eva} #\n")
    print("##########################\n")
    print(f"# Total elapsed time: {t_total} #\n")
    print("##########################\n")


if __name__ == "__main__":
    main()
