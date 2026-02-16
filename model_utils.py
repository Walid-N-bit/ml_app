import torch
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from torchvision import models


def train(dataloader: DataLoader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    train_acc, train_loss = 0, 0

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        train_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
        train_loss += loss.item()

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    train_acc = train_acc / size
    train_loss = train_loss / num_batches

    return train_acc, train_loss


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, test_acc = 0, 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            test_acc += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    test_acc /= size

    print(f"\n Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_acc, test_loss


def eval_general(model, test_data, device, classes: list, print_res: bool = True):
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
            if print_res:
                print(f'Predicted: "{predicted}", Actual: "{actual}"')


def eval_avg_acc(model, testloader, print_res: bool = True):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # images = images.to(DEVICE)
            # labels = labels.to(DEVICE)
            images = torch.as_tensor(images).to(DEVICE)
            labels = torch.as_tensor(labels).to(DEVICE)
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if print_res:
        print(
            f"Accuracy of the network on the 10000 test images: {100 * correct // total} %"
        )
    return correct // total


def eval_per_class(testloader, model, classes):

    actual_values = []
    pred_values = []
    # prepare to count predictions for each class
    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            # images = images.to(DEVICE)
            # labels = labels.to(DEVICE)
            images = torch.as_tensor(images).to(DEVICE)
            labels = torch.as_tensor(labels).to(DEVICE)
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1

            actual_values.extend(labels)
            pred_values.extend(predictions)
    # print accuracy for each class
    for classname, correct_count in correct_pred.items():
        if total_pred[classname] == 0:
            print(f"Total predictions for {classname} = {total_pred[classname]}")
        else:
            accuracy = 100 * float(correct_count) / total_pred[classname]
            print(f"Accuracy for class: {classname:5s} is {accuracy:.1f} %")

    return actual_values, pred_values
