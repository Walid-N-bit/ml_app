import torch
from torch.utils.data import DataLoader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(dataloader: DataLoader, model, loss_fn, optimizer):

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
    return correct, test_loss


def eval_general(model, test_data, device, classes: list):
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


def eval_per_class(testloader, model, classes):
    correct = 0
    total = 0
    actual_values = []
    pred_values = []
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
