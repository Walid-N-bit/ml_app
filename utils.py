import subprocess
import os
import torch
from NeuralNetwork import NeuralNetwork


def cmd(input: str) -> str:
    """
    take input and run as a command. return output.

    :param input: input to run
    :type input: str
    :return: command output
    :rtype: str
    """
    proc = subprocess.Popen(
        input, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, shell=True, text=True
    )
    output, _ = proc.communicate()
    return output


def file_exists(path: str):
    """
    create path directory if it doesn't exist.
    check if path file exists or not. return true if exists, false otherwise.

    :param path: path to file
    :type path: str
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    file_exists = os.path.exists(path)
    return file_exists


def save_model(model, path: str):
    """
    save PyTorch model to a path

    :param model: PyTorch model
    :param path: file path
    :type path: str
    """
    torch.save(model.state_dict(), path)


def load_model(path: str, device):
    """
    load PyTorch model from file if file exists.
    return an initial model otherwise.

    :param path: path to file
    :type path: str
    :param device: device the model uses
    """
    model = NeuralNetwork().to(device)
    model_exists = file_exists(path)
    if model_exists:
        model.load_state_dict(torch.load(path, weights_only=True))
    return model


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
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        x = x.to(device)
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')
