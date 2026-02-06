import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from CustomClasses import NeuralNetwork
import csv
from datetime import datetime


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


def load_model(path: str, model):
    """
    load PyTorch model from file if file exists.
    return an initial model otherwise.

    :param path: path to file
    :type path: str
    :param device: device the model uses
    """
    # model = NeuralNetwork().to(device)
    model.load_state_dict(torch.load(path, weights_only=True))
    return model


def class_to_index(classes: list):
    """
    create a class:index dictionary from a list of classes

    :param classes: classes
    :type classes: list
    """
    return {k: i for k, i in zip(classes, range(len(classes)))}


def image_shape(image):
    channel, height, width = tuple(image.shape)
    return channel, height, width


def prep_image(image: torch.Tensor):
    image = image.detach().cpu()

    # Remove batch dimension if present
    if image.dim() == 4:
        image = image.squeeze(0)

    # If channels-first (C,H,W), move to (H,W,C)
    if image.dim() == 3:
        image = image.permute(1, 2, 0)

    # If grayscale, drop channel dimension
    if image.shape[-1] == 1:
        image = image.squeeze(-1)
    return image


# def save_img(image, label):
#     image = prep_image(image)
#     plt.figure()
#     plt.imshow(image)
#     plt.title(label)
#     plt.axis("off")
#     plt.savefig("output.png")
#     plt.close()


def save_predictions(data: list):

    fig, axes = plt.subplots(2, 3, figsize=(3 * 3, 2 * 3))
    axes = axes.flatten()

    for i, (img, label) in enumerate(data):
        img = prep_image(img)
        axes[i].imshow(img.astype("uint8"))
        axes[i].set_title(str(label))
        axes[i].axis("off")

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("img_evaluation.png")
    plt.close()


def plot_data(
    data: pd.DataFrame, x_col: str, y_col: str, color: str = "b", ls: str = "-"
):

    x = data[x_col]
    y = data[y_col]
    plt.figure(figsize=(16, 6))
    plt.xlabel(x_col)
    plt.xticks(x)
    plt.ylabel(y_col)
    plt.grid()
    plt.plot(x, y, color=color, ls=ls)
    plt.title(y_col)

    plt.tight_layout()
    plt.savefig(f"output_data/{y_col}_vs_{x_col}.png")
