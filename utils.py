import subprocess
import os
import pandas as pd
import matplotlib.pyplot as plt
import torch
from CustomClasses import NeuralNetwork
import csv
from datetime import datetime
import argparse
from pathlib import Path


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
    if path:
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


def cmd_args():
    parser = argparse.ArgumentParser(
        description="options for training, testing, and evaluating pytorch model"
    )
    parser.add_argument(
        "--epochs", "-e", type=int, default=20, help="Number of epochs. default=20"
    )
    parser.add_argument(
        "--batch", "-b", type=int, default=64, help="Batch size. default = 64"
    )
    parser.add_argument(
        "--lr",
        "-l",
        nargs=2,
        type=float,
        default=(1e-5, 1e-3),
        help="Learning rates for model's backbone and classifier. default=(1e-5, 1e-3)",
    )
    parser.add_argument(
        "--freeze",
        "-f",
        action="store_true",
        help="Freeze model backbone parameters during training",
    )
    parser.add_argument("--load", "-ld", type=str, help="Load a model from a file")
    parser.add_argument(
        "--train", "-tr", action="store_true", help="Perform training + testing"
    )
    # parser.add_argument("--test", "-ts", action="store_true", help="")
    parser.add_argument("--eval", "-ev", action="store_true", help="Perform evaluation")
    parser.add_argument(
        "--scheduler", "-s", action="store_true", help="Enable a schedular"
    )
    parser.add_argument(
        "--decay", "-d", type=float, default=1e-2, help="Set weight decay. default=1e-2"
    )

    return parser.parse_args()
