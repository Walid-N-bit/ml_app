from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import decode_image
import pandas as pd
import glob
from PIL import Image
import re
import numpy as np
import torch

# def get_dirs(folder: str) -> list:
#     """
#     get a list of subdirectories in a path

#     :param folder: target path
#     :type folder: str
#     :return: sub directories
#     :rtype: list[str]
#     """
#     path = Path(folder)
#     sub_folders = [p.name for p in path.iterdir()]
#     return sub_folders


# def train_batches_paths(data_folder: str) -> list[str]:
#     """
#     training data folders in the wheat dataset

#     :param data_folder: dataset folder directory
#     :type data_folder: str
#     :return: paths of the training data batches
#     :rtype: list[str]
#     """
#     train_paths = [f"{data_folder}880/", f"{data_folder}864/"]
#     batches = []
#     for f in train_paths:
#         dirs = get_dirs(f)
#         for d in dirs:
#             cond = "Batch" in d
#             if cond:
#                 path = Path(f) / d
#                 batches.append(path.as_posix())
#     return batches


# def test_batches_paths(data_folder: str) -> list[str]:
#     """
#     testing data folders in the wheat dataset

#     :param data_folder: dataset folder directory
#     :type data_folder: str
#     :return: paths of the testing data batches
#     :rtype: list[str]
#     """
#     test_paths = [f"{data_folder}test/", f"{data_folder}test_07/"]
#     batches = []
#     for f in test_paths:
#         paths = train_batches_paths(f)
#         for p in paths:
#             batches.append(p)
#     return batches


# def create_labels_map(path: str) -> dict:
#     """
#     generate a labels map of the target path.
#     {<label>:<class name>...}

#     :param path: data path
#     :type path: str
#     :return: label map
#     :rtype: dict
#     """
#     sub_folder_names = get_dirs(path)
#     classes = {name: i for name, i in enumerate(sub_folder_names, 0)}
#     return classes


# def annotate_imgs(data_path: str, class_folder: str, labels_map: dict) -> list[tuple]:
#     """
#     generate a list of (image name, image label) pairs in a folder

#     :param data_path: path of the data
#     :type data_path: str
#     :param class_folder: folder name of the image
#     :type class_folder: str
#     :param labels_map: labels map of the data
#     :type labels_map: dict
#     :return: annotated images list
#     :rtype: list[tuple]
#     """
#     full_path = Path(data_path) / class_folder / "segmented_256_lcr_png"
#     data = []
#     files = get_dirs(full_path)
#     for img in files:
#         pair = (img, get_label(labels_map, class_folder))
#         data.append(pair)
#     return data


# def imgs_labels(data_path: str) -> list[tuple]:
#     """
#     annotate all images in a data folder

#     :param data_path: data folder path
#     :type data_path: str
#     :return: list of annotated images
#     :rtype: list[tuple]
#     """
#     data = []
#     labels_map = create_labels_map(data_path)
#     labels = labels_map.keys()
#     for label in labels:
#         pairs = annotate_imgs(data_path, labels_map.get(label), labels_map)
#         data.extend(pairs)
#     return data


# def images_data_csv(dataset_dir: str):
#     """
#     create .csv files for training and testing data.
#     run this function once to create the files.

#     :param dataset_dir: dataset directory
#     :type dataset_dir: str
#     """
#     train_patterns = [
#         f"{dataset_dir}864/*Batch/*/*/*",
#         f"{dataset_dir}880/*Batch/*/*/*",
#     ]
#     test_patterns = [
#         f"{dataset_dir}test/864/*Batch/*/*/*",
#         f"{dataset_dir}test/880/*Batch/*/*/*",
#         f"{dataset_dir}test_07/864/*Batch/*/*/*",
#         f"{dataset_dir}test_07/880/*Batch/*/*/*",
#     ]

#     def working_loop(patterns):
#         data = []
#         for t_path in patterns:
#             for path in glob.glob(t_path, recursive=True):
#                 path = Path(path)
#                 name = path.name
#                 class_name = path.parent.parent.name
#                 data.append(
#                     dict(name=name, class_name=class_name, path=path.as_posix())
#                 )
#         return data


#     train_data = working_loop(train_patterns)
#     test_data = working_loop(test_patterns)
#     train_pd = pd.DataFrame(train_data)
#     test_pd = pd.DataFrame(test_data)
#     train_pd.to_csv(f"{dataset_dir}/train.csv")
#     test_pd.to_csv(f"{dataset_dir}/test.csv")
def get_label(label_map: dict, class_name: str) -> int:
    """
    get a label of a given class name

    :param label_map: label map
    :type label_map: dict
    :param class_name: target class name
    :type class_name: str
    :return: label
    :rtype: int
    """
    label = [
        key for key, value in label_map.items() if value.lower() == class_name.lower()
    ]
    return label[0]


def labels_map_from_csv(csv_path: str) -> dict:
    data = pd.read_csv(csv_path)
    classes_names = sorted(set(data["class_name"]))
    labels_map = {name: i for name, i in enumerate(classes_names, 0)}
    return labels_map


def img_labels(data_file: str):
    img_labels = []
    labels_map = labels_map_from_csv(data_file)
    data = pd.read_csv(data_file)
    for row in data.itertuples():
        img_labels.append((row.name, get_label(labels_map, row.class_name)))
    return pd.DataFrame(img_labels)


def images_data_csv(
    dataset_path: str,
    train_folders: list[str],
    test_folders: list[str],
    include: str,
):

    def get_dirs(folders):
        dirs = []
        for f in folders:
            paths = glob.glob(f"{Path(dataset_path) / f / '**'}", recursive=True)
            dirs.extend(paths)
        return dirs

    def clean_dirs(dirs: list[str]):
        new_dirs = []
        for d in dirs:
            is_included = (include in d) and (".png" in d)
            if is_included:
                new_dirs.append(d)
        return new_dirs

    def data_prep(paths: list[str]):
        data = []
        for path in paths:
            match = re.search(r"Batch/(.*?)/segmented_256_lcr_png", path)
            path = Path(path)
            name = path.name
            class_name = match.group(1)
            data.append(dict(name=name, class_name=class_name, path=path.as_posix()))
        return data

    def save_data(data: list, name: str):
        data_pd = pd.DataFrame(data)
        data_pd.to_csv(f"{dataset_path}/{name}.csv")

    all_train_dirs = get_dirs(train_folders)
    all_test_dirs = get_dirs(test_folders)
    clean_train_dirs = clean_dirs(all_train_dirs)
    clean_test_dirs = clean_dirs(all_test_dirs)
    train_data = data_prep(clean_train_dirs)
    test_data = data_prep(clean_test_dirs)
    save_data(train_data, "train")
    save_data(test_data, "test")


def data_summary(data_path: str) -> dict:
    data = pd.read_csv(data_path)
    df = pd.DataFrame(data)
    class_names = sorted(set(df["class_name"].tolist()))
    total_size = len(df.index)
    size_per_class = {}
    for c in class_names:
        count = sum(df["class_name"] == c)
        size_per_class.update({c: count})
    stats = dict(total_size=total_size, size_per_class=size_per_class)
    return stats


def get_class_weights(data_path: str):
    data = pd.read_csv(data_path)
    _, counts = np.unique(data["class_name"], return_counts=True)
    class_counts = torch.tensor(counts)
    total_samples = class_counts.sum()
    class_weights = total_samples / (class_counts * len(class_counts))
    return class_weights / class_weights.sum()


class WheatImgDataset(Dataset):

    def __init__(self, data_file, transform=None, target_transform=None):
        self.img_labels = img_labels(data_file)
        self.data_dir = pd.read_csv(data_file)
        self.transform = transform
        self.target_transform = target_transform
        self.classes = labels_map_from_csv(data_file)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = self.data_dir.loc[idx, "path"]

        # using PIL because torchvision.transforms expect it
        image = Image.open(img_path).convert("RGB")

        label = self.img_labels.loc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
