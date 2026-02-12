from wheat_data_utils import WheatImgDataset, oversampler
from torchvision import datasets
from torchvision import transforms
from typing import Literal
from torch.utils.data import DataLoader


# imagenet images are 224x224 so we resize our custom data to 224
TRANSFORM = transforms.Compose(
    [
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

TRAINING_DATA = WheatImgDataset(
    data_file="compressed_images_wheat/train.csv", transform=TRANSFORM
)
TESTING_DATA = WheatImgDataset(
    data_file="compressed_images_wheat/test.csv", transform=TRANSFORM
)


CLASSES = TRAINING_DATA.classes.values()

SAMPLER = oversampler(data_path="compressed_images_wheat/train.csv")


def data_loader(
    data,
    device: Literal["cuda", "cpu"],
    batch_size: int,
    sampler=SAMPLER,
    num_workers: int = 4,
):
    pin_mem = False
    if device == "cuda":
        pin_mem = True
    loader = DataLoader(
        data,
        num_workers=num_workers,
        pin_memory=pin_mem,
        batch_size=batch_size,
        shuffle=True,
        sampler=sampler,
    )
    return loader
