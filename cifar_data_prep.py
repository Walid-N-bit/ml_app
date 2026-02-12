from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from typing import Literal

AUGS = (
    # transforms.RandomHorizontalFlip(p=0.3),
    # transforms.RandomVerticalFlip(p=0.3),
)

# imagenet images are 224x224 so we resize our custom data to 224
TRANSFORM = transforms.Compose(
    [
        *AUGS,
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

TRAINING_DATA = datasets.CIFAR10(
    root="data", train=True, download=True, transform=TRANSFORM
)
TESTING_DATA = datasets.CIFAR10(
    root="data", train=False, download=True, transform=TRANSFORM
)

CLASSES = TRAINING_DATA.classes


def data_loader(
    data, device: Literal["cuda", "cpu"], batch_size: int, num_workers: int = 4
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
    )
    return loader
