import os
from pathlib import Path
from typing import Callable, List, Optional, Tuple, Union

from PIL import Image
import numpy as np
from puad.common import build_imagenet_normalization
import torch
from torch.utils.data import Dataset, Subset
import torchvision
from torchvision import transforms

TransformType = Callable[[Union[Image.Image, np.ndarray, torch.tensor]], Union[Image.Image, np.ndarray, torch.tensor]]


class RandomAugment:
    def __call__(self, img: Image.Image) -> Image.Image:
        i_aug = torch.randint(1, 4, (1,))
        lamda = torch.rand(1) * 0.4 + 0.8
        if i_aug == 1:
            return transforms.functional.adjust_brightness(img, lamda)
        elif i_aug == 2:
            return transforms.functional.adjust_contrast(img, lamda)
        return transforms.functional.adjust_saturation(img, lamda)


class NormalDataset(Dataset):
    def __init__(self, normal_image_dir: str, transform: Optional[TransformType] = None) -> None:
        super().__init__()
        self.img_paths = self._get_img_paths(normal_image_dir)
        self.transform = transform

    def _get_img_paths(self, img_dir: str) -> List[Path]:
        img_extension = ".png"
        img_paths = [p for p in sorted(Path(img_dir).iterdir()) if p.suffix == img_extension]
        return img_paths

    def __getitem__(self, index: int) -> Image.Image:
        path = self.img_paths[index]
        img = Image.open(path)
        img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self) -> int:
        return len(self.img_paths)


def split_dataset(
    img_dir: str,
    split_ratio: float,
    transform_1: Optional[TransformType] = None,
    transform_2: Optional[TransformType] = None,
) -> Tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    dataset_1 = NormalDataset(img_dir, transform=transform_1)
    dataset_2 = NormalDataset(img_dir, transform=transform_2)

    num_split_data = len(dataset_1) - int(len(dataset_1) * split_ratio)

    generator = torch.Generator()
    generator.manual_seed(42)

    indices = torch.randperm(len(dataset_1), generator=generator).tolist()
    indices_1, indices_2 = indices[:num_split_data], indices[num_split_data:]

    subset_1 = Subset(dataset_1, indices_1)
    subset_2 = Subset(dataset_2, indices_2)

    return subset_1, subset_2


def build_dataset(
    dataset_path: str,
    img_size: int = 256,
) -> Tuple[
    Union[NormalDataset, torch.utils.data.Subset],
    Union[NormalDataset, torch.utils.data.Subset],
    torchvision.datasets.ImageFolder,
]:
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((img_size, img_size)),
            torchvision.transforms.ToTensor(),
            build_imagenet_normalization(),
        ]
    )

    train_img_dir = os.path.join(dataset_path, "train", "good")
    valid_img_dir = os.path.join(dataset_path, "validation", "good")
    test_img_dir = os.path.join(dataset_path, "test")

    if os.path.exists(valid_img_dir):
        train_dataset = NormalDataset(train_img_dir, transform=transform)
        valid_dataset = NormalDataset(valid_img_dir, transform=transform)
    else:
        train_dataset, valid_dataset = split_dataset(
            train_img_dir, split_ratio=0.15, transform_1=transform, transform_2=transform
        )
    test_dataset = torchvision.datasets.ImageFolder(root=test_img_dir, transform=transform)

    return train_dataset, valid_dataset, test_dataset
