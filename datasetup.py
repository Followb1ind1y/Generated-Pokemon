"""
Contains Image datasets for Image generation.
"""
import os
import numpy as np
import torch

from torchvision import transforms
from PIL import Image
from typing import Tuple

class PokemonDataset(torch.utils.data.Dataset):
    """
    Pokemon Image Dataset.

    Args:
        imgs_dir: The root directory contains the image folders.
        image_size: The target size of the resized image.

    Example Usage:
        PokemonDataset(imgs_dir=os.path.join(output_dir, x), image_size=64)
    """
    def __init__(self, imgs_dir: str, image_size: int):
        super(PokemonDataset, self).__init__()
        self.imgs_dir = imgs_dir
        self.image_size = image_size
        self.imgs = list(sorted(os.listdir(self.imgs_dir)))

        self.transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5 for _ in range(3)],
                std=[0.5 for _ in range(3)],
            ),
        ])

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        curr_img_dir = os.path.join(self.imgs_dir, self.imgs[idx])
        image = Image.open(curr_img_dir, mode='r').convert('RGB')
        return self.transform(image)