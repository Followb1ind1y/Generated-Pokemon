"""
Contains various utility functions for PyTorch model training and saving.
"""

import os
import cv2
import torch
import numpy as np
import torchvision
import torch.nn as nn
import logging
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
from torch import optim
from torch.cuda.amp import GradScaler
from pathlib import Path


def display_sample(dataloader, fig_size, rows, cols):
    """
    plot the images in the batch.
    """
    images = next(iter(dataloader))
    
    fig = plt.figure(figsize=fig_size)
    for idx in range(cols*rows):
        fig.add_subplot(rows, cols, idx+1, xticks=[], yticks=[])
        plt.imshow(np.transpose(images[idx], (1,2,0)))

    plt.show()

def Display_forward_diffusion(dataloader, Diffusion_Model, fig_size, noise_step, num_images):
    """
    Simulate forward diffusion process.
    """
    image = next(iter(dataloader))
    plt.figure(figsize=fig_size)
    stepsize = int(noise_step/num_images)

    for idx in range(0, noise_step, stepsize):
        t = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images+1, int(idx/stepsize) + 1, xticks=[], yticks=[])
        x_t, _ = Diffusion_Model.q_sample(x=image, t=t)
        x_t = x_t.clamp(0, 1)
        plt.imshow(np.transpose(x_t[0], (1,2,0)))

@staticmethod
def save_images(images: torch.Tensor, save_path: str) -> None:
    grid = torchvision.utils.make_grid(images)
    img_arr = grid.permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(img_arr)
    img.save(save_path)

@staticmethod
def save_checkpoint(epoch: int,
                    model: nn.Module,
                    filename: str,
                    optimizer: optim.Optimizer = None,
                    scheduler: optim.lr_scheduler = None,
                    grad_scaler: GradScaler = None) -> None:
    checkpoint = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
    }
    if optimizer:
        checkpoint['optimizer'] = optimizer.state_dict()
    if scheduler:
        checkpoint['scheduler'] = scheduler.state_dict()
    if scheduler:
        checkpoint['grad_scaler'] = grad_scaler.state_dict()

    torch.save(checkpoint, filename)
    logging.info("=> Saving checkpoint complete.")

@staticmethod
def load_checkpoint(model: nn.Module,
                    filename: str,
                    optimizer: optim.Optimizer = None,
                    scheduler: optim.lr_scheduler = None,
                    grad_scaler: GradScaler = None) -> int:
    
    logging.info("=> Loading checkpoint")
    checkpoint = torch.load(filename, map_location="cuda")
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    if 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
    if 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])
    if 'grad_scaler' in checkpoint:
        grad_scaler.load_state_dict(checkpoint['grad_scaler'])
    return checkpoint['epoch']