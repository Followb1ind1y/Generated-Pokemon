"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
import numpy as np

import logging
import matplotlib.pyplot as plt


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