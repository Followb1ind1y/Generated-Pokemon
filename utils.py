"""
Contains various utility functions for PyTorch model training and saving.
"""
import torch
import numpy as np

import logging
import matplotlib.pyplot as plt


def display_sample(dataloader):
    """
    plot the images in the batch, along with the corresponding labels.
    """
    images = next(iter(dataloader))

    sampled_images = ((images.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)
    grid = torchvision.utils.make_grid(sampled_images)
    img_arr = grid.permute(1, 2, 0).cpu().numpy()
    img = torchvision.transforms.ToPILImage()(img_arr)
    img.show()

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
        noise = torch.randn_like(image)
        x_t = Diffusion_Model.q_sample(x_0=image, t=t, noise=noise)
        x_t = (((x_t*127.5)+127.5).clamp(0, 255)).type(torch.uint8)
        plt.imshow(np.transpose(x_t[0], (1,2,0)))