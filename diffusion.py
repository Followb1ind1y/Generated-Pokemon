"""
Contains forward and backword steps of the Diffusion Model
"""

import os
import torch
import logging
import torchvision

from PIL import Image
from typing import Tuple
from torch import nn
from torch.nn import functional as F
from tqdm.auto import tqdm, trange

class Diffusion:
    def __init__(
            self,
            device: str,
            img_size: int,
            noise_steps: int = 1000,
            beta_start: float = 1e-4,
            beta_end: float = 0.02,
    ):
        self.device = device
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size

        self.beta = self.linear_noise_schedule()
        self.alpha = 1 - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)

        self.sqrt_alpha_hat = torch.sqrt(self.alpha_hat)
        self.sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat)

        self.sqrt_alpha = torch.sqrt(self.alpha)
        self.std_beta = torch.sqrt(self.beta)
    
    def linear_noise_schedule(self) -> torch.Tensor:
        """
        The Linear Noise Scheduler using for beta setup. Generate beta values in different timestep.

        Example usage:
            Let beta_start = 0.01, beta_end = 0.02, noise_step = 10,
            => Diffusion.linear_noise_schedule() 
            => tensor([0.0100, 0.0111, 0.0122, 0.0133, 0.0144, 0.0156, 0.0167, 0.0178, 0.0189, 0.0200])
        """
        return torch.linspace(start=self.beta_start, end=self.beta_end, steps=self.noise_steps, device=self.device)
    
    def q_sample(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        The Forward Process of the Diffusion (From x_0 to x_T) using for get noise-added images at any timestep t.

        Let alpha_t = 1 - beta_t, alpha_hat_t = Prod^{t}_{i=1} alpha_i
        => q(x_t|x_0) = Normal(x_t; sqrt_alpha_hat * x_0, one_minus_alpha_hat * I)
        => x_t = sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * epsilon(Noise)

        Example usage:
            x_t, noise = Diffusion.q_sample(x=image, t=t)
        """
        sqrt_alpha_hat = self.sqrt_alpha_hat[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_hat = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)
        epsilon = torch.randn_like(x, device=self.device)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * epsilon, epsilon
    
    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        """
        Random timestep for each sample in a batch. Timesteps selected from [1, noise_steps].

        Example usage:
            Let noise_step = 100, batch_size = 4,
            => Diffusion.sample_timesteps(batch_size=4)
            => tensor([80, 64, 10, 39])
        """
        return torch.randint(low=1, high=self.noise_steps, size=(batch_size, ), device=self.device)
    
    def p_sample(self, eps_model: nn.Module, n: int, scale_factor: int = 2) -> torch.Tensor:
        """
        The Backward Process of the Diffusion (From x_T to x_0) using for image generation.

        Let alpha_t = 1 - beta_t, alpha_hat_t = Prod^{t}_{i=1} alpha_i
        => mu_theta(x_t,t) = 1/sqrt_alpha_t * (x_t - beta/sqrt_one_minus_alpha_hat_t * epsilon_theta(x_t, t))
        => x_{t-1} = 1/sqrt_alpha_t * (x_t - beta/sqrt_one_minus_alpha_hat_t * epsilon_theta(x_t, t)) + epsilon_t * random_noise

        Sample noise from normal distribution of timestep t > 1, else noise is 0. Before returning values
        are clamped to [-1, 1] and converted to pixel values [0, 1].

        Args:
            eps_model: Noise prediction model. `eps_theta(x_t, t)` in paper. Theta is the model parameters.
            n: Number of samples to process.
            scale_factor: Scales the output image by the factor.

        Example usage:
            sampled_images = Diffusion.p_sample(eps_model=unet_model, n=sample_count)
        """
        logging.info(f'Sampling {n} images....')

        eps_model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0, leave=False):
                t = torch.ones(n, dtype=torch.long, device=self.device) * i

                sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1, 1, 1)
                beta_t = self.beta[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)
                epsilon_t = self.std_beta[t].view(-1, 1, 1, 1)

                random_noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                x = ((1 / sqrt_alpha_t) * (x - ((beta_t / sqrt_one_minus_alpha_hat_t) * eps_model(x, t)))) +\
                    (epsilon_t * random_noise)

        eps_model.train()

        #x = x.clamp(0, 1)
        x = ((x.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)
        x = F.interpolate(input=x, scale_factor=scale_factor, mode='nearest-exact')
        return x
    
    def generate_gif(
            self,
            eps_model: nn.Module,
            n: int = 1,
            save_path: str = '',
            output_name: str = None,
            skip_steps: int = 20,
            scale_factor: int = 2,
    ) -> None:
        logging.info(f'Generating gif....')
        frames_list = []

        eps_model.eval()
        with torch.no_grad():
            x = torch.randn((n, 3, self.img_size, self.img_size), device=self.device)

            for i in tqdm(reversed(range(1, self.noise_steps)), position=0):
                t = torch.ones(n, dtype=torch.long, device=self.device) * i

                sqrt_alpha_t = self.sqrt_alpha[t].view(-1, 1, 1, 1)
                beta_t = self.beta[t].view(-1, 1, 1, 1)
                sqrt_one_minus_alpha_hat_t = self.sqrt_one_minus_alpha_hat[t].view(-1, 1, 1, 1)
                epsilon_t = self.std_beta[t].view(-1, 1, 1, 1)

                random_noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)

                x = ((1 / sqrt_alpha_t) * (x - ((beta_t / sqrt_one_minus_alpha_hat_t) * eps_model(x, t)))) +\
                    (epsilon_t * random_noise)

                if i % skip_steps == 0:
                    x_img = F.interpolate(input=x, scale_factor=scale_factor, mode='nearest-exact')
                    x_img = ((x_img.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)
                    grid = torchvision.utils.make_grid(x_img)
                    img_arr = grid.permute(1, 2, 0).cpu().numpy()
                    img = Image.fromarray(img_arr)
                    frames_list.append(img)

        eps_model.train()

        output_name = output_name if output_name else 'output'
        frames_list[0].save(
            os.path.join(save_path, f'{output_name}.gif'),
            save_all=True,
            append_images=frames_list[1:],
            optimize=False,
            duration=80,
            loop=0
        )