"""
Contains functions for training a PyTorch model.
"""
import os
import torch
import numpy as np
import pathlib
import logging
import datasetup, model, diffusion, utils

from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
from torch import optim
from torch.functional import F

class DDPM_Trainer:
    def __init__(
            self, 
            dataset_path: str,
            save_path: str = None,
            run_name: str = 'ddpm',
            image_size: int = 64,
            batch_size: int = 2,
            num_epochs: int = 500,
            noise_steps: int = 500,
            accumulation_iters: int = 16,
            learning_rate: float = 2e-4,
            sample_count: int = 1,
            num_workers: int = 0,
            device: str = 'cuda',
            fp16: bool = False,
            save_every: int = 2000,
            enable_train_mode: bool = True,        
    ):
        self.num_epochs = num_epochs
        self.device = device
        self.fp16 = fp16
        self.save_every = save_every
        self.accumulation_iters = accumulation_iters
        self.sample_count = sample_count

        base_path = save_path if save_path is not None else os.getcwd()
        self.save_path = os.path.join(base_path, run_name)
        pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)
        #self.logger = SummaryWriter(log_dir=os.path.join(self.save_path, 'logs'))

        if enable_train_mode:
            image_dataset = datasetup.PokemonDataset(imgs_dir=dataset_path, image_size=image_size)
            self.train_loader = torch.utils.data.DataLoader(
                image_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=num_workers,
                drop_last=False,
            )

        self.unet_model = model.UNet(noise_steps=noise_steps).to(device)
        self.diffusion = diffusion.Diffusion(img_size=image_size, device=self.device, noise_steps=noise_steps)
        self.optimizer = optim.Adam(
            params=self.unet_model.parameters(), lr=learning_rate,  # betas=(0.9, 0.999)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=300)
        self.grad_scaler = GradScaler()
        self.start_epoch = 0

    def train(self) -> None:
        logging.info(f'Training started....')

        progressbar = trange(self.num_epochs)
        for epoch in progressbar:
            # Epoch counter
            progressbar.set_description(f'Epoch {self.start_epoch}')
            self.start_epoch += 1
            accumulated_minibatch_loss = 0.0

            with tqdm(self.train_loader, leave=False) as pbar:
                for batch_idx, (real_images) in enumerate(pbar):
                    real_images = real_images.to(self.device)
                    current_batch_size = real_images.shape[0]
                    t = self.diffusion.sample_timesteps(batch_size=current_batch_size)
                    x_t, noise = self.diffusion.q_sample(x=real_images, t=t)

                    with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.fp16):
                        predicted_noise = self.unet_model(x=x_t, t=t)

                        loss = F.smooth_l1_loss(predicted_noise, noise)
                        loss /= self.accumulation_iters

                        accumulated_minibatch_loss += float(loss)

                    self.grad_scaler.scale(loss).backward()

                    if (batch_idx + 1) % self.accumulation_iters == 0:
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        accumulated_minibatch_loss = 0.0

                    if not batch_idx % self.save_every:
                        self.sample(epoch=epoch, batch_idx=batch_idx, sample_count=self.sample_count)

            self.scheduler.step()
