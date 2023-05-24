"""
Contains functions for training a PyTorch model.
"""
import os
import torch
import time
import copy
import numpy as np
import pathlib
import logging

from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm, trange
from torch import optim
from torch.functional import F

class Trainer:
    def __init__(
            self,
            dataset_path: str,
            save_path: str = None,
            checkpoint_path: str = None,
            run_name: str = 'ddpm',
            image_size: int = 64,
            image_channels: int = 3,
            batch_size: int = 4,
            accumulation_iters: int = 16,
            sample_count: int = 1,
            num_workers: int = 0,
            device: str = 'cuda',
            num_epochs: int = 10000,
            fp16: bool = False,
            save_every: int = 2000,
            learning_rate: float = 2e-4,
            noise_steps: int = 500,
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
            image_dataset = PokemonDataset(imgs_dir=dataset_path, image_size=image_size)
            self.train_loader = torch.utils.data.DataLoader(
                image_dataset,
                batch_size=batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=num_workers,
                drop_last=False,
            )

        self.unet_model = UNet().to(device)
        self.diffusion = Diffusion(img_size=image_size, device=self.device, noise_steps=noise_steps)
        self.optimizer = optim.Adam(
            params=self.unet_model.parameters(), lr=learning_rate,  # betas=(0.9, 0.999)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=300)
        self.grad_scaler = GradScaler()

        self.start_epoch = 0
        if checkpoint_path:
            logging.info(f'Loading model weights...')
            self.start_epoch = load_checkpoint(
                model=self.unet_model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                grad_scaler=self.grad_scaler,
                filename=checkpoint_path,
            )

    def sample(
            self,
            epoch: int = None,
            batch_idx: int = None,
            sample_count: int = 1,
            output_name: str = None
    ) -> None:
        """
        Generates images with reverse process based on sampling method with both training model and ema model.
        """
        sampled_images = self.diffusion.p_sample(eps_model=self.unet_model, n=sample_count)

        model_name = f'model_{epoch}_{batch_idx}.jpg'

        if output_name:
            model_name = f'{output_name}.jpg'

        save_images(
            images=sampled_images,
            save_path=os.path.join(self.save_path, model_name)
        )

    def sample_gif(
            self,
            save_path: str = '',
            sample_count: int = 1,
            output_name: str = None,
    ) -> None:
        """Generates images with reverse process based on sampling method with both training model and ema model.
        """
        self.diffusion.generate_gif(
            eps_model=self.unet_model,
            n=sample_count,
            save_path=save_path,
            output_name=output_name,
        )

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

                        #pbar.set_description(
                        #    # f'Loss minibatch: {float(accumulated_minibatch_loss):.4f}, total: {total_loss:.4f}'
                        #    f'Loss minibatch: {float(accumulated_minibatch_loss):.4f}'
                        #)
                        accumulated_minibatch_loss = 0.0

                    if not batch_idx % self.save_every:
                        self.sample(epoch=epoch, batch_idx=batch_idx, sample_count=self.sample_count)

                        #save_checkpoint(
                        #    epoch=epoch,
                        #    model=self.unet_model,
                        #    optimizer=self.optimizer,
                        #    scheduler=self.scheduler,
                        #    grad_scaler=self.grad_scaler,
                        #    filename=os.path.join(self.save_path, f'model_{epoch}_{batch_idx}.pt')
                        #)

            self.scheduler.step()