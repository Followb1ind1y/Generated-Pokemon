"""
Contains functions for training a PyTorch model.
"""
import os
import copy
import torch
import pathlib
import logging
import torchvision

from torch.cuda.amp import GradScaler
from tqdm.auto import tqdm, trange
from torch import optim, nn
from torch.functional import F
from torch.utils.data import RandomSampler

class DDPM_Trainer:
    def __init__(
            self, 
            dataset_path: str,
            save_path: str = None,
            run_name: str = 'ddpm',
            image_size: int = 64,
            batch_size: int = 2,
            num_epochs: int = 100,
            noise_steps: int = 1000,
            accumulation_iters: int = 16,
            learning_rate: float = 2e-4,
            sample_count: int = 4,
            num_workers: int = 0,
            device: str = 'cuda',
            fp16: bool = False,
            enable_train_mode: bool = True
            ):
        self.num_epochs = num_epochs
        self.device = device
        self.fp16 = fp16
        self.accumulation_iters = accumulation_iters
        self.sample_count = sample_count
        self.noise_steps = noise_steps
        self.image_size = image_size

        base_path = save_path if save_path is not None else os.getcwd()
        self.save_path = os.path.join(base_path, run_name)
        pathlib.Path(self.save_path).mkdir(parents=True, exist_ok=True)

        if enable_train_mode:
            image_dataset = PokemonDataset(imgs_dir=dataset_path, image_size=image_size)
            sampler = RandomSampler(image_dataset, replacement=True, num_samples=len(image_dataset)*3)
            self.train_loader = torch.utils.data.DataLoader(
                image_dataset,
                batch_size=batch_size,
                pin_memory=True,
                num_workers=num_workers,
                drop_last=False,
                sampler=sampler
            )

        self.unet_model = UNet().to(device)
        self.diffusion = DDPM(img_size=image_size, device=self.device, noise_steps=noise_steps)
        self.optimizer = optim.Adam(
            params=self.unet_model.parameters(), lr=learning_rate,  # betas=(0.9, 0.999)
        )
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=300)
        self.grad_scaler = GradScaler()

        self.ema = EMA(beta=0.95)
        self.ema_model = copy.deepcopy(self.unet_model).eval().requires_grad_(False)

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
                for batch_idx, real_images in enumerate(pbar):
                    real_images = real_images.to(self.device)
                    # 1. Get the batch size
                    current_batch_size = real_images.shape[0]
                    # 2. Sample timesteps uniformly
                    t = torch.randint(low=1, high=self.noise_steps, size=(current_batch_size, ), device=self.device)
                    # 3. Sample random noise to be added to the images in the batch
                    noise = torch.randn_like(real_images, device=self.device)
                    # 4. Diffuse the images with noise
                    x_t = self.diffusion.q_sample(x_0=real_images, t=t, noise=noise)

                    with torch.autocast(device_type=self.device, dtype=torch.float16, enabled=self.fp16):
                        # 5. Pass the diffused images and time steps to the network
                        predicted_noise = self.unet_model(x=x_t, t=t)
                        # 6. Calculate the loss
                        #loss = F.mse_loss(predicted_noise, noise)
                        loss = F.smooth_l1_loss(predicted_noise, noise)
                        loss /= self.accumulation_iters

                        accumulated_minibatch_loss += float(loss)

                    self.grad_scaler.scale(loss).backward()

                    if (batch_idx + 1) % self.accumulation_iters == 0:
                        self.grad_scaler.step(self.optimizer)
                        self.grad_scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                        self.ema.ema_step(ema_model=self.ema_model, model=self.unet_model)

                        accumulated_minibatch_loss = 0.0
            
            self.plot_images()
            self.scheduler.step()
    
    def generate_images(self, eps_model: nn.Module, num_images: int):

        eps_model.eval()
        with torch.no_grad():
            # 1. Randomly sample noise (starting point for reverse process)
            samples = torch.randn((num_images, 3, self.image_size, self.image_size), device=self.device)

            # 2. Sample from the model iteratively
            for time_step in reversed(range(self.noise_steps)):
                t = samples.new_ones([samples.shape[0], ], dtype=torch.long) * time_step
                eps = eps_model(samples, t)
                samples = self.diffusion.p_sample(pred_noise=eps, x_t=samples, t=t, time_step=time_step)
        
        eps_model.train()

        return samples
    
    def plot_images(self) -> None:
        """
        Generates images with reverse process based on sampling method with both training model and ema model.
        """
        sampled_images = self.generate_images(eps_model=self.ema_model, num_images=self.sample_count)
        sampled_images = ((sampled_images.clamp(-1, 1) + 1) * 127.5).type(torch.uint8)
        grid = torchvision.utils.make_grid(sampled_images)
        img_arr = grid.permute(1, 2, 0).cpu().numpy()
        img = torchvision.transforms.ToPILImage()(img_arr)
        img.show()