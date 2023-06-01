"""
PyTorch implementation of Denoising Diffusion Probabilistic Models(DDPM) and Denoising Diffusion Implicit Models(DDIM)

References
    - DDPM paper, https://arxiv.org/pdf/2006.11239.pdf.
    - DDIM paper, https://arxiv.org/pdf/2010.02502.pdf.
    - Annotated Diffusion, https://huggingface.co/blog/annotated-diffusion.
    - pytorch-ddpm, https://github.com/w86763777/pytorch-ddpm/tree/master
    - Pytorch Diffusion https://github.com/quickgrid/pytorch-diffusion

"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPM:
    def __init__(self,
                 device: str,
                 img_size: int,
                 noise_steps: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02,
                 ):
        self.device = device
        self.img_size = img_size
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end

        self.betas = torch.linspace(start=self.beta_start, end=self.beta_end, steps=self.noise_steps, device=self.device).double()
        self.alphas = 1. - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.alphas_bar_prev = F.pad(self.alphas_bar, [1, 0], value=1)[:self.noise_steps]

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_bar = torch.sqrt(self.alphas_bar)
        self.sqrt_one_minus_alphas_bar = torch.sqrt(1. - self.alphas_bar)

        self.sqrt_recip_alphas_bar = torch.sqrt(1. / self.alphas_bar)
        self.sqrt_recipm1_alphas_bar = torch.sqrt(1. / self.alphas_bar - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_var = self.betas * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)
        self.posterior_log_var_clipped = torch.log(torch.cat([self.posterior_var[1:2], self.posterior_var[1:]]))
        self.posterior_mean_coef1 = torch.sqrt(self.alphas_bar_prev) * self.betas / (1. - self.alphas_bar)
        self.posterior_mean_coef2 = torch.sqrt(self.alphas) * (1. - self.alphas_bar_prev) / (1. - self.alphas_bar)

    def extract(self, v, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.

        Args:
            v: Tensor to extract from
            t: Timestep for which the coefficients are to be extracted
            x_shape: Shape of the current batched samples
        """
        out = torch.gather(v, index=t, dim=0).float()

        return out.view([t.shape[0]] + [1] * (len(x_shape) - 1))
    
    def q_sample(self, x_0, t, noise):
        """
        The Forward Process of the Diffusion (From x_0 to x_T) using for get noise-added images at any timestep t.

        Let alpha_t = 1 - beta_t, alpha_hat_t = Prod^{t}_{i=1} alpha_i
        => q(x_t|x_0) = Guassian(x_t; sqrt_alpha_hat * x_0, one_minus_alpha_hat * I)
        => x_t = sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * epsilon(Noise)

        Args:
            x_start: Initial sample (before the first diffusion step)
            t: Current timestep
            noise: Gaussian noise to be added at the current timestep

        Returns:
            Diffused samples at timestep `t`
        """
        return (self.extract(self.sqrt_alphas_bar, t, x_0.shape) * x_0 +
               self.extract(self.sqrt_one_minus_alphas_bar, t, x_0.shape) * noise)

    def predict_xstart_from_eps(self, x_t, t, eps):
        """
        x_t = sqrt_alpha_hat * x_0 + sqrt_one_minus_alpha_hat * epsilon(Noise)
        => x_0 = sqrt_recip_alphas_bar * x_t - sqrt_recipm1_alphas_bar * epsilon(Noise)
        """
        assert x_t.shape == eps.shape

        return (self.extract(self.sqrt_recip_alphas_bar, t, x_t.shape) * x_t -
                self.extract(self.sqrt_recipm1_alphas_bar, t, x_t.shape) * eps)

    def q_posterior(self, x_0, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior
        q(x_{t-1} | x_t, x_0)

        Args:
            x_start: Stating point(sample) for the posterior computation
            x_t: Sample at timestep `t`
            t: Current timestep
        Returns:
            Posterior mean and variance at current timestep
        """
        assert x_0.shape == x_t.shape
        posterior_mean = (
            self.extract(self.posterior_mean_coef1, t, x_t.shape) * x_0 +
            self.extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_log_var_clipped = self.extract(self.posterior_log_var_clipped, t, x_t.shape)

        return posterior_mean, posterior_log_var_clipped
    
    def p_mean_variance(self, pred_noise, x_t, t):
        x_0 = self.predict_xstart_from_eps(x_t, t, eps=pred_noise)
        model_mean, model_log_var = self.q_posterior(x_0, x_t, t)

        return model_mean, model_log_var
    
    def p_sample(self, pred_noise, x_t, t, time_step):
        """
        Sample from the diffuison model.

        Args:
            pred_noise: Noise predicted by the diffusion model
            x: Samples at a given timestep for which the noise was predicted
            t: Current timestep
        """
        model_mean, model_log_var = self.p_mean_variance(pred_noise, x_t=x_t, t=t)
        if time_step > 0:
            noise = torch.randn_like(x_t)
        else:
            noise = 0
        return (model_mean + torch.exp(0.5 * model_log_var) * noise)