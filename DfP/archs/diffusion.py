from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from functools import partial
from archs.Transformer_arch import Transformer
from archs.denoising_arch import denoising
from archs.latent_encoder_arch import latent_encoder_gelu
from utils.beta_schedule import default, make_beta_schedule
class DiffusionModel(nn.Module):
    def __init__(self, opt, device=None):
        super(DiffusionModel, self).__init__()
        self.opt = opt
        self.device = torch.device(
            device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        )
        self.net_le=self.model_to_device(latent_encoder_gelu(**opt['network_le']))
        self.net_le_dm=self.model_to_device(latent_encoder_gelu(**opt['network_le_dm']))
        self.net_d=self.model_to_device(denoising(**opt['network_d']))
        self.net_g=self.model_to_device(Transformer(**opt['network_g']))
        
        self._load_pretrained_models()
        self.train_dm=opt['train_dm']
        self.set_new_noise_schedule(self.opt['diffusion_schedule'],self.device)
        self.net_g.train()
        self.net_d.train()
        self.net_le.train()
        self.net_le_dm.train()
        for p in self.net_le.parameters():
            p.requires_grad = False
        if not self.train_dm:
            for p in self.net_le_dm.parameters():
                p.requires_grad = False
            for p in self.net_d.parameters():
                p.requires_grad = False
    
    def _load_pretrained_models(self):
        """Load pretrained model weights."""
        # Load latent encoder
        load_path = self.opt['path'].get('pretrain_network_le', None)
        if load_path is not None:
            self._load_network_with_path(self.net_le, load_path, 'le')
        
        # Load latent encoder for diffusion model
        load_path = self.opt['path'].get('pretrain_network_le_dm', None)
        if load_path is not None:
            self._load_network_with_path(self.net_le_dm, load_path, 'le_dm')
        
        # Load denoiser network
        load_path = self.opt['path'].get('pretrain_network_d', None)
        if load_path is not None:
            self._load_network_with_path(self.net_d, load_path, 'd')
        
        # Load generator network
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self._load_network_with_path(self.net_g, load_path, 'g')
    
    def _load_network_with_path(self, network, load_path, network_label):
        """Helper to load a specific network."""
        param_key = self.opt['path'].get(f'param_key_{network_label}', 'params')
        strict_load = self.opt['path'].get(f'strict_load_{network_label}', True)
        self.load_network(network, load_path, strict_load, param_key)
    def load_network(self, net, load_path, strict=True, param_key='params'):
        """Load network.

        Args:
            load_path (str): The path of networks to be loaded.
            net (nn.Module): Network.
            strict (bool): Whether strictly loaded.
            param_key (str): The parameter key of loaded network. If set to
                None, use the root 'path'.
                Default: 'params'.
        """
        net = self.get_bare_model(net)
        try:
            load_net = torch.load(load_path, map_location="cpu", weights_only=True)
        except TypeError:
            load_net = torch.load(load_path, map_location="cpu")
        if param_key is not None:
            if param_key not in load_net and 'params' in load_net:
                param_key = 'params'
            load_net = load_net[param_key]
        # remove unnecessary 'module.'
        for k, v in deepcopy(load_net).items():
            if k.startswith('module.'):
                load_net[k[7:]] = v
                load_net.pop(k)
        net.load_state_dict(load_net, strict=strict)
    def get_bare_model(self, net):
        """Get bare model, especially under wrapping with
        DistributedDataParallel or DataParallel.
        """
        return net
    def model_to_device(self, net):
        """Model to device. It also warps models with DistributedDataParallel
        or DataParallel.

        Args:
            net (nn.Module)
        """
        net = net.to(self.device)
        return net
    def training_parameters(self):
        optim_params = []
        self._collect_trainable_params(self.net_g, optim_params, 'G')
        if self.train_dm:
            self._collect_trainable_params(self.net_le_dm, optim_params, 'LE-DM')
            self._collect_trainable_params(self.net_d, optim_params, 'D')
        return optim_params
    def _collect_trainable_params(self, network, param_list, network_name):
        """Collect trainable parameters from a network."""
        for k, v in network.named_parameters():
            if v.requires_grad:
                param_list.append(v)
    def set_new_noise_schedule(self,schedule_opt,device):
        """Set up noise schedule for diffusion model."""
        to_torch = partial(torch.tensor, dtype=torch.float32, device=device)
        
        # Create beta schedule
        betas = make_beta_schedule(
            schedule=schedule_opt['schedule'],
            n_timestep=schedule_opt['timesteps'],
            linear_start=schedule_opt['linear_start'],
            linear_end=schedule_opt['linear_end'])
        
        betas = betas.detach().cpu().numpy() if isinstance(betas, torch.Tensor) else betas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))
        
        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        
        # Register buffers for diffusion process
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))
        
        # Posterior calculations
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer(
            'posterior_log_variance_clipped', 
            to_torch(np.log(np.maximum(posterior_variance, 1e-20)))
        )
        self.register_buffer(
            'posterior_mean_coef1', 
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        )
        self.register_buffer(
            'posterior_mean_coef2', 
            to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))
        )
    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x0 from noise."""
        return self.sqrt_recip_alphas_cumprod[t] * x_t - self.sqrt_recipm1_alphas_cumprod[t] * noise

    def q_posterior(self, x_start, x_t, t):
        """Compute posterior q(x_{t-1} | x_t, x_0)."""
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        return posterior_mean, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised=True, condition_x=None, ema_model=False):
        """Compute mean and variance of p(x_{t-1} | x_t)."""
        if condition_x is None:
            raise RuntimeError('Must have LQ/LR condition')
        
        t_tensor = torch.full(x.shape, t+1, device=self.betas.device, dtype=torch.long)
        noise = self.net_d(x, condition_x, t_tensor)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise)
        
        if clip_denoised:
            x_recon.clamp_(-1., 1.)
        
        model_mean, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_log_variance
    
    def p_sample_wo_variance(self, x, t, clip_denoised=True, condition_x=None, ema_model=False):
        """Sample from p(x_{t-1} | x_t) without noise."""
        model_mean, _ = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, ema_model=ema_model)
        return model_mean
    
    def p_sample_loop_wo_variance(self, x_in, x_noisy, ema_model=False):
        """Run full reverse process without adding noise."""
        img = x_noisy
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample_wo_variance(img, i, condition_x=x_in, ema_model=ema_model)
        return img

    def p_sample(self, x, t, clip_denoised=True, condition_x=None, ema_model=False):
        """Sample from p(x_{t-1} | x_t)."""
        model_mean, _ = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, condition_x=condition_x, ema_model=ema_model)
        return model_mean
    
    def p_sample_loop(self, x_in, x_noisy, ema_model=False):
        """Run full reverse process."""
        img = x_noisy
        for i in reversed(range(0, self.num_timesteps)):
            img = self.p_sample(img, i, condition_x=x_in, ema_model=ema_model)
        return img

    def q_sample(self, x_start, sqrt_alpha_cumprod, noise=None):
        """Forward diffusion sample."""
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            sqrt_alpha_cumprod * x_start +
            (1 - sqrt_alpha_cumprod**2).sqrt() * noise
        )
    def forward(self, input_features,gt_rgb=None,phase='train'):
        if self.train_dm and phase=='train':
            prior_z = self.net_le(input_features, gt_rgb)
            prior_d = self.net_le_dm(input_features)
            t = self.opt['diffusion_schedule']['timesteps']
            noise = torch.randn_like(prior_z)
            prior_noisy = self.q_sample(
                x_start=prior_z,
                sqrt_alpha_cumprod=self.alphas_cumprod[t-1],
                noise=noise
            )
            # Diffusion reverse process
            prior = self.p_sample_loop_wo_variance(prior_d, prior_noisy)
            output,feats=self.net_g(input_features[0], prior)
            return output,feats,prior,prior_z
        elif not self.train_dm or phase=='val':
            with torch.no_grad():
                prior_c = self.net_le_dm(input_features)
                prior_noisy = torch.randn_like(prior_c)
                prior = self.p_sample_loop(prior_c, prior_noisy)
            output,feats = self.net_g(input_features[0], prior)
            return output,feats,0,0
