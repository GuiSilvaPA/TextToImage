import torch
import torch.nn as nn
import torch.nn.functional as F

from functools import partial

class SignalNoiseRelationship():
    
    def __init__(self):
        pass
    
    def cosine(self, t, s: float = 0.008):
                
        f = (t + s) / (1 + s)
        w = f * torch.pi * 0.5
        c = (torch.cos(w) ** -2) - 1

        return -torch.log(c.clamp(min = 1e-5))
        
    def linear(self, t):
        
        xi = 1e-4 + 10 * (t ** 2)
        y  = torch.exp(xi) - 1
    
        return -torch.log(y)

def t_equal_x_dim(x, t):
    padding_dims = x.ndim - t.ndim
    
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

class GaussianDiffusion(nn.Module):
    
    def __init__(self, noise_type='cosine', timesteps=1000, device='cpu'):
        
        super(GaussianDiffusion, self).__init__()
        
        SNR = SignalNoiseRelationship()   
        
        self.timesteps = timesteps      
        self.snr_func  = SNR.cosine if 'cosine'==noise_type else SNR.linear    
        self.device    = device
        
    def get_times(self, batch_size, noise_level):
        return torch.full((batch_size,), noise_level, device=self.device, dtype=torch.long)
    
    def sample_random_times(self, batch_size, max_threshold = 0.999):
        return torch.zeros((batch_size,), device=self.device).float().uniform_(0, max_threshold)
    
    def get_condition(self, times):
        return times if times is None else self.snr_func(times)
    
    def get_sampling_timesteps(self, batch):
        times = torch.linspace(1., 0., self.timesteps + 1, device = self.device)
        return tuple([torch.stack((torch.full((batch,), times[i]),torch.full((batch,), times[i+1])), dim = 0) for i in range(len(times)-1)])
    
    def snr_to_alpha_sigma(self, log_snr):
        return torch.sqrt(torch.sigmoid(log_snr)), torch.sqrt(torch.sigmoid(-log_snr))

    def q_posterior(self, x_start, x_t, t, t_next):

        x_start = x_start.to(self.device)
        t_next  = t_next.to(self.device)

        log_snr      = self.snr_func(t)
        log_snr_next = self.snr_func(t_next)

        log_snr, log_snr_next = map(partial(t_equal_x_dim, x_t), (log_snr, log_snr_next))

        alpha, sigma           = self.snr_to_alpha_sigma(log_snr)
        alpha_next, sigma_next = self.snr_to_alpha_sigma(log_snr_next)

        alpha, sigma           = alpha.to(self.device), sigma.to(self.device)
        alpha_next, sigma_next = alpha_next.to(self.device), sigma_next.to(self.device)
        log_snr, log_snr_next  = log_snr.to(self.device), log_snr_next.to(self.device)

        c      = 1 - torch.exp(log_snr - log_snr_next)
        p_mean = alpha_next * (x_t * (1 - c) / alpha + c * x_start)

        p_variance     = (sigma_next ** 2) * c
        p_log_variance = torch.log(p_variance.clamp(min = 1e-20))

        return p_mean, p_variance, p_log_variance
    
    def q_sample(self, x_start, t, noise):

        x_start = x_start.to(self.device)
        noise   = noise.to(self.device)
        
        snr = self.snr_func(t)
        
        snr          = t_equal_x_dim(x_start, snr)
        alpha, sigma = self.snr_to_alpha_sigma(snr)
        
        return alpha * x_start + sigma * noise, snr
    
    def predict_start_from_noise(self, x_t, t, noise):
        
        x_t   = x_t.to(self.device)
        noise = noise.to(self.device)

        snr = self.snr_func(t)
        
        snr          = t_equal_x_dim(x_t, snr)
        alpha, sigma = self.snr_to_alpha_sigma(snr)

        alpha = alpha.to(self.device)
        sigma = sigma.to(self.device)

        num = (x_t - sigma * noise)
        dem = alpha.clamp(min = 1e-8)
        
        return  num/ dem