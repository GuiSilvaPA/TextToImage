import torch
import torch.nn as nn
import torch.nn.functional as F

from resize_right import resize 

from ComplexModels import UNet
from GaussianDiffusion import GaussianDiffusion
from TextEncoder import TextEncoderT5Based

from contextlib import contextmanager
from einops import rearrange, repeat, reduce

from typing import List

from tqdm.notebook import tqdm
# from tqdm import tqdm

import torchvision.transforms as T

def normalize_neg_one_to_one(img):
    return img * 2 - 1

def unnormalize_zero_to_one(normed_img):
    return (normed_img + 1) * 0.5

def resize_image_to(image, target_image_size):
    orig_image_size = image.shape[-1]

    if orig_image_size == target_image_size:
        return image

    scale_factors = target_image_size / orig_image_size
    return resize(image, scale_factors = scale_factors)

def module_device(module):
    return next(module.parameters()).device

@contextmanager
def null_context(*args, **kwargs):
    yield

def default(val, d):
    if val is not None:
        return val
    return d() if callable(d) else d

def right_pad_dims_to(x, t):
    padding_dims = x.ndim - t.ndim
    if padding_dims <= 0:
        return t
    return t.view(*t.shape, *((1,) * padding_dims))

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# ===============================================================================================================================
# ===============================================================================================================================
# ===================== IMAGEN ==================================================================================================
# ===============================================================================================================================
# ===============================================================================================================================

class Imagen(nn.Module):
    def __init__(self, unet, image_sizes, text_encoder_name = 'google/t5-v1_1-small', channels = 3, timesteps = 1000,
                 cond_drop_prob = 0.1, noise_schedules = 'cosine', pred_objectives = 'noise', lowres_noise_schedule = 'linear',
                 lowres_sample_noise_level = 0.2, per_sample_random_aug_noise_level = False,  p2_loss_weight_gamma = 0.5,
                 p2_loss_weight_k = 1, dynamic_thresholding = True, dynamic_thresholding_percentile = 0.9, device='cpu'):

        super(Imagen, self).__init__()
        
        # loss

        self.loss_type   = 'l2'
        self.loss_fn     = F.mse_loss
        self.channels    = channels
        self.image_sizes = image_sizes

        # conditioning hparams

        self.condition_on_text, self.unconditional = True, False   
        
        self.lowres_sample_noise_level         = lowres_sample_noise_level
        self.per_sample_random_aug_noise_level = per_sample_random_aug_noise_level # False

        # n_unets   = len(unets)
        timesteps = (timesteps,)
        
        noise_schedules = (noise_schedules, 'cosine')
        mults           = 1 - len(noise_schedules) if 1 - len(noise_schedules) > 0 else 0
        noise_schedules = (*noise_schedules, *('linear',)*mults)

        self.lowres_noise_schedule = GaussianDiffusion(noise_type=lowres_noise_schedule, device=device)
        self.pred_objectives       = (pred_objectives,)

        self.text_encoder_name = text_encoder_name
        self.text_embed_dim    = TextEncoderT5Based(text_encoder_name).embed_dim
        self.text_encoder      = TextEncoderT5Based(text_encoder_name)     

        self.noise_schedulers = nn.ModuleList([])
        for timestep, noise_schedule in zip(timesteps, noise_schedules):
            noise_scheduler = GaussianDiffusion(noise_type=noise_schedule, timesteps=timestep, device=device)
            self.noise_schedulers.append(noise_scheduler)
            
        self.unets           = nn.ModuleList([unet[0]])        
        self.sample_channels = (self.channels,)

        self.cond_drop_prob          = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.

        self.normalize_img   = normalize_neg_one_to_one  # if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one   # if auto_normalize_img else identity

        self.dynamic_thresholding            = (dynamic_thresholding,)
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

        self.p2_loss_weight_k     = p2_loss_weight_k
        self.p2_loss_weight_gamma = (p2_loss_weight_gamma,)

        self.register_buffer('_temp', torch.tensor([0.]), persistent = False)

        self.to(next(self.unets.parameters()).device)
        
    @property
    def device(self):
        return self._temp.device

    def get_unet(self, unet_number):
        assert 0 < unet_number <= len(self.unets)
        index = unet_number - 1
        return self.unets[index]

    @contextmanager
    def one_unet_in_gpu(self, unet_number = None, unet = None):
        assert (unet_number is not None) ^ (unet is not None)

        if unet_number is not None:
            unet = self.get_unet(unet_number)

        self.cuda()

        devices = [module_device(unet) for unet in self.unets]
        self.unets.cpu()
        unet.cuda()

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)

    # ===============================================================================================================================
    # ===================== FUNCTION FOR MEAN VARIANCE ==============================================================================
    # ===============================================================================================================================

    def mean_variance(self, unet, x, t, t_next = None, text_embeds = None, text_mask = None, cond_images = None, cond_scale = 1.,
                      lowres_cond_img = None, lowres_noise_times = None, noise_scheduler=None, pred_objective = 'noise'):
        
        assert not (cond_scale != 1. and not self.can_classifier_guidance)

        pred    = default(None, lambda: unet.forward_with_cond_scale(x, noise_scheduler.get_condition(t), text_embeds = text_embeds,
                                                       text_mask=text_mask, cond_images=cond_images,
                                                       cond_scale=cond_scale, lowres_cond_img=lowres_cond_img,
                                                       lowres_noise_times=self.lowres_noise_schedule.get_condition(lowres_noise_times)))

        x_start = noise_scheduler.predict_start_from_noise(x, t = t, noise = pred) if pred_objective == 'noise' else pred

        s = torch.quantile(rearrange(x_start, 'b ... -> b (...)').abs(), self.dynamic_thresholding_percentile, dim = -1)
        s.clamp_(min = 1.)
        s = right_pad_dims_to(x_start, s)
        x_start = x_start.clamp(-s, s) / s

        return noise_scheduler.q_posterior(x_start=x_start, x_t=x, t=t, t_next=t_next)

    # ===============================================================================================================================
    # ===================== LOOP FUNCTION FOR OBTAIN AN IMAGE =======================================================================
    # ===============================================================================================================================
    
    @torch.no_grad()
    def sample_loop(self, unet, shape, text_embeds = None, text_mask = None, cond_images = None,
                    cond_scale = 1, lowres_cond_img = None, lowres_noise_times = None,
                    noise_scheduler=None, pred_objective = 'noise'):
        
        device, batch      = self.device, shape[0],
        img                = torch.randn(shape, device = device)
        lowres_cond_img    = lowres_cond_img if lowres_cond_img is None else self.normalize_img(lowres_cond_img)
        timesteps          = noise_scheduler.get_sampling_timesteps(batch)

        for times, times_next in tqdm(timesteps, desc='Obtendo a imagem ...', total = len(timesteps)):

            b = img.shape[0]

            model_mean, _, model_log_variance = self.mean_variance(unet, x = img, t = times, t_next=times_next, text_embeds = text_embeds,
                                                                   text_mask = text_mask, cond_images = cond_images, cond_scale = cond_scale,
                                                                   lowres_cond_img = lowres_cond_img, lowres_noise_times = lowres_noise_times,
                                                                   noise_scheduler = noise_scheduler, pred_objective = pred_objective)

            noise = torch.randn_like(img, device=device)

            is_last_sampling_timestep = (times_next == 0) if isinstance(noise_scheduler, GaussianDiffusion) else (times == 0)
            nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(b, *((1,) * (len(img.shape) - 1)))

            img = model_mean + nonzero_mask.to(device) * (0.5 * model_log_variance).exp() * noise

        img.clamp_(-1., 1.)
        return self.unnormalize_img(img)

    # ===============================================================================================================================
    # ===================== FUNCTION FOR OBTAIN AN IMAGE ============================================================================
    # ===============================================================================================================================

    @torch.no_grad()
    @eval_decorator
    def sample(self, texts: List[str] = None, cond_images = None, batch_size = 1,cond_scale = 1.,
               lowres_sample_noise_level = None, device = 'cpu'):

        # NECESSÁRIO CORREÇÃO
        text_embeds, text_masks = self.text_encoder.textEncoder(texts)
        text_embeds, text_masks = map(lambda t: t.to(device), (text_embeds, text_masks))
            
        batch_size = text_embeds.shape[0]

        assert not (text_embeds.shape[-1] != self.text_embed_dim)

        outputs = []

        is_cuda = next(self.parameters()).is_cuda
        device  = next(self.parameters()).device

        lowres_sample_noise_level = lowres_sample_noise_level if lowres_sample_noise_level is not None else self.lowres_sample_noise_level

        context = self.one_unet_in_gpu(unet=self.unets[0]) if is_cuda else null_context()

        with context:
            lowres_cond_img = lowres_noise_times = None
            shape = (batch_size, self.channels, self.image_sizes[0], self.image_sizes[0])

            img = self.sample_loop(self.unets[0], shape, text_embeds = text_embeds, text_mask = text_masks, cond_images = cond_images,
                                   cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_times = lowres_noise_times,
                                   noise_scheduler = self.noise_schedulers[0], pred_objective = self.pred_objectives[0])

            outputs.append(img)

        return outputs[-1]
            
    def loss_cal(self, unet, x_start, times, text_embeds = None, text_mask = None, noise_scheduler=None,
                 noise = None, pred_objective = 'noise', p2_loss_weight_gamma = 0.):
        
        
        noise = noise if noise is not None else torch.randn_like(x_start)
        
        x_start          = self.normalize_img(x_start)
        x_noisy, log_snr = noise_scheduler.q_sample(x_start = x_start, t = times, noise = noise)
        
        pred   = unet.forward(x_noisy, noise_scheduler.get_condition(times), text_embeds = text_embeds, text_mask = text_mask)  
        target = noise if pred_objective == 'noise' else x_start
        
        losses = reduce(self.loss_fn(pred, target, reduction = 'none'), 'b ... -> b', 'mean')

        if p2_loss_weight_gamma > 0:
            losses = losses*((self.p2_loss_weight_k + log_snr.exp()) ** -p2_loss_weight_gamma)

        return losses.mean()
    
    def forward(self, images, texts: List[str], device='cpu'):  
     
        unet                 = self.unets[0]
        noise_scheduler      = self.noise_schedulers[0]
        p2_loss_weight_gamma = self.p2_loss_weight_gamma[0]
        pred_objective       = self.pred_objectives[0]
        target_image_size    = self.image_sizes[0]
        b                    = images.shape[0]

        times = noise_scheduler.sample_random_times(b)

        # NECESSÁRIO CORREÇÃO
        text_embeds, text_masks = self.text_encoder.textEncoder(texts)
        text_embeds, text_masks = map(lambda t: t.to(images.device), (text_embeds, text_masks))

        assert not (text_embeds.shape[-1] != self.text_embed_dim), f'invalid text embedding dimension'

        images = resize_image_to(images, target_image_size)
        
        return self.loss_cal(unet, images, times, text_embeds=text_embeds, text_mask=text_masks, noise_scheduler=noise_scheduler,
                             noise=None, pred_objective=pred_objective, p2_loss_weight_gamma=p2_loss_weight_gamma)


        