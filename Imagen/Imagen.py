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

from tqdm import tqdm

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

class Imagen(nn.Module):
    def __init__(
        self,
        unets,
        *,
        image_sizes,                                # for cascading ddpm, image size at each stage
        text_encoder_name = 'google/t5-v1_1-small',
        text_embed_dim = None,
        channels = 3,
        timesteps = 1000,
        cond_drop_prob = 0.1,
        loss_type = 'l2',
        noise_schedules = 'cosine',
        pred_objectives = 'noise',
        lowres_noise_schedule = 'linear',
        lowres_sample_noise_level = 0.2,            # in the paper, they present a new trick where they noise the lowres conditioning image, and at sample time, fix it to a certain level (0.1 or 0.3) - the unets are also made to be conditioned on this noise level
        per_sample_random_aug_noise_level = False,  # unclear when conditioning on augmentation noise level, whether each batch element receives a random aug noise value - turning off due to @marunine's find
        condition_on_text = True,
        auto_normalize_img = True,                  # whether to take care of normalizing the image from [0, 1] to [-1, 1] and back automatically - you can turn this off if you want to pass in the [-1, 1] ranged image yourself from the dataloader
        continuous_times = True,
        p2_loss_weight_gamma = 0.5,                 # p2 loss weight, from https://arxiv.org/abs/2204.00227 - 0 is equivalent to weight of 1 across time
        p2_loss_weight_k = 1,
        dynamic_thresholding = True,
        dynamic_thresholding_percentile = 0.9,      # unsure what this was based on perusal of paper
    ):
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

        n_unets   = len(unets)
        timesteps = (timesteps,)*n_unets  
        
        noise_schedules = (noise_schedules, 'cosine')
        mults           = n_unets - len(noise_schedules) if n_unets - len(noise_schedules) > 0 else 0
        noise_schedules = (*noise_schedules, *('linear',)*mults)

        self.lowres_noise_schedule = GaussianDiffusion(noise_type=lowres_noise_schedule)
        self.pred_objectives       = (pred_objectives,)*n_unets

        self.text_encoder_name = text_encoder_name
        self.text_embed_dim    = TextEncoderT5Based(text_encoder_name).embed_dim
        self.text_encoder      = TextEncoderT5Based(text_encoder_name)     

        self.noise_schedulers = nn.ModuleList([])
        for timestep, noise_schedule in zip(timesteps, noise_schedules):
            noise_scheduler = GaussianDiffusion(noise_type=noise_schedule, timesteps=timestep)
            self.noise_schedulers.append(noise_scheduler)
            
        self.unets = nn.ModuleList([])        
        for i, current_unet in enumerate(unets):
            self.unets.append(current_unet.lowres_change(lowres_cond = not (i == 0)))

        self.sample_channels = (self.channels,)*n_unets

        lowres_conditions = tuple([t.lowres_cond for t in self.unets])
        # assert lowres_conditions == (False, *((True,) * (n_unets - 1)))

        self.cond_drop_prob          = cond_drop_prob
        self.can_classifier_guidance = cond_drop_prob > 0.

        self.normalize_img   = normalize_neg_one_to_one  # if auto_normalize_img else identity
        self.unnormalize_img = unnormalize_zero_to_one   # if auto_normalize_img else identity


        self.dynamic_thresholding            = (dynamic_thresholding,)*n_unets
        self.dynamic_thresholding_percentile = dynamic_thresholding_percentile

        self.p2_loss_weight_k     = p2_loss_weight_k
        self.p2_loss_weight_gamma = (p2_loss_weight_gamma,)*n_unets

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
        assert unet_number is not None ^ unet is not None

        if unet_number is not None:
            unet = self.get_unet(unet_number)

        self.cuda()

        devices = [module_device(unet) for unet in self.unets]
        self.unets.cpu()
        unet.cuda()

        yield

        for unet, device in zip(self.unets, devices):
            unet.to(device)

    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################

    def p_mean_variance(self, unet, x, t, t_next = None, text_embeds = None, text_mask = None,
                        cond_images = None, cond_scale = 1., lowres_cond_img = None,
                        lowres_noise_times = None, noise_scheduler=None,  
                        pred_objective = 'noise', dynamic_threshold = True, model_output = None):
        
        assert not (cond_scale != 1. and not self.can_classifier_guidance), 'imagen was not trained with conditional dropout, and thus one cannot use classifier free guidance (cond_scale anything other than 1)'

        pred = default(model_output, lambda: unet.forward_with_cond_scale(x, noise_scheduler.get_condition(t), text_embeds = text_embeds, text_mask = text_mask, cond_images = cond_images, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_times = self.lowres_noise_schedule.get_condition(lowres_noise_times)))

        x_start = noise_scheduler.predict_start_from_noise(x, t = t, noise = pred) if pred_objective == 'noise' else pred

        if dynamic_threshold:
            # following pseudocode in appendix
            # s is the dynamic threshold, determined by percentile of absolute values of reconstructed sample per batch element
            s = torch.quantile(rearrange(x_start, 'b ... -> b (...)').abs(),
                               self.dynamic_thresholding_percentile,
                               dim = -1)

            s.clamp_(min = 1.)
            s = right_pad_dims_to(x_start, s)
            x_start = x_start.clamp(-s, s) / s
        else:
            x_start.clamp_(-1., 1.)

        return noise_scheduler.q_posterior(x_start = x_start, x_t = x, t = t, t_next = t_next)
    
    @torch.no_grad()
    def p_sample(self, unet, x, t, t_next = None, text_embeds = None, text_mask = None,
                 cond_images = None, cond_scale = 1., lowres_cond_img = None,
                 lowres_noise_times = None, noise_scheduler = None,
                 pred_objective = 'noise', dynamic_threshold = True):
        
        # b, *_, device = *x.shape, x.device
        b, device = x.shape[0], x.device
        
        model_mean, _, model_log_variance = self.p_mean_variance(unet, x = x, t = t, t_next = t_next, text_embeds = text_embeds, text_mask = text_mask,
                                                                 cond_images = cond_images, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img,
                                                                 lowres_noise_times = lowres_noise_times, noise_scheduler = noise_scheduler,
                                                                 pred_objective = pred_objective, dynamic_threshold = dynamic_threshold)
        
        noise = torch.randn_like(x)

        is_last_sampling_timestep = (t_next == 0) if isinstance(noise_scheduler, GaussianDiffusion) else (t == 0)
        nonzero_mask = (1 - is_last_sampling_timestep.float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
    
    @torch.no_grad()
    def p_sample_loop(self, unet, shape, text_embeds = None, text_mask = None, cond_images = None,
                      cond_scale = 1, lowres_cond_img = None, lowres_noise_times = None,
                      noise_scheduler=None, pred_objective = 'noise', dynamic_threshold = True):
        
        device = self.device
        batch  = shape[0]
        img    = torch.randn(shape, device = device)
        
        lowres_cond_img = lowres_cond_img if lowres_cond_img is None else self.normalize_img(lowres_cond_img)
        timesteps       = noise_scheduler.get_sampling_timesteps(batch)

        for times, times_next in tqdm(timesteps, desc = 'sampling loop time step', total = len(timesteps)):
            
            img = self.p_sample(unet, img, times, t_next = times_next, text_embeds = text_embeds, text_mask = text_mask,
                                cond_images = cond_images, cond_scale = cond_scale, lowres_cond_img = lowres_cond_img,
                                lowres_noise_times = lowres_noise_times, noise_scheduler = noise_scheduler, 
                                pred_objective = pred_objective, dynamic_threshold = dynamic_threshold)

        img.clamp_(-1., 1.)
        unnormalize_img = self.unnormalize_img(img)
        return unnormalize_img
    
    @torch.no_grad()
    @eval_decorator
    def sample(self, texts: List[str] = None, cond_images = None, batch_size = 1,cond_scale = 1.,
               lowres_sample_noise_level = None, stop_at_unet_number = None, return_all_unet_outputs = False,
               return_pil_images = False, device = 'cpu'):

        text_embeds, text_masks = self.text_encoder.textEncoder(texts)

        # NECESSÁRIO CORREÇÃO

        # text_embeds = [t.to(device) for t in text_embeds]
        # text_masks  = [t.to(device) for t in text_masks]
        text_embeds, text_masks = map(lambda t: t.to(device), (text_embeds, text_masks))
            
        batch_size = text_embeds.shape[0]

        assert not (text_embeds.shape[-1] != self.text_embed_dim)

        outputs = []

        is_cuda = next(self.parameters()).is_cuda
        device  = next(self.parameters()).device

        lowres_sample_noise_level = lowres_sample_noise_level if lowres_sample_noise_level is not None else self.lowres_sample_noise_level

        for unet_number, unet, channel, image_size, noise_scheduler, pred_objective, dynamic_threshold in tqdm(zip(range(1, len(self.unets) + 1), self.unets, self.sample_channels, self.image_sizes, self.noise_schedulers, self.pred_objectives, self.dynamic_thresholding)):

            context = self.one_unet_in_gpu(unet = unet) if is_cuda else null_context()

            with context:
                lowres_cond_img = lowres_noise_times = None
                shape = (batch_size, channel, image_size, image_size)

                if unet.lowres_cond:
                    lowres_noise_times = self.lowres_noise_schedule.get_times(batch_size, lowres_sample_noise_level, device = device)

                    lowres_cond_img = resize_image_to(img, image_size)
                    lowres_cond_img, _ = self.lowres_noise_schedule.q_sample(x_start = lowres_cond_img, t = lowres_noise_times, noise = torch.randn_like(lowres_cond_img))

                shape = (batch_size, self.channels, image_size, image_size)

                img = self.p_sample_loop(unet, shape, text_embeds = text_embeds, text_mask = text_masks, cond_images = cond_images,
                                         cond_scale = cond_scale, lowres_cond_img = lowres_cond_img, lowres_noise_times = lowres_noise_times,
                                         noise_scheduler = noise_scheduler, pred_objective = pred_objective, dynamic_threshold = dynamic_threshold)

                outputs.append(img)

            if stop_at_unet_number is not None and stop_at_unet_number == unet_number:
                break

        output_index = -1 if not return_all_unet_outputs else slice(None) # either return last unet output or all unet outputs

        if not return_pil_images:
            return outputs[output_index]

        if not return_all_unet_outputs:
            outputs = outputs[-1:]

        pil_images = list(map(lambda img: list(map(T.ToPILImage(), img.unbind(dim = 0))), outputs))

        return pil_images[output_index]


    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
    ##################################################################################################################
            
    def p_losses(self, unet, x_start, times, text_embeds = None, text_mask = None, cond_images = None,
                 noise_scheduler=None,
                 lowres_cond_img = None, lowres_aug_times = None, 
                 noise = None, times_next = None,
                 pred_objective = 'noise', p2_loss_weight_gamma = 0.):
        
        
        noise = noise if noise is not None else torch.randn_like(x_start)

        # get x_t
        x_start          = self.normalize_img(x_start)
        x_noisy, log_snr = noise_scheduler.q_sample(x_start = x_start, t = times, noise = noise)
        
        lowres_cond_img_noisy = None
        if lowres_cond_img is not None:      
            lowres_cond_img  = self.normalize_img(lowres_cond_img)            
            lowres_aug_times = lowres_aug_times if lowres_aug_times is not None else times
            
            lowres_cond_img_noisy, _ = self.lowres_noise_schedule.q_sample(x_start = lowres_cond_img, t = lowres_aug_times, noise = torch.randn_like(lowres_cond_img))
            
        pred = unet.forward(x_noisy,
                            noise_scheduler.get_condition(times),
                            text_embeds = text_embeds,
                            text_mask = text_mask)  

                            # cond_images = cond_images,
                            # lowres_noise_times = self.lowres_noise_schedule.get_condition(lowres_aug_times),
                            # lowres_cond_img = lowres_cond_img_noisy,
                            # cond_drop_prob = self.cond_drop_prob
            
        target = noise if pred_objective == 'noise' else x_start
        
        losses = self.loss_fn(pred, target, reduction = 'none')
        losses = reduce(losses, 'b ... -> b', 'mean')

        # p2 loss reweighting

        if p2_loss_weight_gamma > 0:
            loss_weight = (self.p2_loss_weight_k + log_snr.exp()) ** -p2_loss_weight_gamma
            losses = losses * loss_weight

        return losses.mean()
    
    def forward(self, images, texts: List[str], unet_number, device, cond_images = None):  

        unet_index = unet_number - 1        
        unet       = self.unets[unet_index]

        noise_scheduler      = self.noise_schedulers[unet_index]
        p2_loss_weight_gamma = self.p2_loss_weight_gamma[unet_index]
        pred_objective       = self.pred_objectives[unet_index]
        target_image_size    = self.image_sizes[unet_index]
        prev_image_size      = self.image_sizes[unet_index - 1] if unet_index > 0 else None
        b, c, h, w, device,  = *images.shape, images.device

        times = noise_scheduler.sample_random_times(b)

        text_embeds, text_masks = self.text_encoder.textEncoder(texts)

        # NECESSÁRIO CORREÇÃO

        # text_embeds = [t.to(images.device) for t in text_embeds]
        # text_masks  = [t.to(images.device) for t in text_masks]
        text_embeds, text_masks = map(lambda t: t.to(images.device), (text_embeds, text_masks))

        assert not (text_embeds.shape[-1] != self.text_embed_dim), f'invalid text embedding dimension'

        lowres_cond_img = lowres_aug_times = None
        if prev_image_size is not None:      
            lowres_cond_img = resize_image_to(resize_image_to(images, prev_image_size), target_image_size)
            lowres_aug_time = repeat(self.lowres_noise_schedule.sample_random_times(1, device = device), '1 -> b', b = b)


        images = resize_image_to(images, target_image_size)
        
        '''p_losses(self, unet, x_start, times, text_embeds = None, text_mask = None, cond_images = None,
                    noise_scheduler,
                    lowres_cond_img = None, lowres_aug_times = None, 
                    noise = None, times_next = None,
                    pred_objective = 'noise', p2_loss_weight_gamma = 0.)'''

        return self.p_losses(unet, images, times, text_embeds=text_embeds, text_mask=text_masks,cond_images=cond_images,
                             noise_scheduler = noise_scheduler,
                             lowres_cond_img = lowres_cond_img, lowres_aug_times = lowres_aug_times,
                             pred_objective = pred_objective, p2_loss_weight_gamma = p2_loss_weight_gamma)


        