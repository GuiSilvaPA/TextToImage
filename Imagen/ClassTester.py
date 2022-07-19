from regex import T
import torch
from BasicModels import MasterPerceiver
from ComplexModels import TextConditioning, TimeConditioning, UNet
from TextEncoder import TextEncoderT5Based
from Imagen import Imagen

""" time_cond = TimeConditioning(dim=20, cond_dim=20, time_embedding_dim=10, num_time_tokens = 68)

text_cond = TextConditioning(dim=20, cond_dim=20, text_embed_dim=10, dim_head=64, heads=8,
                             num_latents=64, max_text_len=15, Ttype=torch.float, device='cpu')

unet = UNet(dim = 8, cond_dim = 40, text_embed_dim = 40, dim_mults = (1, 2, 4, 8), num_resnet_blocks = 3,
            layer_attns = (False, True, True, True), layer_cross_attns = (False, True, True, True))


text_embeds = torch.randn(4, 15, 10)
text_masks  = torch.ones(4, 15).bool()
time        = torch.rand(4)

time_tokens, t = time_cond(time)
print(time_tokens.shape, t.shape)

text_tokens, text_hiddens = text_cond(text_embeds, text_masks)
print(text_tokens.shape, text_hiddens.shape)

print(TextEncoderT5Based().embed_dim)

x           = torch.randn(4, 3, 32, 32)
text_embeds = torch.randn(4, 32, 40)
text_masks  = torch.ones(4, 32).bool()
time        = torch.rand(4) """

# x, c, t = unet(x, time, text_embeds, text_masks)

""" x = unet(x, time, text_embeds, text_masks)

print(x.shape) """
# print(c.shape)
# print(t.shape)

unet1 = UNet(dim = 8, cond_dim = 40, text_embed_dim = 512, dim_mults = (1, 2, 4, 8), num_resnet_blocks = 3,
             layer_attns = (False, True, True, True), layer_cross_attns = (False, True, True, True))

'''img = Imagen((unet1,),
        image_sizes=(32,),                             
        text_encoder_name = 'google/t5-v1_1-small',
        text_embed_dim = None,
        channels = 3,
        timesteps = 10,
        cond_drop_prob = 0.1,
        loss_type = 'l2',
        noise_schedules = 'cosine',
        pred_objectives = 'noise',
        lowres_noise_schedule = 'linear',
        lowres_sample_noise_level = 0.2,           
        per_sample_random_aug_noise_level = False,  
        condition_on_text = True,
        auto_normalize_img = True,               
        continuous_times = True,
        p2_loss_weight_gamma = 0.5,              
        p2_loss_weight_k = 1,
        dynamic_thresholding = True,
        dynamic_thresholding_percentile = 0.9)'''

img = Imagen((unet1,), (32,), text_encoder_name = 'google/t5-v1_1-small', channels = 3, timesteps = 10,
             cond_drop_prob = 0.1, noise_schedules = 'cosine', pred_objectives = 'noise', lowres_noise_schedule = 'linear',
             lowres_sample_noise_level = 0.2, per_sample_random_aug_noise_level = False,  p2_loss_weight_gamma = 0.5,
             p2_loss_weight_k = 1, dynamic_thresholding = True, dynamic_thresholding_percentile = 0.9, device='cpu')

print('\n ============ VARIAVEIS ============ ')

x           = torch.randn(4, 3, 32, 32)
text_embeds = torch.randn(4, 32, 40)
text_masks  = torch.ones(4, 32).bool()

print('\n ============== SAIDA ============== \n')

# loss = img(x, text_embeds = text_embeds, text_masks = text_masks, unet_number = 0)
texts = ['I love you so much', 'I love you so much', 'I love you so much', 'I love you so much']
loss = img(x, texts=texts, device='cpu')
print(loss)

images = img.sample(texts = ['a whale breaching from afar',
                             'young girl blowing out candles on her birthday cake',
                             'fireworks with blue and green sparkles'], cond_scale = 3.)

print(images.shape) # (3, 3, 256, 256)

with torch.no_grad():
        texts = ['I love you so much', 'I love you so much', 'I love you so much', 'I love you so much']
        loss = img(x, texts=texts, device='cpu')
        print(loss)

