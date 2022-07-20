import torch
import torch.nn as nn
import torch.nn.functional as F
import BasicModels as BM

from TextEncoder import TextEncoderT5Based

from einops.layers.torch import Rearrange
from einops import rearrange



######################################################################################################
################ Time Conditioning ###################################################################
######################################################################################################

class TimeConditioning(nn.Module):
    
    def __init__(self, dim, cond_dim, time_embedding_dim=16, num_time_tokens = 2):
        super(TimeConditioning, self).__init__()
        
        self.to_time_hiddens = nn.Sequential(BM.SinusoidalPositionEmbedding(time_embedding_dim),
                                             nn.Linear(time_embedding_dim+1, dim*4),
                                             BM.Swish())

        self.to_time_cond = nn.Linear(dim*4, dim*4)

        self.to_time_tokens = nn.Sequential(nn.Linear(dim*4, cond_dim*num_time_tokens),
                                            Rearrange('b (r d) -> b r d', r = num_time_tokens))
        
    def forward(self, time):
        
        time_hiddens = self.to_time_hiddens(time)
        
        time_tokens = self.to_time_tokens(time_hiddens)
        t           = self.to_time_cond(time_hiddens)
        
        return time_tokens, t

######################################################################################################
################ Text Conditioning ###################################################################
######################################################################################################

class TextConditioning(nn.Module):
    
    def __init__(self, dim, cond_dim, text_embed_dim, dim_head, heads, num_latents, max_text_len, Ttype, device='cpu'):
        super(TextConditioning, self).__init__()
        
        self.max_text_len = max_text_len
        self.Ttype        = Ttype
        self.device       = device
        
        self.text_to_cond = nn.Linear(text_embed_dim, cond_dim)
        self.attention    = BM.MasterPerceiver(dim=cond_dim, depth=2, dim_head=dim_head, heads=heads, num_latents=num_latents,
                                               num_latents_mean_pooled=4, max_seq_len=512, ff_mult=4)    
        
        self.null_text_embed  = nn.Parameter(torch.randn(1, max_text_len, cond_dim))
        self.null_text_hidden = nn.Parameter(torch.randn(1, dim*4))
        
        self.text_to_hiddens = nn.Sequential(nn.LayerNorm(cond_dim),
                                             nn.Linear(cond_dim, dim*4),
                                             BM.Swish(),
                                             nn.Linear(dim*4, dim*4))
        
    def forward(self, text_embeds, text_mask=None):
        
        b = text_embeds.shape[0]

        # Making Masks
        
        text_keep_mask        = torch.zeros((b,), device = self.device).float().uniform_(0, 1) < 0.9
        text_keep_mask_embed  = text_keep_mask[:, None, None]
        text_keep_mask_hidden = text_keep_mask[:, None]
        
        # Text Tokens
        
        text_tokens = self.text_to_cond(text_embeds)[:, :self.max_text_len]
        
        remainder = self.max_text_len - text_tokens.shape[1]
        if remainder > 0: text_tokens = F.pad(text_tokens, (0, 0, 0, remainder))
        
        if text_mask is not None:
            if remainder > 0: text_mask = F.pad(text_mask, (0, remainder), value = False)
            
            text_mask = text_mask[:, :, None]
            text_keep_mask_embed = text_mask & text_keep_mask_embed
            
        null_text_embed = self.null_text_embed.to(text_tokens.dtype)

        text_tokens = torch.where(text_keep_mask_embed, text_tokens, null_text_embed)        
        text_tokens = self.attention(text_tokens)
        
        # Text Hiddens
        
        text_hiddens     = self.text_to_hiddens(text_tokens.mean(dim = -2))
        null_text_hidden = self.null_text_hidden.to(self.Ttype)
        text_hiddens     = torch.where(text_keep_mask_hidden, text_hiddens, null_text_hidden)
        
        
        return text_tokens, text_hiddens

######################################################################################################
################ U-Net ###############################################################################
######################################################################################################

class UNet(nn.Module):
    def __init__(self, dim, text_embed_dim = TextEncoderT5Based().embed_dim, num_resnet_blocks = 1,
                 cond_dim = None, num_time_tokens = 2, learned_sinu_pos_emb_dim = 16, dim_mults=(1, 2, 4, 8),
                 channels = 3, att_dim_head = 64, att_heads = 8, ff_mult = 2, 
                 layer_attns = True, layer_cross_attns = True, max_text_len = 256, resnet_groups = 8,
                 init_cross_embed_kernel_sizes = (3, 7, 15), att_pool_num_latents = 32,
                 use_global_context_attn = True, device='cpu'):

        self.device = device

        super(UNet, self).__init__()
        
        self.time_cond = TimeConditioning(dim                = dim,
                                          cond_dim           = cond_dim,
                                          time_embedding_dim = learned_sinu_pos_emb_dim,
                                          num_time_tokens    = num_time_tokens)

        self.text_cond = TextConditioning(dim            = dim,
                                          cond_dim       = cond_dim,
                                          text_embed_dim = text_embed_dim,
                                          dim_head       = att_dim_head,
                                          heads          = att_heads,
                                          num_latents    = att_pool_num_latents,
                                          max_text_len   = max_text_len,
                                          Ttype          = torch.float,
                                          device         = device)
        
        self.norm_cond = nn.LayerNorm(cond_dim)
        
        self.init_conv = BM.CrossEmbedding(channels, dim, stride=1, kernel_sizes=init_cross_embed_kernel_sizes)
        
        self.init_resnet_block = BM.ResnetLayer(dim, dim, cond_dim=None, time_cond_dim=dim*4,
                                                groups=resnet_groups, linear_att=False, gca=use_global_context_attn)

        # Params for UNet
        
        dims   = [dim, *[m*dim for m in dim_mults]]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        num_resnet_blocks = (num_resnet_blocks,)*len(in_out)
        resnet_groups     = (resnet_groups,)*len(in_out)
        
        assert len(num_resnet_blocks) == len(in_out), 'num_resnet_blocks and in_out must be the same size'
        assert len(resnet_groups) == len(in_out),     'resnet_groups and in_out must be the same size'
        assert len(layer_attns) == len(in_out),       'layer_attns and in_out must be the same size'
        assert len(layer_cross_attns) == len(in_out), 'layer_cross_attns and in_out must be the same size'
        
        params   = [in_out, num_resnet_blocks, resnet_groups, layer_attns, layer_cross_attns]
        r_params = [reversed(in_out), reversed(num_resnet_blocks), reversed(resnet_groups), reversed(layer_attns), reversed(layer_cross_attns)]
        
        self.downs = nn.ModuleList([])
        self.ups   = nn.ModuleList([])
        
        skip_connect_dims = []
        
        # UNet Encoder ==========================================================================================================================

        for i, ((dim_in, dim_out), resnet_n, groups, layer_attn, layer_cross_attn) in enumerate(zip(*params)):
            
            is_last = i >= (len(in_out) - 1)
            
            layer_cond_dim = cond_dim if layer_cross_attn else None                       


            # Downsample for memory efficiency
            downsample  = nn.Conv2d(dim_in, dim_out, 4, 2, 1)   
            current_dim = dim_out

            skip_connect_dims.append(current_dim) 
             
            # First Resnet
            init_resnet = BM.ResnetLayer(current_dim, current_dim, cond_dim=layer_cond_dim,
                                         time_cond_dim=dim*4, groups=groups, linear_att=False, gca=False)
            
            # Multiples Resnets
            mult_resnet = nn.ModuleList([BM.ResnetLayer(current_dim, current_dim, time_cond_dim=dim*4,
                                                        groups=groups, linear_att=False, gca=use_global_context_attn) for _ in range(resnet_n)])
            
            # Transformer Layer
            if layer_attn:
                transformerLayer = BM.TransformerLayer(dim=current_dim, heads=att_heads, dim_head=att_dim_head,
                                                       ff_mult=ff_mult, att_type='normal', context_dim=cond_dim)
            else: 
                transformerLayer = BM.Identity()
                
            # Append self.downs for Encoder
            self.downs.append(nn.ModuleList([downsample, init_resnet, mult_resnet, transformerLayer]))
            
        # UNet Bottleneck =======================================================================================================================
        
        mid_dim = dims[-1]
        
        self.mid_block1 = BM.ResnetLayer(mid_dim, mid_dim, cond_dim=cond_dim, time_cond_dim=dim*4, groups=resnet_groups[-1])
        self.mid_attn   = BM.AttentionTypes(mid_dim, heads=att_heads, dim_head=att_dim_head)
        self.mid_block2 = BM.ResnetLayer(mid_dim, mid_dim, cond_dim=cond_dim, time_cond_dim=dim*4, groups=resnet_groups[-1])
        
        # UNet Decoder ==========================================================================================================================
            
        for i, ((dim_in, dim_out), resnet_n, groups, layer_attn, layer_cross_attn) in enumerate(zip(*r_params)):
            
            layer_cond_dim = cond_dim if layer_cross_attn else None          
            
            skip_connect_dim = skip_connect_dims.pop()
            # First Resnet
            init_resnet = BM.ResnetLayer(dim_out + skip_connect_dim, dim_out, cond_dim=layer_cond_dim,
                                         time_cond_dim=dim*4, groups=groups, linear_att=False, gca=False)
            
            # Multiples Resnets
            mult_resnet = nn.ModuleList([BM.ResnetLayer(dim_out + skip_connect_dim, dim_out, time_cond_dim=dim*4,
                                                        groups=groups, linear_att=False, gca=use_global_context_attn) for _ in range(resnet_n)])
            
            # Transformer Layer
            if layer_attn:                
                transformerLayer = BM.TransformerLayer(dim=dim_out, heads=att_heads, dim_head=att_dim_head,
                                                       ff_mult=ff_mult, att_type='normal', context_dim=cond_dim)
            else: 
                transformerLayer = BM.Identity()
                
            # Upsample
            upsample = nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                                     nn.Conv2d(dim_out, dim_in, 3, padding=1))           

            # Append self.ups for Decoder
            self.ups.append(nn.ModuleList([init_resnet, mult_resnet, transformerLayer, upsample]))
            
        # Final Layers ==========================================================================================================================

        self.final_resnet = BM.ResnetLayer(dim, dim, time_cond_dim=dim*4, groups=resnet_groups[0], linear_att=False, gca=True)
        self.final_conv   = nn.Conv2d(dim, channels, 3, padding=1)


    def forward_with_cond_scale(self, *args, cond_scale = 1., **kwargs):
    
        logits = self.forward(*args, **kwargs)
        if cond_scale == 1:
            return logits
        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale
        
    def forward(self, x, time, text_embeds, text_mask):
        
        x           = x.to(self.device)
        time        = time.to(self.device)
        text_embeds = text_embeds.to(self.device)
        text_mask   = text_mask.to(self.device)


        # Time Conditioning        
        time_tokens, t = self.time_cond(time)

        # Text Conditioning   
        text_tokens, text_hiddens = self.text_cond(text_embeds, text_mask)        
        
        # Concatenating Time and Text

        c = time_tokens if text_tokens is None else torch.cat((time_tokens, text_tokens), dim = -2)
        c = self.norm_cond(c)
        
        t = t + text_hiddens
        
        # Processing Image
        
        x = self.init_conv(x)
        x = self.init_resnet_block(x, time_emb=t)
        # Encoder
        
        hiddens = []
        for downsample, init_resnet, mult_resnet, transformerLayer in self.downs:

            x = downsample(x)
            
            x = init_resnet(x, t, c)

            for resnet in mult_resnet:
                x = resnet(x, t)
                hiddens.append(x)
                
            x = transformerLayer(x, c)
            hiddens.append(x)
            
        x = self.mid_block1(x, t, c)
        
        w = x.shape[-1]
        
        x = rearrange(x, 'b c h w -> b (h w) c')        
        x = self.mid_attn(x) + x
        x = rearrange(x, 'b (h w) c -> b c h w',w=w)

        x = self.mid_block2(x, t, c)

        for init_resnet, mult_resnet, transformerLayer, upsample in self.ups:
            x = torch.cat((x, hiddens.pop() * (2 ** -0.5)), dim = 1)           
            x = init_resnet(x, t, c)

            for resnet in mult_resnet:
                x = torch.cat((x, hiddens.pop() * (2 ** -0.5)), dim = 1)
                x = resnet(x, t)

            x = transformerLayer(x, c)
            x = upsample(x)
            
        x = self.final_resnet(x)
        x = self.final_conv(x)
        
        return x
        
######################################################################################################
################ Funil for High Resolution ###########################################################
######################################################################################################

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        
        self.conv = nn.Sequential(        
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.conv(x)

class HighResolution(nn.Module):
    def __init__(self, features):
        super(HighResolution, self).__init__()
        
        self.ups = nn.ModuleList()

        for f in range(len(features)-1):
            self.ups.append(nn.ConvTranspose2d(features[f], features[f+1], kernel_size=2, stride=2))
            self.ups.append(DoubleConv(features[f+1], features[f+1]))

        self.final_conv = nn.Conv2d(features[-1], features[0], kernel_size=1)
        
    def forward(self, x):
        
        for i in range(len(self.ups)):
            x = self.ups[i](x)
        
        return self.final_conv(x)