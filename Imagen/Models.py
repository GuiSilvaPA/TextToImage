import torch
import torch.nn as nn

from transformers import T5Tokenizer, T5EncoderModel

import math
from typing import Optional, Tuple, Union, List

MAX_LENGTH = 256

class TextEncoderT5Based():
    
    def __init__(self, name = 'google/t5-v1_1-small', device='cpu'):
        
        self.device    = device
        self.model     = T5EncoderModel.from_pretrained(name).to(device)
        self.tokenizer = T5Tokenizer.from_pretrained(name)
        
    def textEncoder(self, texts):
        
        text_encoded = self.tokenizer.batch_encode_plus(texts, return_tensors = "pt", padding = 'longest',
                                                        max_length = MAX_LENGTH, truncation = True)
        
        text_ids = text_encoded.input_ids.to(self.device)
        mask     = text_encoded.attention_mask.to(self.device)
        
        self.model.eval()
        
        with torch.no_grad(): encoded_text = self.model(text_ids, mask).last_hidden_state.detach()
                
        return encoded_text, mask.bool()


class Swish(nn.Module):

    def forward(self, x):
        return x * torch.sigmoid(x)


class TimeEmbedding(nn.Module):
    
    def __init__(self, time_dim : int, device = 'cpu'):
        super(TimeEmbedding, self).__init__()
        
        self.time_dim = time_dim
        if time_dim >= 8 and time_dim < 16:
            raise ValueError(f'time_dim must be: time_dim < 8 or time_dim >= 16')

        self.half_dim = time_dim // 8
        
        ara = torch.arange(self.half_dim, device=device)
        div = -(math.log(10_000) / (self.half_dim - 1))
        
        self.emb = torch.exp(ara * div)
        
        # Layers
        
        self.linear1 = nn.Linear(self.time_dim // 4, self.time_dim)
        self.linear2 = nn.Linear(self.time_dim, self.time_dim)
        
        self.swish   = Swish()
        
    def forward(self, x):
        
        out = x[:, None] * self.emb[None, :]
        out = torch.cat((out.sin(), out.cos()), dim=1)
        out = self.swish(self.linear1(out))
        
        return self.linear2(out)

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, time_dim):
        super(DoubleConv, self).__init__()
        
        self.conv1 = nn.Sequential(      
            nn.BatchNorm2d(in_channels),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            Swish())
        
        self.conv2 = nn.Sequential( 
            nn.BatchNorm2d(out_channels),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            Swish())
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))
        else:
            self.shortcut = nn.Identity()
            
        self.linear = nn.Linear(time_dim, out_channels)
        
    def forward(self, x, t):
        
        out = self.conv1(x)
        out += self.linear(t)[:, :, None, None]
        out = self.conv2(out)
        
        
        return out + self.shortcut(x)

class ImageAttention(nn.Module):
    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None):

        
        super(ImageAttention, self).__init__()

        self.n_heads = n_heads
        self.d_k     = d_k if d_k is not None else n_channels
        self.scale   = self.d_k ** -0.5

        self.linear_layer = nn.Linear(n_channels, n_heads * self.d_k * 3)
        self.output_layer = nn.Linear(n_heads * self.d_k, n_channels)
        

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):

        batch_size, n_channels, height, width = x.shape
        
        _ = t
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
        
        QKV = self.linear_layer(x)
        QKV = QKV.view(batch_size, -1, self.n_heads, 3 * self.d_k)
        
        Q, K, V = torch.chunk(QKV, 3, dim=-1)
        
        att = (torch.einsum('BIHD, BJHD -> BIJH', Q, K) * self.scale).softmax(dim=1)
        
        out = torch.einsum('BIJH, BJHD -> BIHD', att, V)
        out = out.reshape(batch_size, -1, self.n_heads*self.d_k)
        out = self.output_layer(out)
        out += x
        
        return out.permute(0, 2, 1).view(batch_size, n_channels, height, width)

class UNetEncoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, time_dim, att):
        super(UNetEncoder, self).__init__()
        
        self.conv = DoubleConv(in_channels, out_channels, time_dim)        
        self.att  = ImageAttention(out_channels) if att else nn.Identity()
        
    def forward(self, x, t):
        return self.att(self.conv(x, t))

class UNetBottleneck(nn.Module):
    
    def __init__(self, in_channels, time_dim):
        super(UNetBottleneck, self).__init__()
        
        self.conv1 = DoubleConv(in_channels, in_channels, time_dim) 
        self.conv2 = DoubleConv(in_channels, in_channels, time_dim) 
        self.att   = ImageAttention(in_channels)
        
    def forward(self, x, t):
        out = self.att(self.conv1(x, t))
        return self.conv2(out, t)

class UNetDecoder(nn.Module):
    
    def __init__(self, in_channels, out_channels, time_dim, att):
        super(UNetDecoder, self).__init__()
        
        self.conv = DoubleConv(in_channels + out_channels, out_channels, time_dim)        
        self.att  = ImageAttention(out_channels) if att else nn.Identity()
        
    def forward(self, x, t):
        return self.att(self.conv(x, t))

class Downsample(nn.Module):

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, (3, 3), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)

class Upsample(nn.Module):

    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, (4, 4), (2, 2), (1, 1))

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        _ = t
        return self.conv(x)


class UNet(nn.Module):
    
    def __init__(self, image_channels: int = 3, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 2, 4),
                 is_attn: Union[Tuple[bool, ...], List[int]] = (False, False, True, True),
                 n_blocks: int = 2, device = 'cpu'):
       
        super().__init__()
        n_resolutions = len(ch_mults)
        
        self.downs = nn.ModuleList()
        self.ups   = nn.ModuleList()
        
        out_channels = in_channels = n_channels
        
        self.image_transform = nn.Conv2d(image_channels, n_channels, kernel_size=(3, 3), padding=(1, 1))
        self.time_emb        = TimeEmbedding(n_channels * 4, device)
        self.norm            = nn.BatchNorm2d(n_channels)
        self.act             = Swish()
        self.output_layer    = nn.Conv2d(in_channels, image_channels, kernel_size=(3, 3), padding=(1, 1))
        

        # ENCODER ===========================
        
        for i in range(n_resolutions):
            
            out_channels = in_channels * ch_mults[i] 

            for _ in range(n_blocks):
                self.downs.append(UNetEncoder(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            
            if i < n_resolutions - 1:
                self.downs.append(Downsample(in_channels))
                
        # BOTTLENECK ========================

        self.bottleneck = UNetBottleneck(out_channels, n_channels * 4)

        # DECODER ===========================
        
        in_channels = out_channels
        
        for i in reversed(range(n_resolutions)):
            
            out_channels = in_channels
            
            for _ in range(n_blocks):
                self.ups.append(UNetDecoder(in_channels, out_channels, n_channels * 4, is_attn[i]))
                
            out_channels = in_channels // ch_mults[i]
            self.ups.append(UNetDecoder(in_channels, out_channels, n_channels * 4, is_attn[i]))
            in_channels = out_channels
            
            
            if i > 0:
                self.ups.append(Upsample(in_channels))
                
        
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):

        t = self.time_emb(t)
        x = self.image_transform(x)

        skip_connec = [x]
        
        # ENCODER
        for down in self.downs:
            x = down(x, t)
            skip_connec.append(x)

        # BOTTLENECK
        x = self.bottleneck(x, t)

        # DECODER
        for up in self.ups:
            if isinstance(up, Upsample):
                x = up(x, t)
            else:
                s = skip_connec.pop()
                x = torch.cat((x, s), dim=1)
                x = up(x, t)

        return self.output_layer(self.act(self.norm(x)))


if __name__=='__main__':

    x = torch.randn(10, 3, 32, 32)
    t = torch.tensor([50], dtype=torch.long)

    unet = UNet()

    model_output = unet(x, t)
    
    print(f'\nInput:  {x.shape}')
    print(f'Output: {model_output.shape}\n')

