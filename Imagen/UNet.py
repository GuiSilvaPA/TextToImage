import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many



class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

#######################################################################
#################### CLASSES FOR TIME CONDITIONING ####################
#######################################################################

class SinusoidalPositionEmbedding(nn.Module):

    def __init__(self, dim):
        
        super(SinusoidalPositionEmbedding, self).__init__()
        assert (dim % 2) == 0
        self.weights = nn.Parameter(torch.randn(dim // 2))

    def forward(self, x):
        
        x = rearrange(x, 'b -> b 1')    
        f = x * rearrange(self.weights, 'd -> 1 d') 
        w = f * 2 * torch.pi
        
        return torch.cat((x, torch.sin(w), torch.cos(w)), dim = -1)

class TimeConditioning(nn.Module):
    
    def __init__(self, unet_dim, time_embedding_dim=16, num_time_tokens = 2):
        super(TimeConditioning, self).__init__()
        
        self.to_time_hiddens = nn.Sequential(SinusoidalPositionEmbedding(time_embedding_dim),
                                             nn.Linear(time_embedding_dim+1, unet_dim*4),
                                             Swish())

        self.to_time_cond = nn.Linear(unet_dim*4, unet_dim*4)

        self.to_time_tokens = nn.Sequential(nn.Linear(unet_dim*4, unet_dim * num_time_tokens),
                                            Rearrange('b (r d) -> b r d', r = num_time_tokens))
        
    def forward(self, time):
        
        time_hiddens = self.to_time_hiddens(time)
        
        time_tokens = self.to_time_tokens(time_hiddens)
        t           = self.to_time_cond(time_hiddens)
        
        return t, time_tokens


#######################################################################
#################### CLASSES FOR TEXT CONDITIONING ####################
#######################################################################