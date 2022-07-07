import torch
from torch import einsum
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from einops_exts import rearrange_many, repeat_many



######################################################################################################
################ Tools ###############################################################################
######################################################################################################

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super(Identity, self).__init__()

    def forward(self, x, *args, **kwargs):
        return x
    
class Parallel(nn.Module):
    def __init__(self, *fns):
        super(Parallel, self).__init__()
        self.fns = nn.ModuleList(fns)

    def forward(self, x):
        outputs = [fn(x) for fn in self.fns]
        return sum(outputs)

######################################################################################################
################ Swish Activation Function ###########################################################
######################################################################################################

class Swish(nn.Module):

    def __init__(self):
        super(Swish, self).__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)

######################################################################################################
################ Sinudoidal Position Embeddings ######################################################
######################################################################################################

class SinusoidalPositionEmbedding(nn.Module):

    def __init__(self, dim):
        
        super(SinusoidalPositionEmbedding, self).__init__()
        assert (dim % 2) == 0
        self.weights = nn.Parameter(torch.randn(dim // 2))

    def forward(self, x):
        
        x = x[:, None]
        f = x * self.weights[None, :].to(x.device)
        w = f * 2 * torch.pi
        
        return torch.cat((x, torch.sin(w), torch.cos(w)), dim = -1)

######################################################################################################
################ Attention: Perceiver ################################################################
######################################################################################################

class PerceiverAttention(nn.Module):
    def __init__(self, cond_dim, dim_head = 64, heads = 8):
        super(PerceiverAttention, self).__init__()
        
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.norm_x = nn.LayerNorm(cond_dim)
        self.norm_l = nn.LayerNorm(cond_dim)

        self.Q  = nn.Linear(cond_dim, dim_head * heads, bias = False)
        self.KV = nn.Linear(cond_dim, dim_head * heads * 2, bias = False)

        self.output_layer = nn.Sequential(nn.Linear(dim_head * heads, cond_dim, bias = False),
                                          nn.LayerNorm(cond_dim))
        
    def attention(self, q, k, v, mask=None):
        
        score = torch.matmul(q, k.transpose(-1, -2))
        
        if mask is not None:            
            max_neg = -torch.finfo(score.dtype).max            
            mask    = mask[:, None, None, :]
            score   = score.masked_fill(~mask, max_neg)
            
        probs = score.softmax(dim = -1, dtype = torch.float32)
        
        return torch.matmul(probs, v)

    def forward(self, x, latents, mask = None):
        
        batch = x.shape[0]
        
        x = self.norm_x(x)
        l = self.norm_l(latents)

        q    = self.Q(l)
        k, v = self.KV(torch.cat((x, l), dim = -2)).chunk(2, dim = -1)

        q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = self.heads)

        q = q * self.scale
        
        if mask is not None:
            mask = F.pad(mask, (0, l.shape[-2]), value = True)
            att = self.attention(q, k, v, mask)
        else:
            att = self.attention(q, k, v)        

        out = rearrange(att, 'b h n d -> b n (h d)', h = self.heads)
        return self.output_layer(out)

######################################################################################################
################ Resample Function for Perceiver Attention ###########################################
######################################################################################################

class MasterPerceiver(nn.Module):
    def __init__(self, dim, depth, dim_head = 64, heads = 8, num_latents = 64,
                 num_latents_mean_pooled = 4, max_seq_len = 512, ff_mult = 4):        
        # num_latents_mean_pooled: number of latents derived from mean pooled representation of the sequence
        
        super(MasterPerceiver, self).__init__()
        
        self.embedding = nn.Embedding(max_seq_len, dim)
        self.latents   = nn.Parameter(torch.randn(num_latents, dim))

        self.Mean_Pooled = None
        if num_latents_mean_pooled > 0:
            self.Mean_Pooled = nn.Sequential(nn.LayerNorm(dim),
                                             nn.Linear(dim, dim * num_latents_mean_pooled),
                                             Rearrange('b (n d) -> b n d', n = num_latents_mean_pooled))

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([PerceiverAttention(dim, dim_head, heads),
                               nn.Sequential(nn.LayerNorm(dim),
                                             nn.Linear(dim, dim*ff_mult, bias = False),
                                             Swish(),
                                             nn.LayerNorm(dim*ff_mult),
                                             nn.Linear(dim*ff_mult, dim, bias = False))]))

    def forward(self, x, mask = None):
        
        b, n, device = x.shape[0], x.shape[1], x.device
        
        pos_emb = self.embedding(torch.arange(n, device = device))
        X = x + pos_emb

        lat = repeat(self.latents, 'n d -> b n d', b = b)

        if self.Mean_Pooled is not None:
                
            pool_mask = torch.ones(x.shape[:2], device = x.device, dtype = torch.bool)
                
            denom     = pool_mask.sum(dim = 1, keepdim = True)
            pool_mask = pool_mask[:, :, None]
            masked_x  = x.masked_fill(~pool_mask, 0.)
            
            m_seq = masked_x.sum(dim = 1) / denom.clamp(min = 1e-5)
            m_lat = self.Mean_Pooled(m_seq)
            lat   = torch.cat((m_lat, lat), dim = -2)

        for att, nn in self.layers:
            lat = nn(att(X, lat, mask = mask) + lat) + lat

        return lat

######################################################################################################
################ Different types of Attention ########################################################
######################################################################################################

class AttentionTypes(nn.Module):
    def __init__(self, dim, dim_head = 64, heads = 8, dropout = 0.05, context_dim = None, norm_context = False, att_type='normal'):
        super().__init__()
        
        self.scale         = dim_head ** -0.5
        self.heads         = heads
        self.att_type      = att_type
        self.norm_context  = norm_context
        # Normal Attention Initialization #############################################################################################################
        
        if att_type == 'normal':     
            self.norm = nn.LayerNorm(dim)

            self.null_kv = nn.Parameter(torch.randn(2, dim_head))
            self.Q       = nn.Linear(dim, dim_head * heads, bias = False)
            self.KV      = nn.Linear(dim, dim_head * 2, bias = False)

            self.context      = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head * 2)) if context_dim is not None else None
            self.output_layer = nn.Sequential(nn.Linear(dim_head * heads, dim, bias = False), nn.LayerNorm(dim))
            
        # Linear Attention Initialization #############################################################################################################
            
        elif att_type == 'linear':
            
            self.g1 = nn.Parameter(torch.ones(1, dim, 1, 1))
            
            self.Q = nn.Sequential(nn.Dropout(dropout),
                                   nn.Conv2d(dim, dim_head*heads, 1, bias=False),
                                   nn.Conv2d(dim_head*heads, dim_head*heads, 3, bias=False, padding=1, groups=dim_head*heads))

            self.K = nn.Sequential(nn.Dropout(dropout),
                                   nn.Conv2d(dim, dim_head*heads, 1, bias=False),
                                   nn.Conv2d(dim_head*heads, dim_head*heads, 3, bias=False, padding=1, groups=dim_head*heads))

            self.V = nn.Sequential(nn.Dropout(dropout),
                                   nn.Conv2d(dim, dim_head*heads, 1, bias=False),
                                   nn.Conv2d(dim_head*heads, dim_head*heads, 3, bias=False, padding=1, groups=dim_head*heads))
            
            self.context = nn.Sequential(nn.LayerNorm(context_dim), nn.Linear(context_dim, dim_head*heads*2, bias=False)) if context_dim is not None else None

            self.activation   = Swish()
            self.output_layer = nn.Conv2d(dim_head*heads, dim, 1, bias=False)            
            self.g2           = nn.Parameter(torch.ones(1, dim, 1, 1))
            
        # Cross and Linear Cross Attention Initialization #############################################################################################

        elif att_type == 'cross' or att_type == 'linear-cross':

            context_dim = context_dim if context_dim is not None else dim

            self.norm         = nn.LayerNorm(dim)
            if norm_context:
                self.norm_context = nn.LayerNorm(context_dim)

            self.null_kv = nn.Parameter(torch.randn(2, dim_head))
            self.Q       = nn.Linear(dim, dim_head*heads, bias=False)
            self.KV      = nn.Linear(context_dim, dim_head*heads*2, bias=False)

            self.output_layer = nn.Sequential(nn.Linear(dim_head*heads, dim, bias=False), nn.LayerNorm(dim))
    
    #===================================================================================#
    #                               ATTENTION FUNCTION                                  #
    #===================================================================================#    
        
    def attention(self, q, k, v, mask = None, att_bias = None):
        
        # Normal Attention ######################################################
        
        if self.att_type == 'normal':
            score = einsum('b h i d, b j d -> b h i j', q, k)

            if att_bias is not None: score = score + att_bias

            if mask is not None:            
                max_neg = -torch.finfo(score.dtype).max            
                mask    = mask[:, None, None, :]
                score   = score.masked_fill(~mask, max_neg)

            probs = score.softmax(dim = -1, dtype = torch.float32)
            
            att = einsum('b h i j, b j d -> b h i d', probs, v)
            att = rearrange(att, 'b h n d -> b n (h d)')
            
        # Linear Attention ######################################################
            
        elif self.att_type == 'linear':
            
            att = torch.matmul(k.transpose(-1, -2), v)
            att = torch.matmul(q, att)
            
        # Cross Attention #######################################################
            
        elif self.att_type == 'cross':
            score = torch.matmul(q, k.transpose(-1, -2))

            if mask is not None:            
                max_neg = -torch.finfo(score.dtype).max            
                mask    = mask[:, None, None, :]
                score   = score.masked_fill(~mask, max_neg)

            probs = score.softmax(dim = -1, dtype = torch.float32)

            att = torch.matmul(probs, v)
            att = rearrange(att, 'b h n d -> b n (h d)')
            
        # Linear Cross Attention ################################################
            
        elif self.att_type == 'linear-cross':           

            if mask is not None: 
                max_neg = -torch.finfo(score.dtype).max
                mask = mask[:, :, None]
                k, v = k.masked_fill(~mask, max_neg), v.masked_fill(~mask, 0.)

            q, k = q.softmax(dim = -1)*self.scale, k.softmax(dim = -2)

            att = torch.matmul(k.transpose(-1, -2), v)
            att = torch.matmul(q, att)
            att = rearrange(att, '(b h) n d -> b n (h d)', h = self.heads)
        
        return att
    
    #===================================================================================#
    #                                      FORWARD                                      #
    #===================================================================================#      

    def forward(self, x, context = None, mask = None, att_bias = None):
        
        # Normal Attention #####################################################################################
        
        if self.att_type == 'normal': 
        
            b, n, device = x.shape[0], x.shape[1], x.device

            x = self.norm(x)

            q = self.Q(x)
            q = rearrange(q, 'b n (h d) -> b h n d', h = self.heads) * self.scale
            
            k, v = self.KV(x).chunk(2, dim = -1)

            nk, nv = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b 1 d', b = b)
            k, v = torch.cat((nk, k), dim = -2), torch.cat((nv, v), dim = -2)

            if context is not None:
                assert self.context is not None
                ck, cv = self.context(context).chunk(2, dim = -1)
                k, v = torch.cat((ck, k), dim = -2), torch.cat((cv, v), dim = -2)

            # ATTENTION

            if mask is not None:
                mask = F.pad(mask, (1, 0), value = True)
                att = self.attention(q=q, k=k, v=v, mask=mask, att_bias=att_bias)
            else:
                att = self.attention(q=q, k=k, v=v, att_bias=att_bias)
                
            out = self.output_layer(att) 
            
        # Linear Attention #####################################################################################
        
        elif self.att_type == 'linear': 

            x1, y1, h = x.shape[-2], x.shape[-1], self.heads
            
            var  = torch.var(x, dim = 1, unbiased = False, keepdim = True)
            mean = torch.mean(x, dim = 1, keepdim = True)
            x    = (x - mean) / (var + 1e-15).sqrt() * self.g1   
            
            q, k, v = self.Q(x), self.K(x), self.V(x)
            q, k, v = rearrange_many((q, k, v), 'b (h c) x y -> (b h) (x y) c', h = h)

            if context is not None:
                assert self.context is not None
                ck, cv = self.context(context).chunk(2, dim = -1)
                ck, cv = rearrange_many((ck, cv), 'b n (h d) -> (b h) n d', h = h)
                k, v = torch.cat((k, ck), dim = -2), torch.cat((v, cv), dim = -2)

            q, k = q.softmax(dim = -1)*self.scale, k.softmax(dim = -2)
            
            att = self.attention(q, k, v)

            out = rearrange(att, '(b h) (x y) d -> b (h d) x y', h = h, x = x1, y = y1)
            out = self.activation(out)
            out = self.output_layer(out)
            
            var_out  = torch.var(out, dim = 1, unbiased = False, keepdim = True)
            mean_out = torch.mean(out, dim = 1, keepdim = True)
            out      = (x - mean_out) / (var_out + 1e-15).sqrt() * self.g2  
            
        # Cross and Linear Cross Attention #####################################################################
            
        elif self.att_type == 'cross' or self.att_type == 'linear-cross':
        # elif self.att_type == 'cross':
            
            b, n, device = x.shape[0], x.shape[1], x.device

            x       = self.norm(x)
            context = self.norm_context(context) if self.norm_context else context

            q    = self.Q(x)
            k, v = self.KV(context).chunk(2, dim = -1)

            if self.att_type == 'cross':    
                
                q, k, v = rearrange_many((q, k, v), 'b n (h d) -> b h n d', h = self.heads)
                nk, nv  = repeat_many(self.null_kv.unbind(dim = -2), 'd -> b h 1 d', h = self.heads,  b = b)
                k, v, q = torch.cat((nk, k), dim = -2), torch.cat((nv, v), dim = -2), q*self.scale
                
            else:
                q, k, v = rearrange_many((q, k, v), 'b n (h d) -> (b h) n d', h = self.heads)
                nk, nv  = repeat_many(self.null_kv.unbind(dim = -2), 'd -> (b h) 1 d', h = self.heads,  b = b)
                k, v    = torch.cat((nk, k), dim = -2), torch.cat((nv, v), dim = -2)
            
            if mask is not None:
                mask = F.pad(mask, (1, 0), value = True)
                att = self.attention(q=q, k=k, v=v, mask=mask)
            else:
                att = self.attention(q=q, k=k, v=v)

            out = self.output_layer(att)
            
        return out

######################################################################################################
################ Cross Embeddings ####################################################################
######################################################################################################

class CrossEmbedding(nn.Module):
    
    def __init__(self, in_channels, out_channels, stride, kernel_sizes):
        
        super(CrossEmbedding, self).__init__()        
        assert all([ks % 2 == stride % 2 for ks in kernel_sizes]), 'Kernel and Stride must be odd or even, both'
        assert all([ks >= stride for ks in kernel_sizes]), 'All Kernels must be larger than Stride'
        
        kernel_sizes, n_kernels = sorted(kernel_sizes), len(kernel_sizes)
        
        dim_scales = [int(out_channels / (2 ** i)) for i in range(1, n_kernels)]
        dim_scales.append(out_channels - sum(dim_scales))
        
        self.conv_layer = nn.ModuleList([nn.Conv2d(in_channels, dim_scale, kernel,
                                                   stride = stride, padding = (kernel - stride) // 2)
                                        for kernel, dim_scale in zip(kernel_sizes, dim_scales)])
        
    def forward(self, x):
        out = [conv(x) for conv in self.conv_layer]
        return torch.cat(out, dim = 1)

######################################################################################################
################ Resnet Layer for time and image #####################################################
######################################################################################################

class ResnetLayer(nn.Module):
    def __init__(self, dim, dim_out, cond_dim = None, time_cond_dim = None, groups = 8, linear_att = False, gca = False):
        
        super(ResnetLayer, self).__init__()
        
        self.gca = gca

        self.time_NN = nn.Sequential(Swish(), nn.Linear(time_cond_dim, dim_out*2))

        self.att_layer = None
        if cond_dim is not None:
            att_type = 'cross' if not linear_att else 'linear-cross'    
            self.att_layer = AttentionTypes(dim = dim_out, context_dim = cond_dim, att_type=att_type)

        self.seq1 = nn.Sequential(nn.GroupNorm(groups, dim),
                                  Swish(),
                                  nn.Conv2d(dim, dim_out, kernel_size=3, padding=1))
        
        self.norm = nn.GroupNorm(groups, dim_out)
        self.seq2 = nn.Sequential(Swish(),
                                  nn.Conv2d(dim_out, dim_out, kernel_size=3, padding=1))
        
        if gca:
            self.conv = nn.Conv2d(dim_out, 1, 1)
            self.seq3 = nn.Sequential(nn.Conv2d(dim_out, max(3, dim_out//2), 1),
                                      Swish(),
                                      nn.Conv2d(max(3, dim_out//2), dim_out, 1),
                                      nn.Sigmoid())

        self.output_layer = None
        if dim != dim_out:
            self.output_layer = nn.Conv2d(dim, dim_out, 1)


    def forward(self, x, time_emb = None, cond = None):

        scale, shift = None, None
        if self.time_NN is not None and time_emb is not None:
            
            time_emb = self.time_NN(time_emb)
            scale, shift = time_emb[:, :, None, None].chunk(2, dim = 1)

        h = self.seq1(x)

        if self.att_layer is not None:
            assert cond is not None
            
            w = h.shape[-1]            
            m = rearrange(h, 'b c h w -> b (h w) c')
            m = self.att_layer(m, context = cond) 
            h = rearrange(m, 'b (h w) c -> b c h w',w=w) + h    
                
        h = self.norm(h)*(scale + 1) + shift if scale is not None else self.norm(h)        
        h = self.seq2(h)
        
        if self.gca:
            c    = self.conv(h)
            y, c = rearrange_many((h, c), 'b n ... -> b n (...)')

            c = c.softmax(dim = -1) 
            o = torch.matmul(y, c.transpose(1, 2))[:, :, :, None]
        
            h = h*self.seq3(o)
        
        return h + self.output_layer(x) if self.output_layer is not None else h + x

######################################################################################################
################ Normal and Linear Transformer Layer #################################################
######################################################################################################

class TransformerLayer(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 32, ff_mult = 2, att_type='normal', context_dim = None):
        super(TransformerLayer, self).__init__()
        
        self.att_type = att_type

        ff_mult = int(ff_mult)
        
        self.att   = AttentionTypes(dim=dim, heads=heads, dim_head=dim_head, context_dim=context_dim, att_type=att_type)
        self.g1    = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.conv1 = nn.Conv2d(dim, dim*ff_mult, 1, bias = False)
        self.activ = Swish()
        self.g2    = nn.Parameter(torch.ones(1, dim*ff_mult, 1, 1))
        self.conv2 = nn.Conv2d(dim*ff_mult, dim, 1, bias = False)

    def forward(self, x, context = None):
        
        if self.att_type == 'normal':
            w = x.shape[-1]    
            x = rearrange(x, 'b c h w -> b (h w) c')
            x = self.att(x, context = context) + x
            x = rearrange(x, 'b (h w) c -> b c h w',w=w)
            
        elif self.att_type == 'linear':
            x = self.att(x, context = context) + x
        
        num = x - torch.mean(x, dim=1, keepdim=True)
        den = (torch.var(x, dim=1, unbiased=False, keepdim=True) + 1e-5).sqrt()
        x1  = (num/den)*self.g1
        
        x1 = self.activ(self.conv1(x1))
        
        num = x1 - torch.mean(x1, dim=1, keepdim=True)
        den = (torch.var(x1, dim=1, unbiased=False, keepdim=True) + 1e-5).sqrt()
        x1  = (num/den)*self.g2
        
        out = self.conv2(x1) + x
        return out