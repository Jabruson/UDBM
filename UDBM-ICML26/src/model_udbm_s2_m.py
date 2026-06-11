import copy
import glob
import math
import os
import random
import lpips
from collections import namedtuple
from functools import partial
from multiprocessing import cpu_count
from pathlib import Path
import torchvision.transforms as transforms
import Augmentor
import cv2
import numpy as np
import torch
from torchvision import utils as vutils
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from accelerate import Accelerator
from einops import rearrange, reduce
from einops.layers.torch import Rearrange
from ema_pytorch import EMA
from PIL import Image
import time
from torch import einsum, nn
from torch.optim import Adam, RAdam
from torch.utils.data import DataLoader
from torchvision import transforms as T
from torchvision import utils
from tqdm.auto import tqdm
from thop import profile
import copy
import importlib
from .local_arch import Local_Base
ModelResPrediction = namedtuple(
    'ModelResPrediction', ['pred_res', 'pred_noise', 'pred_x_start'])
# helpers functions
metric_module = importlib.import_module('metrics')

def set_seed(SEED):
    # initialize random seed
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

def tensor2img(tensor, rgb2bgr=True, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        rgb2bgr (bool): Whether to change rgb to bgr.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = img_np.transpose(1, 2, 0)
            if rgb2bgr:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = img_np.transpose(1, 2, 0)
            if img_np.shape[2] == 1:  # gray image
                img_np = np.squeeze(img_np, axis=2)
            else:
                if rgb2bgr:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np = img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result

def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num) ** 2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


# normalization functions


def normalize_to_neg_one_to_one(img):
    if isinstance(img, list):
        return [img[k] * 2 - 1 for k in range(len(img))]
    else:
        return img * 2 - 1


def unnormalize_to_zero_to_one(img):
    if isinstance(img, list):
        return [(img[k] + 1) * 0.5 for k in range(len(img))]
    else:
        return (img + 1) * 0.5

# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)

# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(
            half_dim), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered

# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y',
                        h=self.heads, x=h, y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(
            t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        condition=False,
    ):
        super().__init__()

        # determine dimensions

        self.channels = channels
        self.depth = len(dim_mults)
        input_channels = channels + channels * (1 if condition else 0)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(nn.ModuleList([
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                    dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(nn.ModuleList([
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                    dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def check_image_size(self, x, h, w):
        s = int(math.pow(2, self.depth))
        mod_pad_h = (s - h % s) % s
        mod_pad_w = (s - w % s) % s
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        return x
    
    def forward(self, x, time):
        H, W = x.shape[2:]
        x = self.check_image_size(x, H, W)
        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        x = self.final_conv(x)
        x = x[..., :H, :W].contiguous()
        return x


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, time_emb_dim=None, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            SimpleGate(), nn.Linear(time_emb_dim // 2, c * 4)
        ) if time_emb_dim else None

        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        self.norm1 = LayerNorm(c)
        self.norm2 = LayerNorm(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def time_forward(self, time, mlp):
        time_emb = mlp(time)
        time_emb = rearrange(time_emb, 'b c -> b c 1 1')
        return time_emb.chunk(4, dim=1)

    def forward(self, x):
        inp, time = x
        shift_att, scale_att, shift_ffn, scale_ffn = self.time_forward(time, self.mlp)

        x = inp

        x = self.norm1(x)
        x = x * (scale_att + 1) + shift_att
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.norm2(y)
        x = x * (scale_ffn + 1) + shift_ffn
        x = self.conv4(x)
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        x = y + x * self.gamma

        return x, time
# building block modules
class ConditionalNAFNet(nn.Module):
    # 48 28 or 64 16
    def __init__(self, img_channel=3, width=24, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1], upscale=1):
        super().__init__()
        self.upscale = upscale


        time_dim = width * 4
        
        sinu_pos_emb = SinusoidalPosEmb(width)
        fourier_dim = width
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )


        self.intro = nn.Conv2d(in_channels=img_channel*2, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan, time_dim) for _ in range(middle_blk_num)]
            )

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan, time_dim) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, x, time):



        t = self.time_mlp(time)

        B, C, H, W = x.shape
        x = self.check_image_size(x)

        x = self.intro(x)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x, _ = encoder([x, t])
            encs.append(x)
            x = down(x)

        x, _ = self.middle_blks([x, t])

        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip
            x, _ = decoder([x, t])

        x = self.ending(x)

        x = x[..., :H, :W].contiguous()

        return x
    def check_image_size(self, x):
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x
class ConditionalNAFNet_local(Local_Base, ConditionalNAFNet):
    def __init__(self, *args, train_size=(1, 6, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        ConditionalNAFNet.__init__(self)

        N, C, H, W = train_size
        base_size = (int(H * 1.5), int(W * 1.5))
        time_size=(1,)
        self.eval()
        with torch.no_grad():
            self.convert(base_size=base_size, train_size=train_size,time_size=time_size, fast_imp=fast_imp)

class UnetRes2(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        num_unet=1,
        condition=False,
        objective='pred_res_noise',
        test_res_or_noise="res_noise"
    ):
        super().__init__()
        self.condition = condition
        self.channels = channels
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.num_unet = num_unet
        self.objective = objective
        self.test_res_or_noise = test_res_or_noise
        # determine dimensions
    
        self.unet0 = ConditionalNAFNet(img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1], upscale=1)

    def forward(self, x, time):
        if self.objective == "pred_noise":
            time = time[1]
        elif self.objective == "pred_res":
            time = time[0]
        return [self.unet0(x, time)]
class UnetRes(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        num_unet=1,
        condition=False,
        objective='pred_res_noise',
        test_res_or_noise="res_noise",
        test_mode=False
    ):
        super().__init__()
        self.condition = condition
        self.channels = channels
        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features
        self.num_unet = num_unet
        self.objective = objective
        self.test_res_or_noise = test_res_or_noise
        # determine dimensions
        if not test_mode:
            self.unet0 = ConditionalNAFNet()
        else:
            self.unet0 = ConditionalNAFNet_local()

    def forward(self, x, time):
        if self.objective == "pred_noise":
            time = time[1]
        elif self.objective == "pred_res":
            time = time[0]
        return [self.unet0(x, time)]

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def gen_coefficients(timesteps, schedule="increased", sum_scale=1, ratio=1):
    if schedule == "increased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        alphas = y/y_sum
    elif schedule == "decreased":
        x = np.linspace(0, 1, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        y_sum = y.sum()
        y = torch.flip(y, dims=[0])
        alphas = y/y_sum
    elif schedule == "lamda":
        x = np.linspace(0.0001, 0.02, timesteps, dtype=np.float32)
        y = x**ratio
        y = torch.from_numpy(y)
        alphas = 1 - y
    elif schedule == "average":
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)
    elif schedule == "normal":
        sigma = 1.0
        mu = 0.0
        x = np.linspace(-3+mu, 3+mu, timesteps, dtype=np.float32)
        y = np.e**(-((x-mu)**2)/(2*(sigma**2)))/(np.sqrt(2*np.pi)*(sigma**2))
        y = torch.from_numpy(y)
        alphas = y/y.sum()
    else:
        alphas = torch.full([timesteps], 1/timesteps, dtype=torch.float32)

    return alphas*sum_scale

# Copied from diffusers.schedulers.scheduling_ddpm.betas_for_alpha_bar
def betas_for_alpha_bar(num_diffusion_timesteps, max_beta=0.999) -> torch.Tensor:
    """
    Create a beta schedule that discretizes the given alpha_t_bar function, which defines the cumulative product of
    (1-beta) over time from t = [0,1].

    Contains a function alpha_bar that takes an argument t and transforms it to the cumulative product of (1-beta) up
    to that part of the diffusion process.


    Args:
        num_diffusion_timesteps (`int`): the number of betas to produce.
        max_beta (`float`): the maximum beta to use; use values lower than 1 to
                     prevent singularities.

    Returns:
        betas (`np.ndarray`): the betas used by the scheduler to step the model outputs
    """

    def alpha_bar(time_step):
        return math.cos((time_step + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return torch.tensor(betas, dtype=torch.float32)


class ResidualDiffusion(nn.Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        delta_end = 1.5e-3,
        sampling_timesteps=None,
        loss_type='l1',
        objective='pred_res_noise',
        ddim_sampling_eta=0.,
        condition=False,
        sum_scale=None,
        test_res_or_noise="None",
        b_H_max=0.8,
        b_H_min=0.13,
        b_E_max=0.13,
        b_E_min=0.08
    ):  
        super().__init__()
        assert not (
            type(self) == ResidualDiffusion and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond
        self.b_H_max = b_H_max
        self.b_H_min = b_H_min
        self.b_E_max = b_E_max
        self.b_E_min = b_E_min
        self.model = model
        self.channels = self.model.channels
        self.image_size = image_size
        self.objective = objective
        self.condition = condition
        self.test_res_or_noise = test_res_or_noise
        self.delta_end = delta_end
       
        if self.condition:
            self.sum_scale = sum_scale if sum_scale else 0.01
        else:
            self.sum_scale = sum_scale if sum_scale else 1.

        beta_schedule = "linear"
        beta_start = 0.0001
        beta_end = 0.02
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            betas = (torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32) ** 2)
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            betas = betas_for_alpha_bar(timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")
            
        delta_start = 1e-6
        delta = torch.linspace(delta_start, self.delta_end, timesteps, dtype=torch.float32)
        delta_cumsum = delta.cumsum(dim=0).clip(0, 1)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumsum = 1-alphas_cumprod ** 0.5
        betas2_cumsum = 1-alphas_cumprod

        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        alphas = alphas_cumsum-alphas_cumsum_prev
        alphas[0] = 0
        betas2 = betas2_cumsum-betas2_cumsum_prev
        betas2[0] = 0
        betas_cumsum = torch.sqrt(betas2_cumsum) 

        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)
        
        # Fixed uncertainty D=0.5
        
        
        self.loss_type = loss_type

        # sampling related parameters
        # default num sampling timesteps to number of timesteps at training
        self.sampling_timesteps = default(sampling_timesteps, timesteps)

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        def register_buffer(name, val): return self.register_buffer(
            name, val.to(torch.float32))
        t_sche=self.get_t_sche()
        register_buffer('t_sche', t_sche)
        register_buffer('alphas', alphas)
        # register_buffer('s_curve_lut',s_curve_lut)
        register_buffer('alphas_cumsum', alphas_cumsum)
        register_buffer('delta', delta)
        register_buffer('delta_cumsum', delta_cumsum)
        register_buffer('one_minus_alphas_cumsum', 1-alphas_cumsum)
        register_buffer('betas2', betas2)
        register_buffer('betas', torch.sqrt(betas2))
        register_buffer('betas2_cumsum', betas2_cumsum)
        register_buffer('betas_cumsum', betas_cumsum)
        register_buffer('posterior_mean_coef1',
                        betas2_cumsum_prev/betas2_cumsum)
        register_buffer('posterior_mean_coef2', (betas2 *
                        alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum)
        register_buffer('posterior_mean_coef3', betas2/betas2_cumsum)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6
    def get_t_sche3(self):
        p_min = 1.5
        p_max = 3.5
        max_beta_scale = 5.0
        d_fixed = torch.full((1, 1, 1, 1), 0.5)

        # Full timestep sequence [0, 1, 2, ..., T-1]
        t_steps = torch.arange(self.num_timesteps)
        
        # Normalize and reshape timesteps.
        t_norm = t_steps.float() / (self.num_timesteps-1)
        t_reshaped = t_norm.view(-1, 1, 1, 1)

        # Compute the noise schedule.
        # Only the noise coefficient is needed here.
        peak_magnitude = (1. + d_fixed) * max_beta_scale
        endpoint_magnitude = (d_fixed + 1.) * max_beta_scale * 0.2
        
        hump_curve = 4 * t_reshaped * (1 - t_reshaped)
        ramp_curve = t_reshaped**2
    
    
        noise_variance = peak_magnitude * hump_curve + endpoint_magnitude * ramp_curve
        noise_variance=noise_variance.squeeze(-1).squeeze(-1).squeeze(-1)
        return noise_variance
    def get_t_sche(self):
        p_min = 0.5
        p_max = 1.0
        max_beta_scale = 5.0
        epsilon=1e-9
        d_fixed = torch.full((1, 1, 1, 1), 0.5)

        # Full timestep sequence [0, 1, 2, ..., T-1]
        t_steps = torch.arange(self.num_timesteps)
        
        # Normalize and reshape timesteps.
        t_norm = t_steps.float() / (self.num_timesteps-1)
        t_reshaped = t_norm.view(-1, 1, 1, 1)

        p = p_min + d_fixed * (p_max - p_min)                                                                                 
                                                                                                                        
        # Parameterized S-curve.                                                                                                
        numerator = t_reshaped ** p                                                                                     
        denominator = numerator + (1 - t_reshaped) ** p + epsilon                                                       
                                                                                                                        
        lq_coeff = numerator / denominator                                                                              
        hq_coeff = 1 - lq_coeff
        return hq_coeff.squeeze(-1).squeeze(-1).squeeze(-1)
    def init(self):
        timesteps = 1000

        beta_schedule = "linear"
        beta_start = 0.0001
        beta_end = 0.02
        if beta_schedule == "linear":
            betas = torch.linspace(
                beta_start, beta_end, timesteps, dtype=torch.float32)
        elif beta_schedule == "scaled_linear":
            # this schedule is very specific to the latent diffusion model.
            betas = (
                torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32) ** 2
            )
        elif beta_schedule == "squaredcos_cap_v2":
            # Glide cosine schedule
            betas = betas_for_alpha_bar(timesteps)
        else:
            raise NotImplementedError(f"{beta_schedule} does is not implemented for {self.__class__}")
                
        delta_start = 1e-6
        delta = torch.linspace(delta_start, self.delta_end, timesteps, dtype=torch.float32)
        delta_cumsum = delta.cumsum(dim=0).clip(0, 1)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumsum = 1-alphas_cumprod ** 0.5
        betas2_cumsum = 1-alphas_cumprod

        alphas_cumsum_prev = F.pad(alphas_cumsum[:-1], (1, 0), value=1.)
        betas2_cumsum_prev = F.pad(betas2_cumsum[:-1], (1, 0), value=1.)
        alphas = alphas_cumsum-alphas_cumsum_prev
        alphas[0] = alphas[1]
        betas2 = betas2_cumsum-betas2_cumsum_prev
        betas2[0] = betas2[1]

        betas_cumsum = torch.sqrt(betas2_cumsum)
    
        posterior_variance = betas2*betas2_cumsum_prev/betas2_cumsum
        posterior_variance[0] = 0

        timesteps, = alphas.shape
        self.num_timesteps = int(timesteps)

        self.alphas = alphas
        self.alphas_cumsum = alphas_cumsum
        self.delta = delta
        self.delta_cumsum = delta_cumsum
        self.one_minus_alphas_cumsum = 1-alphas_cumsum
        self.betas2 = betas2
        self.betas = torch.sqrt(betas2)
        self.betas2_cumsum = betas2_cumsum
        self.betas_cumsum = betas_cumsum
        self.posterior_mean_coef1 = betas2_cumsum_prev/betas2_cumsum
        self.posterior_mean_coef2 = (
            betas2 * alphas_cumsum_prev-betas2_cumsum_prev*alphas)/betas2_cumsum
        self.posterior_mean_coef3 = betas2/betas2_cumsum
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(
            posterior_variance.clamp(min=1e-20))

        self.posterior_mean_coef1[0] = 0
        self.posterior_mean_coef2[0] = 0
        self.posterior_mean_coef3[0] = 1
        self.one_minus_alphas_cumsum[-1] = 1e-6

    def predict_noise_from_res(self, x_t, t, x_input, pred_res):
        return (
            (x_t - (1-extract(self.delta_cumsum,t,x_t.shape)) * x_input - (extract(self.alphas_cumsum, t, x_t.shape)-1)
             * pred_res)/extract(self.betas_cumsum, t, x_t.shape)
        )

    def predict_start_from_xinput_noise(self, x_t, t, x_input, noise):
        return (
            (x_t-extract(self.alphas_cumsum, t, x_t.shape)*x_input -
             extract(self.betas_cumsum, t, x_t.shape) * noise)/extract(self.one_minus_alphas_cumsum, t, x_t.shape)
        )

    def predict_start_from_res_noise(self, x_t, t, x_res, noise):
        return (
            x_t-extract(self.alphas_cumsum, t, x_t.shape) * x_res -
            extract(self.betas_cumsum, t, x_t.shape) * noise
        )

    def q_posterior_from_res_noise(self, x_res, noise, x_t, t):
        return (x_t-extract(self.alphas, t, x_t.shape) * x_res -
                (extract(self.betas2, t, x_t.shape)/extract(self.betas_cumsum, t, x_t.shape)) * noise)

    def q_posterior(self, pred_res, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_t +
            extract(self.posterior_mean_coef2, t, x_t.shape) * pred_res +
            extract(self.posterior_mean_coef3, t, x_t.shape) * x_start
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x_input, x, t, task=None, clip_denoised=True):
        if not self.condition:
            x_in = x
        else:
            x_in = torch.cat((x, x_input), dim=1)
        model_output = self.model(x_in,
                                  [self.alphas_cumsum[t]*self.num_timesteps,
                                      self.betas_cumsum[t]*self.num_timesteps])
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_denoised else identity

        if self.objective == 'pred_res_noise':
            if self.test_res_or_noise == "res_noise":
                pred_res = model_output[0]
                pred_noise = model_output[1]
                pred_res = maybe_clip(pred_res)
                x_start = self.predict_start_from_res_noise(
                    x, t, pred_res, pred_noise)
                x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "res":
                pred_res = model_output[0]
                pred_res = maybe_clip(pred_res)
                pred_noise = self.predict_noise_from_res(
                    x, t, x_input, pred_res)
                x_start = x_input - pred_res
                x_start = maybe_clip(x_start)
            elif self.test_res_or_noise == "noise":
                pred_noise = model_output[1]
                x_start = self.predict_start_from_xinput_noise(
                    x, t, x_input, pred_noise)
                x_start = maybe_clip(x_start)
                pred_res = x_input - x_start
                pred_res = maybe_clip(pred_res)
        elif self.objective == 'pred_x0_noise':
            pred_res = x_input-model_output[0]
            pred_noise = model_output[1]
            pred_res = maybe_clip(pred_res)
            x_start = maybe_clip(model_output[0])
        elif self.objective == "pred_noise":
            pred_noise = model_output[0]
            x_start = self.predict_start_from_xinput_noise(
                x, t, x_input, pred_noise)
            x_start = maybe_clip(x_start)
            pred_res = x_input - x_start
            pred_res = maybe_clip(pred_res)
        elif self.objective == "pred_res":
            pred_res = model_output[0]
            pred_res = maybe_clip(pred_res)
            pred_noise = self.predict_noise_from_res(x, t, x_input, pred_res)
            x_start = x_input - pred_res
            x_start = maybe_clip(x_start)

        return ModelResPrediction(pred_res, pred_noise, x_start)

    def p_mean_variance(self, x_input, x, t):
        preds = self.model_predictions(x_input, x, t)
        pred_res = preds.pred_res
        x_start = preds.pred_x_start

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            pred_res=pred_res, x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self, x_input, x, t: int):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x_input, x=x, t=batched_times)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self, x_input, shape, last=True):
        x_input = x_input[0]

        batch, device = shape[0], self.betas.device

        if self.condition:
            img = x_input+math.sqrt(self.sum_scale) * \
                torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None

        if not last:
            img_list = []

        for t in tqdm(reversed(range(0, self.num_timesteps)), desc='sampling loop time step', total=self.num_timesteps):
            img, x_start = self.p_sample(x_input, img, t)

            if not last:
                img_list.append(img)

        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)

    @torch.no_grad()
    def ddim_sample(self, x_input, shape, last=True, task=None):
        x_input = x_input[0]

        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))
   
        if self.condition:
            img = (1-self.delta_cumsum[-1]) * x_input + math.sqrt(self.sum_scale) * torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        x_start = None
        type = "use_pred_noise"
        last=False

        if not last:
            img_list = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            preds = self.model_predictions(x_input, img, time_cond, task)

            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            alpha_cumsum = self.alphas_cumsum[time]
            alpha_cumsum_next = self.alphas_cumsum[time_next]
            alpha = alpha_cumsum-alpha_cumsum_next
            delta_cumsum = self.delta_cumsum[time]
            delta_cumsum_next = self.delta_cumsum[time_next]
            delta = delta_cumsum-delta_cumsum_next
            betas2_cumsum = self.betas2_cumsum[time]
            betas2_cumsum_next = self.betas2_cumsum[time_next]
            betas2 = betas2_cumsum-betas2_cumsum_next
            betas = betas2.sqrt()
            betas_cumsum = self.betas_cumsum[time]
            betas_cumsum_next = self.betas_cumsum[time_next]
            sigma2 = eta * (betas2*betas2_cumsum_next/betas2_cumsum)
            sqrt_betas2_cumsum_next_minus_sigma2_divided_betas_cumsum = (
                betas2_cumsum_next-sigma2).sqrt()/betas_cumsum
            q = betas2_cumsum_next / betas2_cumsum

            if eta == 0:
                noise = 0
            else:
                noise = torch.randn_like(img)
            if type == "use_pred_noise":
                img = img - alpha*pred_res + delta*x_input + sigma2.sqrt()*noise
            elif type == "use_x_start":
                img = q*img + \
                    (1-q)*x_start + \
                    (alpha_cumsum_next-alpha_cumsum*q)*pred_res + \
                    (delta_cumsum*q-delta_cumsum_next)*x_input + \
                    sigma2.sqrt()*noise
            elif type == "special_eta_0":
                img = img - alpha*pred_res - \
                    (betas_cumsum-betas_cumsum_next)*pred_noise
            elif type == "special_eta_1":
                img = img - alpha*pred_res - betas2/betas_cumsum*pred_noise + \
                    betas*betas2_cumsum_next.sqrt()/betas_cumsum*noise
                    
            if not last:
                img_list.append(img)
    
        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)
    def sample_uncertainty_aware_bridge(self, x_input, shape, model_s1=None,last=True, task=None):
        x_input = x_input[0]
        self.sum_scale=1
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        t0 = torch.randint(1, 2, (batch,), device=x_input.device).long()
        uncertain=model_s1(torch.cat((x_input, x_input), dim=1),[self.alphas_cumsum[t0]*self.num_timesteps,self.betas_cumsum[t0]*self.num_timesteps])[0]
        uncertain.clamp_(-2,2)
        uncertain=torch.abs(uncertain)*0.5
            
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))
        time_condTx = torch.full(
                (batch,), self.num_timesteps-1, device=device, dtype=torch.long)
        if self.condition:
            thq,tlq,tn=self._calculate_schedule(time_condTx,uncertain)
            img=tlq*x_input+tn * torch.randn(shape, device=device)
            input_add_noise = img
        else:
            img = torch.randn(shape, device=device)

        
        


        x_start = None
        type = "use_pred_noise"
        last=False

        if not last:
            img_list = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            time_cond_next = torch.full(
                (batch,), time_next, device=device, dtype=torch.long)
            x_in = torch.cat((img, x_input), dim=1)
            model_output = self.model(x_in,
                                  [self.t_sche[time_cond]*self.num_timesteps,
                                      self.betas_cumsum[time_cond]*self.num_timesteps])
            clip_denoised=True
            maybe_clip = partial(torch.clamp, min=-1.,
                                    max=1.) if clip_denoised else identity
            pred_res = model_output[0]
            
            pred_noise = self.predict_noise_from_res(img, time_cond, x_input, pred_res)
            x_start = x_input - pred_res
            x_start = maybe_clip(x_start)
            preds=ModelResPrediction(pred_res, pred_noise, x_start)


            pred_res = preds.pred_res
            pred_noise = preds.pred_noise
            x_start = preds.pred_x_start

            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue
            thq,tlq,tn=self._calculate_schedule(time_cond,uncertain)
            thq_prev,tlq_prev,tn_prev=self._calculate_schedule(time_cond_next,uncertain)
            beta_t_prev_beta_t_beta=tn_prev/tn

            img=beta_t_prev_beta_t_beta*img+(tlq_prev-beta_t_prev_beta_t_beta*tlq)*x_input+(thq_prev-beta_t_prev_beta_t_beta*thq)*x_start

            
                    
            if not last:
                img_list.append(img)
    
        if self.condition:
            if not last:
                img_list = [input_add_noise]+img_list
            else:
                img_list = [input_add_noise, img]
            return unnormalize_to_zero_to_one(img_list)
        else:
            if not last:
                img_list = img_list
            else:
                img_list = [img]
            return unnormalize_to_zero_to_one(img_list)
    @torch.no_grad()
    def sample(self,x_input=0, batch_size=16, last=True, task=None,glcm=0,model_s1=None,x_hq=None):
        image_size, channels = self.image_size, self.channels
        sample_fn = self.sample_uncertainty_aware_bridge
        if self.condition:
            x_input = 2 * x_input - 1
            x_input = x_input.unsqueeze(0)

            batch_size, channels, h, w = x_input[0].shape
            size = (batch_size, channels, h, w)
        else:
            size = (batch_size, channels, image_size, image_size)
        return sample_fn(x_input, size,model_s1=model_s1, last=last, task=task)
    def _calculate_B_schedule(self, t, d, a=5):
        """
        Compute the uncertainty-conditioned noise coefficient B_t(d).
        This keeps the original tensor interface and uses a pulse-plus-ramp schedule.

        Args:
        t (torch.Tensor): timestep tensor with shape [B].
        d (torch.Tensor): uncertainty tensor with shape [B, C, H, W] or a broadcastable shape.
        
        Required attributes:
        self.num_timesteps (int): number of timesteps.
        self.b_H_max, self.b_H_min (float): pulse peak bounds.
        self.b_E_max, self.b_E_min (float): endpoint bounds.
        """
        # Validate inputs.
        assert t.ndim == 1 
        # d may have any shape that broadcasts with t.
        assert t.shape[0] == d.shape[0]

        # Compute dynamic parameters.
        # h_d and E_d are derived from the uncertainty map.
        # The operation is elementwise, so their shape follows d.
        # For [B, C, H, W] d, h_d and E_d have the same shape.
        h_d = self.b_H_min + (self.b_H_max - self.b_H_min) * (1 - d)
        E_d = self.b_E_min + (self.b_E_max - self.b_E_min) * (1 - d)

        # Schedule computation with broadcasting.
        # Normalize timesteps to [0, 1].
        t_norm = t.float() / self.num_timesteps

        # Reshape t from [B] to [B, 1, 1, 1] so it broadcasts with d.
        
        t_norm_reshaped = t_norm.view(-1, 1, 1, 1)

        # Pulse term.
        
        pulse = h_d * torch.sin(torch.pi * t_norm_reshaped)

        # Ramp term.
        
        ramp = t_norm_reshaped * E_d

        # The output shape follows d.
        b_val = pulse + ramp
        
        return b_val
    def _calculate_B_schedulexx(self, t):
        """
        Sigmoid B schedule.
        t: timestep tensor with shape [B].
        """
        T = self.num_timesteps - 1
        t_norm = t.float().unsqueeze(1) / T # [B, 1]
        # Fixed sigmoid parameters.
        a = 5
        d = 0.5
        raw = torch.sigmoid(a * (t_norm - d))  # [B, 1]

        # Precomputed sigmoid endpoints.
        raw0 = 0.0758581800  # sigmoid(-2.5)
        rawT = 0.9241418200  # sigmoid(2.5)

        # Normalize to [0, 1].
        normalized = (raw - raw0) / (rawT - raw0 + 1e-8)

        return normalized.reshape(-1, 1, 1, 1)
    def q_sample(self, x_gt, x_lq, t, noise=None,uncertain=None):
        noise = default(noise, lambda: torch.randn_like(x_gt))
        thq,tlq,tn=self._calculate_schedule(t,uncertain)
        # Dynamic B_t(d).
        return thq * x_gt + tlq * x_lq+tn*noise

    @property
    def loss_fn(self, loss_type='l1'):
        if loss_type == 'l1':
            return F.l1_loss
        elif loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {loss_type}')
    
    
    @torch.no_grad()
    def ddpm_sample_resshift(self, x_input, shape, model_s1,last=True, task=None,d=None,v_max=None,x_hq=None):

        type = "use_pred_noise"
        x_input = x_input[0]
        self.num_timesteps=4
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective
        
        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]
        time_pairs = list(zip(times[:-1], times[1:]))
        time_condT = torch.full(
                (batch,), self.num_timesteps-1, device=device, dtype=torch.long)

        t0 = torch.randint(1, 2, (1,), device=x_input.device).long()
        uncertain=model_s1(torch.cat((x_input, x_input), dim=1),
                               [self.alphas_cumsum[t0]*self.num_timesteps,
                                self.betas_cumsum[t0]*self.num_timesteps])[0]

        uncertain=(uncertain+1)*0.5
        uncertain.clamp_(0,1)
        uncertain=torch.pow(uncertain,2)
        max_v_u = torch.clamp(uncertain.view(batch, -1).max(dim=1, keepdim=True)[0][:, None, None], min=0).to(uncertain[0].device)

        uncertain_n=1
        if self.condition:
            
            img = x_input+ self._calculate_B_schedule(time_condT, uncertain)*torch.randn(shape, device=device)*uncertain_n
            input_add_noise = img

        else:
            img = torch.randn(shape, device=device)

        x_start = None

        last=False

        if not last:
            img_list = []

        for time, time_next in tqdm(time_pairs, desc='sampling loop time step'):
            time_cond = torch.full(
                (batch,), time, device=device, dtype=torch.long)
            time_cond_next = torch.full(
                (batch,), time_next, device=device, dtype=torch.long)
            btd=self._calculate_B_schedulexx(time_cond).squeeze(-1).squeeze(-1).squeeze(-1)*self.num_timesteps
            B_t_d = self._calculate_B_schedule(time_condT, uncertain)
            pred_res = self.model(torch.cat((img, x_input), dim=1), [btd,self.betas_cumsum[time_cond]*self.num_timesteps])
            pred_res=pred_res[0]
            
            x_start=x_input - pred_res
            x_start.clamp_(-1., 1.)
            if time_next < 0:
                img = x_start
                if not last:
                    img_list.append(img)
                continue

            time_prev_cond = torch.full((batch,), time_next, device=device, dtype=torch.long)
            B_t_prev_d = self._calculate_B_schedule(time_prev_cond, uncertain)+(1e-8)
            A_t_prev=1-B_t_prev_d

            # Previous-step update.
            bt_prev_bt=B_t_prev_d/B_t_d
            at=torch.sqrt(B_t_d)-torch.sqrt(B_t_prev_d)
            bt_min_bt_prev=B_t_d-B_t_prev_d
            img=bt_prev_bt*img+bt_min_bt_prev/B_t_d*x_start+torch.sqrt(bt_min_bt_prev*bt_prev_bt)*torch.randn(shape, device=device)*uncertain_n
            if not last:
                img_list.append(img)
    
        # Return results.   
        if not last:
            img_list=[input_add_noise]+img_list
        else:
            img_list = [input_add_noise, img]

        return unnormalize_to_zero_to_one(img_list)
    def _calculate_schedule(self, t,d):
           # Schedule parameters.                                                                                                
            p_min = 0.5                                                                                                     
            p_max = 1.0                                                                                                    
            max_beta_scale = 5.0                                                                                            
            epsilon = 1e-9  # Avoid division by zero.                                                                                         
            t_norm = t.float() / (self.num_timesteps-1)                                                                         
            # Reshape t for broadcasting.                                                                                      
            # t changes from [B] to [B, 1, 1, 1] for operations with d.                                                               
            t_reshaped = t_norm.view(-1, 1, 1, 1)                                                                           
                                                                                                                            
            # Compute HQ and LQ path coefficients.                                                                                     
            # Map uncertainty to the S-curve exponent.                                                                                          
            p = p_min + (1-d) * (p_max - p_min)                                                                                 
                                                                                                                            
            # Parameterized S-curve.                                                                                                
            numerator = t_reshaped ** p                                                                                     
            denominator = numerator + (1 - t_reshaped) ** p + epsilon                                                       
                                                                                                                            
            lq_coeff = numerator / denominator                                                                              
            hq_coeff = 1 - lq_coeff                                                                                         
                                                                                                                            
            # Compute the noise coefficient.                                                                                      
            # Peak and endpoint magnitudes.                                                                                                
            peak_magnitude = (1. + d) * max_beta_scale                                                                      
            endpoint_magnitude = (d+1.) * max_beta_scale * 0.2                                                              
                                                                                                                            
            # Base curve shape.                                                                                                   
            hump_curve = 4 * t_reshaped * (1 - t_reshaped)                                                                  
            ramp_curve = t_reshaped**2                                                                                      
                                                                                                                            
            # Noise variance.                                                                                      
            noise_variance = peak_magnitude * hump_curve + endpoint_magnitude * ramp_curve                                  
                                                                                                                            
            return hq_coeff, lq_coeff, noise_variance                                                                          
    def p_losses(self, imgs, t,model_s1, noise=None):
        if isinstance(imgs, list):  # Condition
            x_input = 2 * imgs[1] - 1
            x_start = 2 * imgs[0] - 1  # gt: imgs[0], cond: imgs[1]
            task = imgs[2][0]
        
            
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_res = x_input - x_start

        b, c, h, w = x_start.shape
        with torch.no_grad():
            t0 = torch.randint(1, 2, (b,), device=x_input.device).long()
            uncertain=model_s1(torch.cat((x_input, x_input), dim=1),[self.alphas_cumsum[t0]*self.num_timesteps,self.betas_cumsum[t0]*self.num_timesteps])[0]
            uncertain.clamp_(-2,2)
            uncertain=torch.abs(uncertain)*0.5
     
        x = self.q_sample(x_gt=x_start, x_lq=x_input, t=t, noise=noise,uncertain=uncertain)
        if not self.condition:
            x_in = x
        else:
            x_in = torch.cat((x, x_input), dim=1)
        model_out = self.model(x_in,
                               [self.t_sche[t]*self.num_timesteps,
                                self.betas_cumsum[t]*self.num_timesteps])

        target = []
        if self.objective == 'pred_res_noise':
            target.append(x_res)
            target.append(noise)

            pred_res = model_out[0]
            pred_noise = model_out[1]
        elif self.objective == 'pred_x0_noise':
            target.append(x_start)
            target.append(noise)

            pred_res = x_input-model_out[0]
            pred_noise = model_out[1]
        elif self.objective == "pred_noise":
            target.append(noise)

            pred_noise = model_out[0]

        elif self.objective == "pred_res":
            target.append(x_res)
        
            pred_res = model_out[0]

        else:
            raise ValueError(f'unknown objective {self.objective}')

        u_loss = False
        if u_loss:
            x_u = self.q_posterior_from_res_noise(pred_res, pred_noise, x, t)
            u_gt = self.q_posterior_from_res_noise(x_res, noise, x, t)
            loss = 10000*self.loss_fn(x_u, u_gt, reduction='none')
            return [loss]
        else:
            loss_list = []
            for i in range(len(model_out)):
                loss = self.loss_fn(model_out[i], target[i], reduction='none')
                loss = reduce(loss, 'b ... -> b (...)', 'mean').mean()
                loss_list.append(loss)
            return loss_list

    def forward(self, img, model_s1,*args, **kwargs):
        if isinstance(img, list):
            b, c, h, w, device, img_size, = * \
                img[0].shape, img[0].device, self.image_size
        else:
            b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        return self.p_losses(img, t,model_s1, *args, **kwargs)

class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        dataset,
        opts,
        *,
        train_batch_size=16,
        gradient_accumulate_every=1,
        augment_flip=True,
        train_lr=1e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        adam_betas=(0.9, 0.99),
        save_and_sample_every=1000,
        num_unet=1,
        num_samples=25,
        results_folder='./results/sample',
        amp=False,
        fp16=False,
        split_batches=True,
        convert_image_to=None,
        condition=False,
        sub_dir=False,
        test_mode=False
    ):
        super().__init__()

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no'
        )
        self.sub_dir = sub_dir
        self.accelerator.native_amp = amp
        self.num_unet = num_unet
        self.model = diffusion_model

        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size
        self.condition = condition

        if self.condition:
            if opts.phase == "train":
                task_batch_sizes = getattr(opts, "task_batch_sizes", None)
                if task_batch_sizes is None:
                    task_batch_sizes = (16, 4, 8, 8, 4) if self.gradient_accumulate_every == 1 else (8, 2, 4, 4, 2)
                elif isinstance(task_batch_sizes, str):
                    task_batch_sizes = tuple(int(x) for x in task_batch_sizes.split(","))
                if len(task_batch_sizes) != 5:
                    raise ValueError("task_batch_sizes must contain fog,light,rain,snow,blur batch sizes")
                fog_bs, light_bs, rain_bs, snow_bs, blur_bs = task_batch_sizes
                self.dl_fog = cycle(self.accelerator.prepare(DataLoader(dataset[0], batch_size=fog_bs, shuffle=True, pin_memory=True, num_workers=4)))
                self.dl_light = cycle(self.accelerator.prepare(DataLoader(dataset[1], batch_size=light_bs, shuffle=True, pin_memory=True, num_workers=1)))
                self.dl_rain = cycle(self.accelerator.prepare(DataLoader(dataset[2], batch_size=rain_bs, shuffle=True, pin_memory=True, num_workers=2)))
                self.dl_snow = cycle(self.accelerator.prepare(DataLoader(dataset[3], batch_size=snow_bs, shuffle=True, pin_memory=True, num_workers=2)))
                self.dl_blur = cycle(self.accelerator.prepare(DataLoader(dataset[4], batch_size=blur_bs, shuffle=True, pin_memory=True, num_workers=1)))
            else:
                self.sample_dataset = dataset
                

        # optimizer
        self.opt0 = Adam(diffusion_model.parameters(), lr=train_lr, betas=adam_betas)

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model, beta=ema_decay,
                           update_every=ema_update_every)

            self.set_results_folder(results_folder)

        # step counter state
        self.step = 0

        # prepare model, dataloader, optimizer with accelerator
        self.model, self.opt0 = self.accelerator.prepare(self.model, self.opt0)
        device = self.accelerator.device
        self.device = device
        self.model_s1 = self.accelerator.unwrap_model(UnetRes2(dim=64,dim_mults=(1, 2, 4, 8),num_unet=1,condition=True,objective='pred_res',test_res_or_noise = "res"))
        data_model_s1 = torch.load(opts.ckpt_path_s1, map_location=self.device)['model']
        model2_state_dict = {k.replace('model.', '', 1): v for k, v in data_model_s1.items() if k.startswith('model.')}
        self.model_s1.load_state_dict(model2_state_dict)
        self.model_s1=self.model_s1.to(self.device)
        for param in self.model_s1.parameters():
            param.requires_grad = False
        self.model_s1.eval()
    def save(self, milestone):
        if not self.accelerator.is_local_main_process:
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt0': self.opt0.state_dict(),
            'ema': self.ema.state_dict(),
            'scaler': self.accelerator.scaler.state_dict() if exists(self.accelerator.scaler) else None
        }
        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        path = Path(self.results_folder / f'model-{milestone}.pt')
        if path.exists():
            data = torch.load(str(path), map_location=self.device)
            self.model = self.accelerator.unwrap_model(self.model)
  
            self.model.load_state_dict(data['model'])
            self.step = data['step']
            
            self.opt0.load_state_dict(data['opt0'])
            self.opt0.param_groups[0]['capturable'] = True
            self.ema.load_state_dict(data['ema'])

            if exists(self.accelerator.scaler) and exists(data['scaler']):
                self.accelerator.scaler.load_state_dict(data['scaler'])

            print("load model - "+str(path))


    def train(self):
        accelerator = self.accelerator

        with tqdm(initial=self.step, total=self.train_num_steps, disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = [0]
                
                for _ in range(self.gradient_accumulate_every):
                    if self.condition:
                        batch1 = next(self.dl_fog)
                        batch2 = next(self.dl_light)
                        batch3 = next(self.dl_rain)
                        batch4 = next(self.dl_snow)
                        batch5 = next(self.dl_blur)
                       
                        data = {}
                        for k, v in batch1.items():
                            if 'path' in k:
                                data[k] = batch1[k] + batch2[k] + batch3[k] + batch4[k] + batch5[k]
                            else:
                                data[k] = torch.cat([batch1[k], batch2[k], batch3[k], batch4[k], batch5[k]], dim=0)
                        gt = data["gt"].to(self.device)
                        cond_input = data["adap"].to(self.device)
                        task = data["A_paths"]
                        data = [gt, cond_input, task]
                    else:
                        data = next(self.dl)
                        data = data[0] if isinstance(data, list) else data
                        data = data.to(self.device)

                    with self.accelerator.autocast():
                        loss = self.model(data,self.model_s1)
                        for i in range(self.num_unet):
                            loss[i] = loss[i] / self.gradient_accumulate_every
                            total_loss[i] = total_loss[i] + loss[i].item()

                    for i in range(self.num_unet):
                        self.accelerator.backward(loss[i])

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)

                accelerator.wait_for_everyone()

                self.opt0.step()
                self.opt0.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.to(self.device)
                    self.ema.update()

                    if self.step != 0 and self.step % (self.save_and_sample_every*10) == 0:
                        milestone = self.step // self.save_and_sample_every
                        self.save(milestone)
                        
                pbar.set_description(f'loss_unet0: {total_loss[0]:.4f}')
                pbar.update(1)

        accelerator.print('training complete')
    
    def test(self, sample=False, last=True, FID=False,task=None):
        self.ema.ema_model.init()
        self.ema.to(self.device)
        print("test start")
        if self.condition:
            self.ema.ema_model.eval()
            loader = DataLoader(
                dataset=self.sample_dataset,
                batch_size=1)
            i = 0
            cnt = 0
            opt_metric = {
                'psnr': {
                    'type': 'calculate_psnr',
                    'crop_border': 0,
                    'test_y_channel': True
                    },
                'ssim': {
                    'type': 'calculate_ssim',
                    'crop_border': 0,
                    'test_y_channel': True
                    },
                'niqe': {
                        'type': 'calculate_niqe'
                        }
                    
                }
            self.metric_results = {
                metric: 0
                for metric in opt_metric.keys()
            }
            tran = transforms.ToTensor()
            for items in loader:
                if self.condition:
                    file_ = items["A_paths"][0]
                    file_name = file_.split('/')[-1]
                else:
                    file_name = f'{i}.png'

                i += 1
                
                with torch.no_grad():
                    batches = self.num_samples
                    
                    data = items
                    x_input_sample = data["adap"].to(self.device)
                    gt = data["gt"].to(self.device)

                    all_images_list = list(self.ema.ema_model.sample(x_input_sample, batches, last=last, task=file_,model_s1=self.model_s1))
                all_images_list = [all_images_list[-1]]
                all_images = torch.cat(all_images_list, dim=0)
 
                if last:
                    nrow = int(math.sqrt(self.num_samples))
                else:
                    nrow = all_images.shape[0]
                file_name = file_.split('/')[-1]
                save_path = str(self.results_folder /"fsf_final_small_s1_16_24" /task)
                os.makedirs(save_path, exist_ok=True)
                full_path = os.path.join(save_path,file_name)
                if 'real' in task:
                    utils.save_image(all_images, full_path, nrow=nrow)
                else:
                    utils.save_image(all_images, full_path.replace(".jpg",".png"), nrow=nrow)
                print("test-save "+full_path)
                
            #calculate the metric
            
                sr_img = tensor2img(all_images, rgb2bgr=True)
                gt_img = tensor2img(gt, rgb2bgr=True)
                if 'real' in task:
                    opt_metric_ = {
                        'psnr': {
                            'type': 'calculate_psnr',
                            'crop_border': 0,
                            'test_y_channel': True
                            },
                        'ssim': {
                            'type': 'calculate_ssim',
                            'crop_border': 0,
                            'test_y_channel': True
                            },
                    
                }
                        
                else:
                    opt_metric_ = {
                        'psnr': {
                            'type': 'calculate_psnr',
                            'crop_border': 0,
                            'test_y_channel': True
                            },
                        'ssim': {
                            'type': 'calculate_ssim',
                            'crop_border': 0,
                            'test_y_channel': True
                            }
                        }
                for name, opt_ in opt_metric_.items():
                    metric_type = opt_.pop('type')
                    if metric_type=='calculate_niqe':
                        print(sr_img.max())
                        self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, **opt_)
                    else:
                        self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)
               
                cnt += 1

            current_metric = {}
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric[metric] = self.metric_results[metric]
            print(current_metric)
        
        print("test end")
        return current_metric
    def test_udc(self, sample=False, last=True, FID=False,task=None):
        self.ema.ema_model.init()
        self.ema.to(self.device)
        print("test start")
        self.loss_fn = lpips.LPIPS(net='vgg')
        self.loss_fn.cuda()
        if self.condition:
            self.ema.ema_model.eval()
            i = 0
            cnt = 0
            opt_metric = {
                'psnr': {
                    'type': 'calculate_psnr',
                    'crop_border': 0,
                    'test_y_channel': False
                    },
                'ssim': {
                    'type': 'calculate_ssim',
                    'crop_border': 0,
                    'test_y_channel': False
                    }
                }
            self.metric_results = {
                metric: 0
                for metric in opt_metric.keys()
            }
            dist_all = 0

            for i in range(self.sample_dataset[0].shape[0]):
                x_input_sample = torch.tensor(self.sample_dataset[0][i]).to(self.device)
                gt = torch.tensor(self.sample_dataset[1][i]).to(self.device)
                
                x_input_sample = x_input_sample.permute(2,0,1).contiguous().unsqueeze(0)/255
                gt = gt.permute(2,0,1).contiguous().unsqueeze(0)/255
                i += 1
                temp = []
                for k in range(2):
                        with torch.no_grad():
                            batches = self.num_samples
                            asd = x_input_sample[:, :, :, k*1024:(k+1)*1024]
                            all_images_list = list(self.ema.ema_model.sample(asd, batch_size=batches, last=last, task=None,model_s1=self.model_s1))
                        all_images_list = [all_images_list[-1]]
                        all_images = torch.cat(all_images_list, dim=0)
                        temp.append(all_images)
                all_images = torch.cat([temp[0],temp[1]],dim=-1)
                
                #------------------------calculate lpips----------------------------------------
                with torch.no_grad():
                    dist01 = self.loss_fn.forward(all_images, gt)
                dist_all += dist01

                #------------------------save_image----------------------------------------
                save_path = self.results_folder / "udc" / f"{cnt}_test.jpg"
                os.makedirs(save_path.parent, exist_ok=True)
                vutils.save_image(all_images, str(save_path), normalize=True)
                
                sr_img = tensor2img(all_images, rgb2bgr=True)
                gt_img = tensor2img(gt, rgb2bgr=True)
                opt_metric_ = {
                    'psnr': {
                        'type': 'calculate_psnr',
                        'crop_border': 0,
                        'test_y_channel': False
                        },
                    'ssim': {
                        'type': 'calculate_ssim',
                        'crop_border': 0,
                        'test_y_channel': False
                        }
                    }
                for name, opt_ in opt_metric_.items():
                    metric_type = opt_.pop('type')
                    self.metric_results[name] += getattr(metric_module, metric_type)(sr_img, gt_img, **opt_)

                cnt += 1

            current_metric = {}
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric[metric] = self.metric_results[metric]
            print(current_metric['psnr'])
            print(current_metric['ssim'])
            print(dist_all / cnt)
        
        print("test end")

    def set_results_folder(self, path):
        self.results_folder = Path(path)
        if not self.results_folder.exists():
            os.makedirs(self.results_folder)


AuxiliaryUncertaintyEstimator = UnetRes2
RestorationDenoiser = UnetRes
UncertaintyAwareDiffusionBridge = ResidualDiffusion
UDBMTrainer = Trainer



