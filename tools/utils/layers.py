import random
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf import DictConfig

from tools.torch_utils import persistence
from tools.torch_utils.ops import bias_act, upfirdn2d, conv2d_resample
from tools.torch_utils import misc

#----------------------------------------------------------------------------

@misc.profiled_function
def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()

#----------------------------------------------------------------------------

@persistence.persistent_class
class MappingNetwork(torch.nn.Module):
    def __init__(self,
        z_dim,                      # Input latent (Z) dimensionality, 0 = no latent.
        c_dim,                      # Conditioning label (C) dimensionality, 0 = no label.
        w_dim,                      # Intermediate latent (W) dimensionality.
        num_ws,                     # Number of intermediate latents to output, None = do not broadcast.
        num_layers      = 8,        # Number of mapping layers.
        embed_features  = None,     # Label embedding dimensionality, None = same as w_dim.
        layer_features  = None,     # Number of intermediate features in the mapping layers, None = same as w_dim.
        activation      = 'lrelu',  # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 0.01,     # Learning rate multiplier for the mapping layers.
        w_avg_beta      = 0.995,    # Decay for tracking the moving average of W during training, None = do not track.
        cfg             = {},       # Additional config
    ):
        super().__init__()

        self.cfg = cfg
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.w_dim = w_dim
        self.num_ws = num_ws
        self.num_layers = num_layers
        self.w_avg_beta = w_avg_beta

        if embed_features is None:
            embed_features = w_dim
        if c_dim == 0:
            embed_features = 0
        if layer_features is None:
            layer_features = w_dim

        features_list = [z_dim + embed_features] + [layer_features] * (num_layers - 1) + [w_dim]

        if c_dim > 0:
            self.embed = FullyConnectedLayer(c_dim, embed_features)

        for idx in range(num_layers):
            in_features = features_list[idx]
            out_features = features_list[idx + 1]
            layer = FullyConnectedLayer(in_features, out_features, activation=activation, lr_multiplier=lr_multiplier)
            setattr(self, f'fc{idx}', layer)

        if num_ws is not None and w_avg_beta is not None:
            self.register_buffer('w_avg', torch.zeros([w_dim]))

    def forward(self, z, c, truncation_psi=1, truncation_cutoff=None, skip_w_avg_update=False):
        # Embed, normalize, and concat inputs.
        x = None
        with torch.autograd.profiler.record_function('input'):
            if self.z_dim > 0:
                misc.assert_shape(z, [None, self.z_dim])
                x = normalize_2nd_moment(z.to(torch.float32))

            if self.c_dim > 0:
                misc.assert_shape(c, [None, self.c_dim])
                y = normalize_2nd_moment(self.embed(c.to(torch.float32)))
                x = torch.cat([x, y], dim=1) if x is not None else y

        # Main layers.
        for idx in range(self.num_layers):
            layer = getattr(self, f'fc{idx}')
            x = layer(x)

        # Update moving average of W.
        if self.w_avg_beta is not None and self.training and not skip_w_avg_update:
            with torch.autograd.profiler.record_function('update_w_avg'):
                self.w_avg.copy_(x.detach().mean(dim=0).lerp(self.w_avg, self.w_avg_beta))

        # Broadcast.
        if self.num_ws is not None:
            with torch.autograd.profiler.record_function('broadcast'):
                x = x.unsqueeze(1).repeat([1, self.num_ws, 1])

        # Apply truncation.
        if truncation_psi != 1:
            with torch.autograd.profiler.record_function('truncate'):
                assert self.w_avg_beta is not None
                if self.num_ws is None or truncation_cutoff is None:
                    x = self.w_avg.lerp(x, truncation_psi)
                else:
                    x[:, :truncation_cutoff] = self.w_avg.lerp(x[:, :truncation_cutoff], truncation_psi)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class FullyConnectedLayer(torch.nn.Module):
    def __init__(self,
        in_features,                # Number of input features.
        out_features,               # Number of output features.
        bias            = True,     # Apply additive bias before the activation function?
        activation      = 'linear', # Activation function: 'relu', 'lrelu', etc.
        lr_multiplier   = 1,        # Learning rate multiplier.
        bias_init       = 0,        # Initial value for the additive bias.
    ):
        super().__init__()
        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], float(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features)
        self.bias_gain = lr_multiplier

    def forward(self, x):
        w = self.weight.to(x.dtype) * self.weight_gain
        b = self.bias
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        if self.activation == 'linear' and b is not None:
            x = torch.addmm(b.unsqueeze(0), x, w.t())
        else:
            x = x.matmul(w.t())
            x = bias_act.bias_act(x, b, act=self.activation)
        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class Conv2dLayer(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        kernel_size,                    # Width and height of the convolution kernel.
        bias            = True,         # Apply additive bias before the activation function?
        activation      = 'linear',     # Activation function: 'relu', 'lrelu', etc.
        up              = 1,            # Integer upsampling factor.
        down            = 1,            # Integer downsampling factor.
        resample_filter = [1,3,3,1],    # Low-pass filter to apply when resampling activations.
        conv_clamp      = None,         # Clamp the output to +-X, None = disable clamping.
        channels_last   = False,        # Expect the input to have memory_format=channels_last?
        trainable       = True,         # Update the weights of this layer during training?
        instance_norm   = False,        # Should we apply instance normalization to y?
        lr_multiplier   = 1.0,          # Learning rate multiplier.
    ):
        super().__init__()
        self.activation = activation
        self.up = up
        self.down = down
        self.conv_clamp = conv_clamp
        self.register_buffer('resample_filter', upfirdn2d.setup_filter(resample_filter))
        self.padding = kernel_size // 2
        self.weight_gain = 1 / np.sqrt(in_channels * (kernel_size ** 2))
        self.act_gain = bias_act.activation_funcs[activation].def_gain
        self.instance_norm = instance_norm
        self.lr_multiplier = lr_multiplier

        memory_format = torch.channels_last if channels_last else torch.contiguous_format
        weight = torch.randn([out_channels, in_channels, kernel_size, kernel_size]).to(memory_format=memory_format)
        bias = torch.zeros([out_channels]) if bias else None
        if trainable:
            self.weight = torch.nn.Parameter(weight)
            self.bias = torch.nn.Parameter(bias) if bias is not None else None
        else:
            self.register_buffer('weight', weight)
            if bias is not None:
                self.register_buffer('bias', bias)
            else:
                self.bias = None

    def forward(self, x, gain=1):
        w = self.weight * (self.weight_gain * self.lr_multiplier)
        b = (self.bias.to(x.dtype) * self.lr_multiplier) if self.bias is not None else None
        flip_weight = (self.up == 1) # slightly faster
        x = conv2d_resample.conv2d_resample(x=x, w=w.to(x.dtype), f=self.resample_filter, up=self.up, down=self.down, padding=self.padding, flip_weight=flip_weight)

        act_gain = self.act_gain * gain
        act_clamp = self.conv_clamp * gain if self.conv_clamp is not None else None
        x = bias_act.bias_act(x, b, act=self.activation, gain=act_gain, clamp=act_clamp)

        if self.instance_norm:
            x = (x - x.mean(dim=(2,3), keepdim=True)) / (x.std(dim=(2,3), keepdim=True) + 1e-8) # [batch_size, c, h, w]

        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class GenInput(nn.Module):
    def __init__(self, cfg: DictConfig, channel_dim: int, motion_v_dim: int=None):
        super().__init__()

        self.cfg = cfg

        if self.cfg.input.type == 'const':
            self.input = torch.nn.Parameter(torch.randn([channel_dim, 4, 4]))
            self.total_dim = channel_dim
        elif self.cfg.input.type == 'temporal':
            self.input = TemporalInput(self.cfg, channel_dim, motion_v_dim=motion_v_dim)
            self.total_dim = self.input.get_dim()
        else:
            raise NotImplementedError(f'Unkown input type: {self.cfg.input.type}')

    def forward(self, batch_size: int, motion_v: Optional[torch.Tensor]=None, dtype=None, memory_format=None) -> torch.Tensor:
        if self.cfg.input.type == 'const':
            x = self.input.to(dtype=dtype, memory_format=memory_format)
            x = x.unsqueeze(0).repeat([batch_size, 1, 1, 1])
        elif self.cfg.input.type == 'temporal':
            x = self.input(motion_v=motion_v) # [batch_size, d, h, w]
        else:
            raise NotImplementedError(f'Unkown input type: {self.cfg.input.type}')

        return x

#----------------------------------------------------------------------------

@persistence.persistent_class
class TemporalInput(nn.Module):
    def __init__(self, cfg: DictConfig, channel_dim: int, motion_v_dim: int):
        super().__init__()

        self.cfg = cfg
        self.motion_v_dim = motion_v_dim
        self.const = nn.Parameter(torch.randn(1, channel_dim, 4, 4))

    def get_dim(self):
        return self.motion_v_dim + self.const.shape[1]

    def forward(self, motion_v: torch.Tensor) -> torch.Tensor:
        """
        motion_v: [batch_size, motion_v_dim]
        """
        out = torch.cat([
            self.const.repeat(len(motion_v), 1, 1, 1),
            motion_v.unsqueeze(2).unsqueeze(3).repeat(1, 1, *self.const.shape[2:]),
        ], dim=1) # [batch_size, channel_dim + num_fourier_feats * 2]

        return out

#----------------------------------------------------------------------------

class TemporalDifferenceEncoder(nn.Module):
    def __init__(self, cfg: DictConfig):
        super().__init__()

        self.cfg = cfg

        if self.cfg.sampling.num_frames_per_video > 1:
            self.d = 256
            self.const_embed = nn.Embedding(self.cfg.sampling.max_num_frames, self.d)
            self.time_encoder = FixedTimeEncoder(
                self.cfg.sampling.max_num_frames,
                skip_small_t_freqs=self.cfg.get('skip_small_t_freqs', 0))

    def get_dim(self) -> int:
        if self.cfg.sampling.num_frames_per_video == 1:
            return 1
        else:
            if self.cfg.sampling.type == 'uniform':
                return self.d + self.time_encoder.get_dim()
            else:
                return (self.d + self.time_encoder.get_dim()) * (self.cfg.sampling.num_frames_per_video - 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        misc.assert_shape(t, [None, self.cfg.sampling.num_frames_per_video])

        batch_size = t.shape[0]

        if self.cfg.sampling.num_frames_per_video == 1:
            out = torch.zeros(len(t), 1, device=t.device)
        else:
            if self.cfg.sampling.type == 'uniform':
                num_diffs_to_use = 1
                t_diffs = t[:, 1] - t[:, 0] # [batch_size]
            else:
                num_diffs_to_use = self.cfg.sampling.num_frames_per_video - 1
                t_diffs = (t[:, 1:] - t[:, :-1]).view(-1) # [batch_size * (num_frames - 1)]
            # Note: float => round => long is necessary when it's originally long
            const_embs = self.const_embed(t_diffs.float().round().long()) # [batch_size * num_diffs_to_use, d]
            fourier_embs = self.time_encoder(t_diffs.unsqueeze(1)) # [batch_size * num_diffs_to_use, num_fourier_feats]
            out = torch.cat([const_embs, fourier_embs], dim=1) # [batch_size * num_diffs_to_use, d + num_fourier_feats]
            out = out.view(batch_size, num_diffs_to_use, -1).view(batch_size, -1) # [batch_size, num_diffs_to_use * (d + num_fourier_feats)]

        return out

#----------------------------------------------------------------------------

@persistence.persistent_class
class FixedTimeEncoder(nn.Module):
    def __init__(self,
            max_num_frames: int,            # Maximum T size
            skip_small_t_freqs: int=0,      # How many high frequencies we should skip
        ):
        super().__init__()

        assert max_num_frames >= 1, f"Wrong max_num_frames: {max_num_frames}"
        fourier_coefs = construct_log_spaced_freqs(max_num_frames, skip_small_t_freqs=skip_small_t_freqs)
        self.register_buffer('fourier_coefs', fourier_coefs) # [1, num_fourier_feats]

    def get_dim(self) -> int:
        return self.fourier_coefs.shape[1] * 2

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        assert t.ndim == 2, f"Wrong shape: {t.shape}"

        t = t.view(-1).float() # [batch_size * num_frames]
        fourier_raw_embs = self.fourier_coefs * t.unsqueeze(1) # [bf, num_fourier_feats]

        fourier_embs = torch.cat([
            fourier_raw_embs.sin(),
            fourier_raw_embs.cos(),
        ], dim=1) # [bf, num_fourier_feats * 2]

        return fourier_embs

#----------------------------------------------------------------------------

@persistence.persistent_class
class EqLRConv1d(nn.Module):
    def __init__(self,
        in_features: int,
        out_features: int,
        kernel_size: int,
        padding: int=0,
        stride: int=1,
        activation: str='linear',
        lr_multiplier: float=1.0,
        bias=True,
        bias_init=0.0,
    ):
        super().__init__()

        self.activation = activation
        self.weight = torch.nn.Parameter(torch.randn([out_features, in_features, kernel_size]) / lr_multiplier)
        self.bias = torch.nn.Parameter(torch.full([out_features], float(bias_init))) if bias else None
        self.weight_gain = lr_multiplier / np.sqrt(in_features * kernel_size)
        self.bias_gain = lr_multiplier
        self.padding = padding
        self.stride = stride

        assert self.activation in ['lrelu', 'linear']

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.ndim == 3, f"Wrong shape: {x.shape}"

        w = self.weight.to(x.dtype) * self.weight_gain # [out_features, in_features, kernel_size]
        b = self.bias # [out_features]
        if b is not None:
            b = b.to(x.dtype)
            if self.bias_gain != 1:
                b = b * self.bias_gain

        y = F.conv1d(input=x, weight=w, bias=b, stride=self.stride, padding=self.padding) # [batch_size, out_features, out_len]
        if self.activation == 'linear':
            pass
        elif self.activation == 'lrelu':
            y = F.leaky_relu(y, negative_slope=0.2) # [batch_size, out_features, out_len]
        else:
            raise NotImplementedError
        return y

#----------------------------------------------------------------------------

def sample_frames(cfg: Dict, total_video_len: int, **kwargs) -> np.ndarray:
    if cfg['type'] == 'random':
        return random_frame_sampling(cfg, total_video_len, **kwargs)
    elif cfg['type'] == 'uniform':
        return uniform_frame_sampling(cfg, total_video_len, **kwargs)
    else:
        raise NotImplementedError

#----------------------------------------------------------------------------

def random_frame_sampling(cfg: Dict, total_video_len: int, use_fractional_t: bool=False) -> np.ndarray:
    min_time_diff = cfg["num_frames_per_video"] - 1
    max_time_diff = min(total_video_len - 1, cfg.get('max_dist', float('inf')))

    if type(cfg.get('total_dists')) in (list, tuple):
        time_diff_range = [d for d in cfg['total_dists'] if min_time_diff <= d <= max_time_diff]
    else:
        time_diff_range = range(min_time_diff, max_time_diff)

    time_diff: int = random.choice(time_diff_range)
    if use_fractional_t:
        offset = random.random() * (total_video_len - time_diff - 1)
    else:
        offset = random.randint(0, total_video_len - time_diff - 1)
    frames_idx = [offset]

    if cfg["num_frames_per_video"] > 1:
        frames_idx.append(offset + time_diff)

    if cfg["num_frames_per_video"] > 2:
        frames_idx.extend([(offset + t) for t in random.sample(range(1, time_diff), k=cfg["num_frames_per_video"] - 2)])

    frames_idx = sorted(frames_idx)

    return np.array(frames_idx)

#----------------------------------------------------------------------------

def uniform_frame_sampling(cfg: Dict, total_video_len: int, use_fractional_t: bool=False) -> np.ndarray:
    # Step 1: Select the distance between frames
    if type(cfg.get('dists_between_frames')) in (list, tuple):
        valid_dists = [d for d in cfg['dists_between_frames'] if d <= ['max_dist_between_frames']]
        valid_dists = [d for d in valid_dists if (d * cfg['num_frames_per_video'] - d + 1) <= total_video_len]
        d = random.choice(valid_dists)
    else:
        max_dist = min(cfg.get('max_dist', float('inf')), total_video_len // cfg['num_frames_per_video'])
        d = random.randint(1, max_dist)

    d_total = d * cfg['num_frames_per_video'] - d + 1

    # Step 2: Sample.
    if use_fractional_t:
        offset = random.random() * (total_video_len - d_total)
    else:
        offset = random.randint(0, total_video_len - d_total)

    frames_idx = offset + np.arange(cfg['num_frames_per_video']) * d

    return frames_idx

#----------------------------------------------------------------------------

def construct_log_spaced_freqs(max_num_frames: int, skip_small_t_freqs: int=0) -> Tuple[int, torch.Tensor]:
    time_resolution = 2 ** np.ceil(np.log2(max_num_frames))
    num_fourier_feats = np.ceil(np.log2(time_resolution)).astype(int)
    powers = torch.tensor([2]).repeat(num_fourier_feats).pow(torch.arange(num_fourier_feats)) # [num_fourier_feats]
    powers = powers[:len(powers) - skip_small_t_freqs] # [num_fourier_feats]
    fourier_coefs = powers.unsqueeze(0).float() * np.pi # [1, num_fourier_feats]

    return fourier_coefs / time_resolution

#----------------------------------------------------------------------------
