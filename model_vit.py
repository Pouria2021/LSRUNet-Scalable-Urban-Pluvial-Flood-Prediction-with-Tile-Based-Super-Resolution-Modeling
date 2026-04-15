"""MaxViT-based U-Net for flood super-resolution prediction.

Implements a MaxViT encoder-decoder architecture compatible with the
ResUNet_aux training interface. Uses multi-axis attention (block + grid)
with relative position bias and MBConv stem blocks.
"""

import math
from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

# GPU memory usage note:
# ------4-level-512, batchsize-16, 36 M, ~23 GB
# ------4-level-512, batchsize-24, 36 M, ~34 GB
# ------4-level-1024, batchsize-8, 145 M, ~24 GB
# ------4-level-1024, batchsize-12, 145 M, ~34 GB

# -------------------------
# Small conv blocks that mirror your UNet style (BN + ReLU)
# -------------------------
class _DownConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=False),
        )
    def forward(self, x): return self.net(x)

class _UpConv(nn.Module):
    """Upsample x2 (bilinear) + 1x1 on skip + double conv, matching your UpConv pattern."""
    def __init__(self, in_plane: int, skip_plane: int, num_plane: int):
        super().__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.skip_connect = nn.Conv2d(skip_plane, in_plane, kernel_size=1, bias=False)
        self.conv = _DownConv(2*in_plane, num_plane)
    def forward(self, x, skip_x):
        x = self.upsample(x)
        x = torch.cat([x, self.skip_connect(skip_x)], dim=1)
        return self.conv(x)


# -------------------------
# MaxViT building blocks (MBConv + Block- & Grid-Attn w/ relative bias)
# -------------------------
class _ChannelLayerNorm(nn.Module):
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(c, eps=eps)
    def forward(self, x):
        return self.ln(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

class _SqueezeExcite(nn.Module):
    def __init__(self, c: int, se_ratio: float = 0.25):
        super().__init__()
        hidden = max(8, int(c * se_ratio))
        self.fc1 = nn.Conv2d(c, hidden, 1)
        self.fc2 = nn.Conv2d(hidden, c, 1)
    def forward(self, x):
        s = x.mean((2, 3), keepdim=True)
        s = F.gelu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class _MBConv(nn.Module):
    """Pre-act MBConv with optional stride-2 downsample in the DWConv, as in MaxViT."""
    def __init__(self, in_ch: int, out_ch: int, expansion: int = 4, se_ratio: float = 0.25, downsample: bool = False):
        super().__init__()
        mid = in_ch * expansion
        stride = 2 if downsample else 1

        self.expand = nn.Sequential(_ChannelLayerNorm(in_ch),
                                    nn.Conv2d(in_ch, mid, 1, bias=False),
                                    nn.GELU())
        self.dw = nn.Sequential(nn.Conv2d(mid, mid, 3, stride=stride, padding=1, groups=mid, bias=False),
                                nn.GELU())
        self.se = _SqueezeExcite(mid, se_ratio=se_ratio)
        self.project = nn.Conv2d(mid, out_ch, 1, bias=False)

        self.use_skip = (in_ch == out_ch) and (not downsample)
        if not self.use_skip:
            if downsample:
                self.skip = nn.Sequential(nn.AvgPool2d(2, 2), nn.Conv2d(in_ch, out_ch, 1, bias=False))
            else:
                self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=False)

    def forward(self, x):
        y = self.expand(x)
        y = self.dw(y)
        y = self.se(y)
        y = self.project(y)
        return x + y if self.use_skip else self.skip(x) + y


class _RelPosBias(nn.Module):
    """2D relative position bias table for square windows/grids."""
    def __init__(self, size: int, num_heads: int):
        super().__init__()
        self.size = size
        self.num_heads = num_heads
        self.table = nn.Parameter(torch.zeros((2*size-1)*(2*size-1), num_heads))

        coords = torch.stack(torch.meshgrid(torch.arange(size), torch.arange(size), indexing="ij"))
        coords_flat = coords.flatten(1)                  # (2, L)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, L, L)
        rel = rel.permute(1, 2, 0)
        rel[:, :, 0] += size - 1
        rel[:, :, 1] += size - 1
        rel_idx = rel[:, :, 0] * (2*size - 1) + rel[:, :, 1]
        self.register_buffer("index", rel_idx, persistent=False)
        nn.init.trunc_normal_(self.table, std=0.02)

    def forward(self):
        bias = self.table[self.index.reshape(-1)].reshape(self.size*self.size, self.size*self.size, self.num_heads)
        return bias.permute(2, 0, 1)  # (heads, L, L)


class _MHA(nn.Module):
    def __init__(self, dim: int, heads: int, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        assert dim % heads == 0, "dim must be divisible by heads"
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim*3, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_dropout)

    def forward(self, x, attn_bias=None):  # x: (B, N, C)
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None: attn = attn + attn_bias.unsqueeze(0)
        attn = self.attn_drop(attn.softmax(-1))
        y = (attn @ v).transpose(1, 2).reshape(B, N, C)
        y = self.proj_drop(self.proj(y))
        return y


class _MLP(nn.Module):
    def __init__(self, dim: int, expansion: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = int(dim * expansion)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))

def _window_attn_with_pad(x, attn: _MHA, rel_bias: _RelPosBias, P: int):
    """Window (block) attention with automatic right/bottom padding."""
    B, C, H, W = x.shape
    pad_h = (P - H % P) % P
    pad_w = (P - W % P) % P
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    Hp, Wp = x.shape[2], x.shape[3]
    xw = x.reshape(B, C, Hp//P, P, Wp//P, P).permute(0, 2, 4, 3, 5, 1).reshape(B*(Hp//P)*(Wp//P), P*P, C)
    y = attn(xw, attn_bias=rel_bias())
    y = y.reshape(B, Hp//P, Wp//P, P, P, C).permute(0, 5, 3, 1, 4, 2).reshape(B, C, Hp, Wp)
    return y[:, :, :H, :W]  # crop

def _grid_attn_with_pad(x, attn: _MHA, rel_bias: _RelPosBias, G: int):
    """Grid attention with automatic right/bottom padding."""
    B, C, H, W = x.shape
    pad_h = (G - H % G) % G
    pad_w = (G - W % G) % G
    if pad_h or pad_w:
        x = F.pad(x, (0, pad_w, 0, pad_h), mode='replicate')
    Hp, Wp = x.shape[2], x.shape[3]
    y = x.reshape(B, C, G, Hp//G, G, Wp//G).permute(0, 3, 5, 2, 4, 1).reshape(B*(Hp//G)*(Wp//G), G*G, C)
    y = attn(y, attn_bias=rel_bias())
    y = y.reshape(B, Hp//G, Wp//G, G, G, C).permute(0, 5, 3, 1, 4, 2).reshape(B, C, Hp, Wp)
    return y[:, :, :H, :W]  # crop


class _MaxViTBlock(nn.Module):
    """MBConv -> block/window attn -> MLP -> grid attn -> MLP."""
    def __init__(self, in_ch: int, out_ch: int, heads: int, window_size: int, grid_size: int,
                 downsample: bool, mlp_ratio: float = 4.0, attn_dropout: float = 0.0, dropout: float = 0.0):
        super().__init__()
        self.mbconv = _MBConv(in_ch, out_ch, expansion=4, se_ratio=0.25, downsample=downsample)

        self.norm1 = _ChannelLayerNorm(out_ch)
        self.ba = _MHA(out_ch, heads, attn_dropout, dropout)
        self.ba_bias = _RelPosBias(window_size, heads)
        self.ba_mlp = _MLP(out_ch, mlp_ratio, dropout)

        self.norm2 = _ChannelLayerNorm(out_ch)
        self.ga = _MHA(out_ch, heads, attn_dropout, dropout)
        self.ga_bias = _RelPosBias(grid_size, heads)
        self.ga_mlp = _MLP(out_ch, mlp_ratio, dropout)

        self.P = window_size
        self.G = grid_size

    def forward(self, x):
        x = self.mbconv(x)

        y = self.norm1(x)
        y = _window_attn_with_pad(y, self.ba, self.ba_bias, self.P)
        x = x + y
        y = self.ba_mlp(x.permute(0, 2, 3, 1).reshape(x.size(0), -1, x.size(1)))
        y = y.reshape(x.size(0), x.size(2), x.size(3), x.size(1)).permute(0, 3, 1, 2)
        x = x + y

        y = self.norm2(x)
        y = _grid_attn_with_pad(y, self.ga, self.ga_bias, self.G)
        x = x + y
        y = self.ga_mlp(x.permute(0, 2, 3, 1).reshape(x.size(0), -1, x.size(1)))
        y = y.reshape(x.size(0), x.size(2), x.size(3), x.size(1)).permute(0, 3, 1, 2)
        x = x + y
        return x


# -------------------------
# MaxVIT encoder-decoder with the "ResUNet_aux" interface
# -------------------------
class MaxVIT(nn.Module):
    """
    MaxVIT U-Net for DEM correction, with the SAME interface as ResUNet_aux.

    Args (kept for drop-in compatibility):
        num_input_channels (int): channels in x (e.g., DEM + predictors)
        num_target_channels (int): top encoder width (final stage channels); default 512
        num_levels (int): number of down/up levels; default 4
        num_aux_target_channels (list[int]): kept for compatibility (unused)
        num_aux_levels_list (list[int]): the length of this list determines channels in x_aux (concatenated)

    Notes:
      • Forward is forward(x, x_aux) and output passes through ReLU.
      • Encoder widths follow your pattern: base = num_target_channels / 2**num_levels,
        then [base*2, base*4, base*8, base*16] for the 4 stages (e.g., 64,128,256,512).
      • MaxViT block order MBConv → block-attn → grid-attn with relative position bias.
      • Works well with 224×224 patching/tiling and blended inference (per FathomDEM).
    """
    def __init__(
        self,
        num_input_channels: int = 2,
        num_target_channels: int = 512,
        num_levels: int = 4,
        num_aux_target_channels: List[int] = [512],   # compatibility placeholder
        num_aux_levels_list: List[int] = [4],         # only its length matters (channels in x_aux)
        window_size: int = 7,
        grid_size: int = 7,
        depths: List[int] | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_levels == 4, "This implementation expects num_levels=4 to match your UNet layout."

        # --- Channel plan to mirror your model: base -> 2x -> 4x -> 8x -> 16x
        base = int(num_target_channels // (2 ** num_levels))    # e.g., 512/16=32
        widths = [base * 2, base * 4, base * 8, base * 16]      # [64, 128, 256, 512] with defaults
        if depths is None:
            depths = [2, 2, 5, 2]  # MaxViT-S style; change to [2,2,2,2] if you want parity in blocks.
        heads = [max(1, w // 32) for w in widths]               # head_dim=32 per MaxViT

        in_ch_total = num_input_channels + len(num_aux_levels_list)

        # --- Shallow "conv_ini" at full resolution + maxpool, matching your encoder front
        self.conv_ini = _DownConv(in_ch_total, base)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # --- Encoder stages: first stage no downsample, later stages downsample in first block
        self.enc1 = self._make_stage(base, widths[0], depths[0], heads[0], window_size, grid_size, first_downsample=False, dropout=dropout)
        self.enc2 = self._make_stage(widths[0], widths[1], depths[1], heads[1], window_size, grid_size, first_downsample=True,  dropout=dropout)
        self.enc3 = self._make_stage(widths[1], widths[2], depths[2], heads[2], window_size, grid_size, first_downsample=True,  dropout=dropout)
        self.enc4 = self._make_stage(widths[2], widths[3], depths[3], heads[3], window_size, grid_size, first_downsample=True,  dropout=dropout)

        # --- Decoder: same shapes and order as your ResUNet_aux up path
        self.up3 = _UpConv(in_plane=widths[3], skip_plane=widths[2], num_plane=widths[2])
        self.up2 = _UpConv(in_plane=widths[2], skip_plane=widths[1], num_plane=widths[1])
        self.up1 = _UpConv(in_plane=widths[1], skip_plane=widths[0], num_plane=widths[0])
        self.up0 = _UpConv(in_plane=widths[0], skip_plane=base,      num_plane=base)

        self.out_conv = nn.Conv2d(base, 1, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)

    def _make_stage(self, in_ch, out_ch, depth, heads, P, G, first_downsample, dropout):
        blocks = []
        for i in range(depth):
            blocks.append(
                _MaxViTBlock(
                    in_ch if i == 0 else out_ch,
                    out_ch,
                    heads=heads,
                    window_size=P,
                    grid_size=G,
                    downsample=(first_downsample and i == 0),
                    mlp_ratio=4.0,
                    attn_dropout=0.0,
                    dropout=dropout
                )
            )
        return nn.Sequential(*blocks)

    def forward(self, x, x_aux):
        # 1) concat inputs exactly like your ResUNet_aux
        x = torch.cat([x, x_aux], dim=1)                         # (B, Cx+Ca, H, W)

        # 2) encoder with a full-res skip (x0) and four deeper features
        x0 = self.conv_ini(x)                                     # full resolution (skip-0)
        x_in = self.maxpool(x0)                                   # 1/2
        e1 = self.enc1(x_in)                                      # 1/2, C=widths[0]
        e2 = self.enc2(e1)                                        # 1/4, C=widths[1]
        e3 = self.enc3(e2)                                        # 1/8, C=widths[2]
        e4 = self.enc4(e3)                                        # 1/16, C=widths[3]

        # 3) decoder (4 ups) with the same skip ordering as your ResUNet_aux
        d3 = self.up3(e4, e3)   # 1/8
        d2 = self.up2(d3, e2)   # 1/4
        d1 = self.up1(d2, e1)   # 1/2
        d0 = self.up0(d1, x0)   # 1/1

        out = self.out_conv(d0)
        out = self.relu(out)     # keep non-negative outputs like your model
        return out

    def get_tot_prm(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)