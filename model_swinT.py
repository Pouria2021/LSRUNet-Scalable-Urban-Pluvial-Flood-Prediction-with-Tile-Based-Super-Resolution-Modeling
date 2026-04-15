"""Swin Transformer V2 U-Net for flood super-resolution prediction.

Implements SwinV2 encoder-decoder with:
    - Scaled cosine attention with learnable temperature per head
    - Log-spaced continuous position bias (Log-CPB)
    - Res-post-norm (LayerNorm after residual addition)
    - Patch merging for spatial downsampling

Supports both single-output (SwinV2) and multi-task (SwinV2_MTL) variants,
compatible with the ResUNet_aux training interface.
"""

import math
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

# GPU memory usage note:
# ------4-level-1024, batchsize-16, 65 M, ~25 GB
# ------4-level-1024, batchsize-24, 65 M, ~37 GB


# -------------------------
# Small UNet-style conv blocks (BN + ReLU) for decoder, like your ResUNet_aux
# -------------------------
class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x): return self.net(x)


class _Up(nn.Module):
    def __init__(self, in_plane: int, skip_plane: int, out_plane: int):
        super().__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.skip_proj = nn.Conv2d(skip_plane, in_plane, kernel_size=1, bias=False)
        self.conv = _ConvBlock(in_plane * 2, out_plane)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, self.skip_proj(skip)], dim=1)
        return self.conv(x)


# -------------------------
# Utilities for SwinV2
# -------------------------
def _window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """x: (B, H, W, C) -> (B*nW, ws*ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size * window_size, C)
    return windows


def _window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    """windows: (B*nW, ws*ws, C) -> (B, H, W, C)"""
    B = windows.shape[0] // (H // window_size * W // window_size)
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, H, W, -1)
    return x


class _LogCPB(nn.Module):
    """
    Log-spaced Continuous Position Bias (CPB) for SwinV2.
    Generates per-head bias for any relative coord in a ws×ws window.  (SwinV2 §3.3)
    """
    def __init__(self, window_size: int, num_heads: int, hidden: int = 64):
        super().__init__()
        self.window_size = window_size
        self.num_heads = num_heads
        self.mlp = nn.Sequential(
            nn.Linear(2, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, num_heads)
        )
        # Precompute relative coords for this window
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), torch.arange(window_size), indexing="ij"
        ))  # (2, ws, ws)
        coords_flat = coords.flatten(1)  # (2, L)
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # (2, L, L)
        rel = rel.permute(1, 2, 0).contiguous()  # (L, L, 2)
        self.register_buffer("rel_coords", rel, persistent=False)  # int coords

    def forward(self) -> torch.Tensor:
        # log-spaced coordinates per SwinV2: sign(x)*log(1+|x|)
        rel = self.rel_coords.to(dtype=torch.float32)
        rel = torch.sign(rel) * torch.log1p(torch.abs(rel))
        B = self.mlp(rel)  # (L, L, heads)
        return B.permute(2, 0, 1)  # (heads, L, L)


class _SwinV2Attention(nn.Module):
    """
    Windowed multi-head self-attention with:
      • scaled cosine attention (learnable τ per head),
      • log-spaced continuous position bias (Log-CPB).   (SwinV2 §3.2–3.3)
    """
    def __init__(self, dim: int, num_heads: int, window_size: int, qkv_bias: bool = True, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # τ: one learnable scalar per head
        self.tau = nn.Parameter(torch.ones(num_heads))
        self.log_cpb = _LogCPB(window_size, num_heads)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x: (B*nW, N, C) where N = ws*ws
        mask: (nW, N, N) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B_, heads, N, d)

        # Scaled cosine attention  (normalize queries/keys to unit length, divide by τ)
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        attn = q @ k.transpose(-2, -1)  # (B_, heads, N, N)
        tau = self.tau.clamp(min=0.01).view(1, -1, 1, 1)
        attn = attn / tau

        # Add Log-CPB
        cpb = self.log_cpb()  # (heads, N, N)
        attn = attn + cpb.unsqueeze(0)

        # Optional attention mask for shifted windows (add -inf to disallowed connections)
        if mask is not None:
            # attn: (B_* , nH, N, N) with B_* = B * nW
            B_, nH, N, _ = attn.shape
            nW = mask.shape[0]                        # mask: (nW, N, N)
            B = B_ // nW

            # Make sure mask matches dtype/device
            mask = mask.to(attn.device, dtype=attn.dtype)   # (nW, N, N)

            # Expand mask over batch, keep singleton over heads
            mask = mask.unsqueeze(0).repeat(B, 1, 1, 1)     # (B, nW, N, N)
            mask = mask.view(B_, 1, N, N)                   # (B*nW, 1, N, N)

            # Add (broadcasts over heads)
            attn = attn + mask

        attn = F.softmax(attn, dim=-1)
        y = attn @ v  # (B_, heads, N, d)
        y = y.transpose(1, 2).reshape(B_, N, C)
        y = self.proj_drop(self.proj(y))
        return y


class _Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))


def _calc_attn_mask(H: int, W: int, window_size: int, shift_size: int, device) -> Optional[torch.Tensor]:
    """
    Create attention mask for SW-MSA so tokens in different windows (after shift) do not attend to each other.  (Swin)
    Returns (nW, N, N) or None if shift_size==0.
    """
    if shift_size == 0:
        return None

    # Pad height/width up to multiples of window_size (same as feature padding in the block)
    Hp = int(math.ceil(H / window_size) * window_size)
    Wp = int(math.ceil(W / window_size) * window_size)

    img_mask = torch.zeros((1, Hp, Wp, 1), device=device)  # (1, Hp, Wp, 1)
    cnt = 0
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    # Partition the PADDED mask into windows
    mask_windows = _window_partition(img_mask, window_size)  # (nW, ws*ws, 1)
    mask_windows = mask_windows.view(-1, window_size * window_size)

    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, 0.0)
    return attn_mask  # (nW, N, N)


class _SwinV2Block(nn.Module):
    """
    One SwinV2 block with W-MSA and SW-MSA, both using res-post-norm (norm after residual add).
    """
    def __init__(self, dim: int, num_heads: int, window_size: int = 7, shift_size: int = 0,
                 mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.shift_size = shift_size
        self.attn = _SwinV2Attention(dim, num_heads, window_size)
        self.mlp = _Mlp(dim, mlp_ratio, drop)
        # res-post-norm: put LN AFTER residual add (both paths)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, H: int, W: int, attn_mask: Optional[torch.Tensor]):
        """
        x: (B, H*W, C) channels-last
        """
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)

        # cyclic shift if needed
        if self.shift_size > 0:
            shifted = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted = x

        # pad to multiples of window
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        if pad_b or pad_r:
            shifted = F.pad(shifted, (0, 0, 0, pad_r, 0, pad_b), mode="replicate")
        Hp, Wp = shifted.shape[1], shifted.shape[2]

        windows = _window_partition(shifted, self.window_size)  # (B*nW, N, C)
        y = self.attn(windows, mask=attn_mask)
        shifted = _window_reverse(y, self.window_size, Hp, Wp)

        # remove padding
        if pad_b or pad_r:
            shifted = shifted[:, :H, :W, :]

        # reverse shift
        if self.shift_size > 0:
            x_attn = torch.roll(shifted, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x_attn = shifted

        # res + post-norm
        x = x + x_attn
        x = self.norm1(x.view(B, H * W, C))

        # MLP path
        y = self.mlp(x)
        x = x + y
        x = self.norm2(x)
        return x  # (B, H*W, C)


class _PatchEmbed(nn.Module):
    """
    Overlapping patch embed via Conv (kernel=stride=patch_size). Here we use patch_size=2 to start at 1/2 res.
    """
    def __init__(self, in_ch: int, embed_dim: int, patch_size: int = 2):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor):
        x = self.proj(x)  # (B, C, H/ps, W/ps)
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)
        x = self.norm(x)
        return x, H, W


class _PatchMerging(nn.Module):
    """
    2x downsample: concat 2x2 neighbors (4*C) -> Linear(4C -> 2C) like Swin.  (keeps hierarchy C->2C)
    """
    def __init__(self, dim: int):
        super().__init__()
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = nn.LayerNorm(4 * dim)

    def forward(self, x: torch.Tensor, H: int, W: int):
        B, L, C = x.shape
        assert L == H * W
        x = x.view(B, H, W, C)

        pad_b = H % 2
        pad_r = W % 2
        if pad_b or pad_r:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b), mode="replicate")

        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1)  # (B, H/2, W/2, 4C)
        x = x.view(B, -1, 4 * C)
        x = self.norm(x)
        x = self.reduction(x)  # (B, H/2*W/2, 2C)
        H2, W2 = (H + pad_b) // 2, (W + pad_r) // 2
        return x, H2, W2


class _BasicLayer(nn.Module):
    """
    A SwinV2 stage: depth blocks with alternating shift, optional patch merging at end.
    """
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: int, mlp_ratio: float, downsample: bool):
        super().__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            self.blocks.append(
                _SwinV2Block(
                    dim=dim,
                    num_heads=num_heads,
                    window_size=window_size,
                    shift_size=0 if (i % 2 == 0) else window_size // 2,  # W-MSA then SW-MSA
                    mlp_ratio=mlp_ratio,
                )
            )
        self.downsample = _PatchMerging(dim) if downsample else None

    def forward(self, x: torch.Tensor, H: int, W: int):
        for blk in self.blocks:
            # Compute mask only for shifted blocks; reset to None for non-shifted blocks
            if blk.shift_size > 0:
                attn_mask = _calc_attn_mask(H, W, blk.window_size, blk.shift_size, x.device)
            else:
                attn_mask = None
            x = blk(x, H, W, attn_mask)

        # return stage output and possibly downsampled tensor for next stage
        if self.downsample is not None:
            x_down, H2, W2 = self.downsample(x, H, W)
            return x, H, W, x_down, H2, W2
        else:
            return x, H, W, None, None, None


# -------------------------
# SwinV2 UNet with the ResUNet_aux / MaxVIT(x, x_aux) interface
# -------------------------
class SwinV2(nn.Module):
    """
    Swin Transformer V2 U-Net for DEM correction (pixel-wise regression).

    Interface kept for drop-in compatibility with your training/test code:
      - __init__(num_input_channels, num_target_channels=512, num_levels=4, num_aux_levels_list=[4], ...)
      - forward(x, x_aux) -> (B,1,H,W) with ReLU
      - get_tot_prm()

    Defaults:
      • window_size=7 (used widely and recommended in SwinFlood to avoid boundary artifacts).
      • depths=[2,2,6,2] (Swin-T/S style).
      • patch_size=2 so stages sit at 1/2, 1/4, 1/8, 1/16 resolution (aligns to your 4-level decoder).
    """
    def __init__(
        self,
        num_input_channels: int = 2,
        num_target_channels: int = 512,
        num_levels: int = 4,
        num_aux_target_channels: List[int] = [512],  # kept for compatibility (unused)
        num_aux_levels_list: List[int] = [4],        # its length = #aux channels
        window_size: int = 8,
        depths: List[int] = (2, 2, 6, 2),
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_levels == 4, "This implementation expects num_levels=4."

        # Channel plan like your UNet: base -> 2x -> 4x -> 8x -> 16x
        base = int(num_target_channels // (2 ** num_levels))         # e.g., 512/16=32
        widths = [base * 2, base * 4, base * 8, base * 16]           # [64,128,256,512] default
        heads = [max(1, w // 32) for w in widths]                    # ~32 head dim (Swin practice).

        in_ch_total = num_input_channels + len(num_aux_levels_list)

        # High-res skip (full resolution), mirroring your ResUNet_aux/MaxVIT
        self.conv_ini = _ConvBlock(in_ch_total, base)

        # ---- SwinV2 encoder ----
        self.patch_embed = _PatchEmbed(in_ch_total, widths[0], patch_size=patch_size)

        # Stage 1..4
        self.stage1 = _BasicLayer(widths[0], depths[0], heads[0], window_size, mlp_ratio, downsample=True)
        self.stage2 = _BasicLayer(widths[1], depths[1], heads[1], window_size, mlp_ratio, downsample=True)
        self.stage3 = _BasicLayer(widths[2], depths[2], heads[2], window_size, mlp_ratio, downsample=True)
        self.stage4 = _BasicLayer(widths[3], depths[3], heads[3], window_size, mlp_ratio, downsample=False)

        # ---- UNet decoder ----
        self.up3 = _Up(in_plane=widths[3], skip_plane=widths[2], out_plane=widths[2])
        self.up2 = _Up(in_plane=widths[2], skip_plane=widths[1], out_plane=widths[1])
        self.up1 = _Up(in_plane=widths[1], skip_plane=widths[0], out_plane=widths[0])
        self.up0 = _Up(in_plane=widths[0], skip_plane=base,      out_plane=base)

        self.out_conv = nn.Conv2d(base, 1, kernel_size=1)
        self.out_act = nn.ReLU(inplace=False)  # preserve your output non-negativity

    def forward(self, x: torch.Tensor, x_aux: torch.Tensor) -> torch.Tensor:
        """
        x:     (B, num_input_channels, H, W)
        x_aux: (B, #aux_channels, H, W)
        """
        # Concat inputs exactly as in ResUNet_aux / MaxVIT
        xin = torch.cat([x, x_aux], dim=1)

        # High-res skip at full resolution
        x0 = self.conv_ini(xin)  # (B, base, H, W)

        # Patch embed (start encoder at 1/2 res)
        x1, H1, W1 = self.patch_embed(xin)           # (B, H/2*W/2, C1)
        # Stage 1
        e1, H1, W1, x2, H2, W2 = self.stage1(x1, H1, W1)  # e1: (B, H1*W1, C1)
        # Stage 2
        e2, H2, W2, x3, H3, W3 = self.stage2(x2, H2, W2)
        # Stage 3
        e3, H3, W3, x4, H4, W4 = self.stage3(x3, H3, W3)
        # Stage 4 (no downsample)
        e4, H4, W4, _, _, _ = self.stage4(x4, H4, W4)

        # reshape encoder outputs to BCHW
        def to_bchw(t, H, W, C): return t.view(t.size(0), H, W, C).permute(0, 3, 1, 2).contiguous()
        E1 = to_bchw(e1, H1, W1, e1.size(-1))
        E2 = to_bchw(e2, H2, W2, e2.size(-1))
        E3 = to_bchw(e3, H3, W3, e3.size(-1))
        E4 = to_bchw(e4, H4, W4, e4.size(-1))

        # ---- decoder ----
        d3 = self.up3(E4, E3)  # 1/8
        d2 = self.up2(d3, E2)  # 1/4
        d1 = self.up1(d2, E1)  # 1/2
        d0 = self.up0(d1, x0)  # 1/1

        out = self.out_conv(d0)
        return self.out_act(out)

    def get_tot_prm(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# -------------------------
# SwinV2 MTL (Multi-Task Learning) with dual-head prediction
# -------------------------
class SwinV2_MTL(nn.Module):
    """
    Swin Transformer V2 U-Net with Multi-Task Learning (MTL) dual-head prediction.

    Outputs:
      - out_depth: water depth prediction (non-negative, ReLU activation)
      - out_mask: flood probability (0-1 range, Sigmoid activation)

    Interface compatible with ResUNet_aux_MTL:
      - __init__(num_input_channels, num_target_channels=512, num_levels=4, ...)
      - forward(x, x_aux) -> (out_depth, out_mask)
      - get_tot_prm()
    """
    def __init__(
        self,
        num_input_channels: int = 2,
        num_target_channels: int = 512,
        num_levels: int = 4,
        num_aux_target_channels: List[int] = [512],
        num_aux_levels_list: List[int] = [4],
        window_size: int = 8,
        depths: List[int] = (2, 2, 6, 2),
        mlp_ratio: float = 4.0,
        patch_size: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        assert num_levels == 4, "This implementation expects num_levels=4."

        # Channel plan like UNet: base -> 2x -> 4x -> 8x -> 16x
        base = int(num_target_channels // (2 ** num_levels))
        widths = [base * 2, base * 4, base * 8, base * 16]
        heads = [max(1, w // 32) for w in widths]

        in_ch_total = num_input_channels + len(num_aux_levels_list)

        # ===== Shared Encoder (same as SwinV2) =====
        self.conv_ini = _ConvBlock(in_ch_total, base)
        self.patch_embed = _PatchEmbed(in_ch_total, widths[0], patch_size=patch_size)

        self.stage1 = _BasicLayer(widths[0], depths[0], heads[0], window_size, mlp_ratio, downsample=True)
        self.stage2 = _BasicLayer(widths[1], depths[1], heads[1], window_size, mlp_ratio, downsample=True)
        self.stage3 = _BasicLayer(widths[2], depths[2], heads[2], window_size, mlp_ratio, downsample=True)
        self.stage4 = _BasicLayer(widths[3], depths[3], heads[3], window_size, mlp_ratio, downsample=False)

        # ===== Depth Decoder Branch =====
        self.up3_depth = _Up(in_plane=widths[3], skip_plane=widths[2], out_plane=widths[2])
        self.up2_depth = _Up(in_plane=widths[2], skip_plane=widths[1], out_plane=widths[1])
        self.up1_depth = _Up(in_plane=widths[1], skip_plane=widths[0], out_plane=widths[0])
        self.up0_depth = _Up(in_plane=widths[0], skip_plane=base, out_plane=base)
        self.out_conv_depth = nn.Conv2d(base, 1, kernel_size=1)
        self.out_act_depth = nn.ReLU(inplace=False)

        # ===== Mask Decoder Branch =====
        self.up3_mask = _Up(in_plane=widths[3], skip_plane=widths[2], out_plane=widths[2])
        self.up2_mask = _Up(in_plane=widths[2], skip_plane=widths[1], out_plane=widths[1])
        self.up1_mask = _Up(in_plane=widths[1], skip_plane=widths[0], out_plane=widths[0])
        self.up0_mask = _Up(in_plane=widths[0], skip_plane=base, out_plane=base)
        self.out_conv_mask = nn.Conv2d(base, 1, kernel_size=1)
        self.out_act_mask = nn.Sigmoid()

    def forward(self, x: torch.Tensor, x_aux: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x:     (B, num_input_channels, H, W)
        x_aux: (B, #aux_channels, H, W)

        Returns:
            out_depth: (B, 1, H, W) - water depth prediction (non-negative)
            out_mask:  (B, 1, H, W) - flood probability (0-1 range)
        """
        # Concat inputs
        xin = torch.cat([x, x_aux], dim=1)

        # High-res skip at full resolution
        x0 = self.conv_ini(xin)

        # Patch embed (start encoder at 1/2 res)
        x1, H1, W1 = self.patch_embed(xin)

        # Encoder stages
        e1, H1, W1, x2, H2, W2 = self.stage1(x1, H1, W1)
        e2, H2, W2, x3, H3, W3 = self.stage2(x2, H2, W2)
        e3, H3, W3, x4, H4, W4 = self.stage3(x3, H3, W3)
        e4, H4, W4, _, _, _ = self.stage4(x4, H4, W4)

        # Reshape encoder outputs to BCHW
        def to_bchw(t, H, W, C): return t.view(t.size(0), H, W, C).permute(0, 3, 1, 2).contiguous()
        E1 = to_bchw(e1, H1, W1, e1.size(-1))
        E2 = to_bchw(e2, H2, W2, e2.size(-1))
        E3 = to_bchw(e3, H3, W3, e3.size(-1))
        E4 = to_bchw(e4, H4, W4, e4.size(-1))

        # ===== Depth Decoder Branch =====
        d3_depth = self.up3_depth(E4, E3)
        d2_depth = self.up2_depth(d3_depth, E2)
        d1_depth = self.up1_depth(d2_depth, E1)
        d0_depth = self.up0_depth(d1_depth, x0)
        out_depth = self.out_act_depth(self.out_conv_depth(d0_depth))

        # ===== Mask Decoder Branch =====
        d3_mask = self.up3_mask(E4, E3)
        d2_mask = self.up2_mask(d3_mask, E2)
        d1_mask = self.up1_mask(d2_mask, E1)
        d0_mask = self.up0_mask(d1_mask, x0)
        out_mask = self.out_act_mask(self.out_conv_mask(d0_mask))

        return out_depth, out_mask

    def get_tot_prm(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
