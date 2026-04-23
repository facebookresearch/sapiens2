# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Linear, Sequential


# ----------------------------------------------------------------------------
def to_2tuple(x):
    if isinstance(x, (str, bytes)):
        return (x, x)
    if isinstance(x, Sequence):
        x = tuple(x)
        if len(x) == 2:
            return x
        raise ValueError("Expected scalar or length-2 iterable")
    return (x, x)


def resize_pos_embed(
    pos_embed, src_shape, dst_shape, mode="bicubic", num_extra_tokens=1
):
    if src_shape[0] == dst_shape[0] and src_shape[1] == dst_shape[1]:
        return pos_embed
    assert pos_embed.ndim == 3, "shape of pos_embed must be [1, L, C]"
    _, L, C = pos_embed.shape
    src_h, src_w = src_shape
    assert L == src_h * src_w + num_extra_tokens, (
        f"The length of `pos_embed` ({L}) doesn't match the expected "
        f"shape ({src_h}*{src_w}+{num_extra_tokens}). Please check the"
        "`img_size` argument."
    )
    extra_tokens = pos_embed[:, :num_extra_tokens]

    src_weight = pos_embed[:, num_extra_tokens:]
    src_weight = src_weight.reshape(1, src_h, src_w, C).permute(0, 3, 1, 2)

    # The cubic interpolate algorithm only accepts float32
    dst_weight = F.interpolate(
        src_weight.float(), size=dst_shape, align_corners=False, mode=mode
    )
    dst_weight = torch.flatten(dst_weight, 2).transpose(1, 2)
    dst_weight = dst_weight.to(src_weight.dtype)

    return torch.cat((extra_tokens, dst_weight), dim=1)


# ----------------------------------------------------------------------------
class AdaptivePadding(nn.Module):
    def __init__(self, kernel_size=1, stride=1, dilation=1, padding="corner"):
        super().__init__()
        assert padding in ("same", "corner")

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        self.padding = padding
        self.kernel_size = kernel_size
        self.stride = stride
        self.dilation = dilation

    def get_pad_shape(self, input_shape):
        input_h, input_w = input_shape
        kernel_h, kernel_w = self.kernel_size
        stride_h, stride_w = self.stride
        output_h = math.ceil(input_h / stride_h)
        output_w = math.ceil(input_w / stride_w)
        pad_h = max(
            (output_h - 1) * stride_h + (kernel_h - 1) * self.dilation[0] + 1 - input_h,
            0,
        )
        pad_w = max(
            (output_w - 1) * stride_w + (kernel_w - 1) * self.dilation[1] + 1 - input_w,
            0,
        )
        return pad_h, pad_w

    def forward(self, x):
        pad_h, pad_w = self.get_pad_shape(x.size()[-2:])
        if pad_h > 0 or pad_w > 0:
            if self.padding == "corner":
                x = F.pad(x, [0, pad_w, 0, pad_h])
            elif self.padding == "same":
                x = F.pad(
                    x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2]
                )
        return x


# ----------------------------------------------------------------------------
class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_channels=3,
        embed_dims=768,
        kernel_size=16,
        stride=16,
        padding="corner",
        dilation=1,
        bias=True,
        input_size=None,
    ):
        super().__init__()

        self.embed_dims = embed_dims
        if stride is None:
            stride = kernel_size

        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        dilation = to_2tuple(dilation)

        if isinstance(padding, str):
            self.adaptive_padding = AdaptivePadding(
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                padding=padding,
            )
            padding = 0
        else:
            self.adaptive_padding = None
        padding = to_2tuple(padding)

        self.projection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=embed_dims,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )

        if input_size:
            input_size = to_2tuple(input_size)
            self.init_input_size = input_size
            if self.adaptive_padding:
                pad_h, pad_w = self.adaptive_padding.get_pad_shape(input_size)
                input_h, input_w = input_size
                input_h = input_h + pad_h
                input_w = input_w + pad_w
                input_size = (input_h, input_w)

            h_out = (
                input_size[0] + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1
            ) // stride[0] + 1
            w_out = (
                input_size[1] + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1
            ) // stride[1] + 1
            self.init_out_size = (h_out, w_out)
        else:
            self.init_input_size = None
            self.init_out_size = None

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        if self.adaptive_padding:
            x = self.adaptive_padding(x)

        x = self.projection(x)
        out_size = (x.shape[2], x.shape[3])
        x = x.flatten(2).transpose(1, 2)
        return x, out_size


# ----------------------------------------------------------------------------
class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        inplace: bool = False,
        data_format: str = "channels_last",
        scale: float = 1e-5,
    ):
        super().__init__()
        assert data_format in (
            "channels_last",
            "channels_first",
        ), "'data_format' could only be channels_last or channels_first."
        self.inplace = inplace
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(dim) * scale)

    def forward(self, x) -> torch.Tensor:
        if self.data_format == "channels_first":
            shape = tuple((1, -1, *(1 for _ in range(x.dim() - 2))))
        else:
            shape = tuple((*(1 for _ in range(x.dim() - 1)), -1))
        if self.inplace:
            return x.mul_(self.weight.view(*shape))
        else:
            return x * self.weight.view(*shape)


# ----------------------------------------------------------------------------
class FFN(nn.Module):
    def __init__(
        self,
        embed_dims=256,
        feedforward_channels=1024,
        num_fcs=2,
        ffn_drop=0.0,
        add_identity=True,
        layer_scale_init_value=0.0,
    ):
        super().__init__()
        assert num_fcs >= 2, f"num_fcs should be no less than 2. got {num_fcs}."
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs

        layers = []
        in_channels = embed_dims
        for _ in range(num_fcs - 1):
            layers.append(
                Sequential(
                    Linear(in_channels, feedforward_channels),
                    nn.GELU(),
                    nn.Dropout(ffn_drop),
                )
            )
            in_channels = feedforward_channels
        layers.append(Linear(feedforward_channels, embed_dims))
        layers.append(nn.Dropout(ffn_drop))
        self.layers = Sequential(*layers)
        self.dropout_layer = nn.Identity()
        self.add_identity = add_identity

        if layer_scale_init_value > 0:
            self.gamma2 = LayerScale(embed_dims, scale=layer_scale_init_value)
        else:
            self.gamma2 = nn.Identity()

    def forward(self, x, identity=None):
        out = self.layers(x)
        out = self.gamma2(out)
        if not self.add_identity:
            return out
        if identity is None:
            identity = x
        return identity + out


# ----------------------------------------------------------------------------
class MultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads,
        input_dims=None,
        attn_drop=0.0,
        proj_drop=0.0,
        qkv_bias=True,
        proj_bias=True,
        v_shortcut=False,
    ):
        super(MultiheadAttention, self).__init__()

        self.input_dims = input_dims or embed_dims
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.v_shortcut = v_shortcut

        self.head_dims = embed_dims // num_heads
        self.scaled_dot_product_attention = F.scaled_dot_product_attention

        self.qkv = nn.Linear(self.input_dims, embed_dims * 3, bias=qkv_bias)
        self.attn_drop = attn_drop
        self.proj = nn.Linear(embed_dims, embed_dims, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)
        self.gamma1 = nn.Identity()

    def forward(self, x):
        B, N, _ = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dims)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn_drop = self.attn_drop if self.training else 0.0
        x = self.scaled_dot_product_attention(q, k, v, dropout_p=attn_drop)
        x = x.transpose(1, 2).reshape(B, N, self.embed_dims)

        x = self.proj(x)
        x = self.gamma1(self.proj_drop(x))

        if self.v_shortcut:
            x = v.squeeze(1) + x
        return x


# ----------------------------------------------------------------------------
class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        embed_dims,
        num_heads,
        feedforward_channels,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        num_fcs=2,
        qkv_bias=True,
    ):
        super(TransformerEncoderLayer, self).__init__()

        self.embed_dims = embed_dims
        self.ln1 = nn.LayerNorm(self.embed_dims, eps=1e-6, elementwise_affine=True)
        self.attn = MultiheadAttention(
            embed_dims=embed_dims,
            num_heads=num_heads,
            attn_drop=attn_drop_rate,
            proj_drop=drop_rate,
            qkv_bias=qkv_bias,
        )

        self.ln2 = nn.LayerNorm(self.embed_dims, eps=1e-6, elementwise_affine=True)
        self.ffn = FFN(
            embed_dims=embed_dims,
            feedforward_channels=feedforward_channels,
            num_fcs=num_fcs,
            ffn_drop=drop_rate,
            add_identity=True,
        )

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = self.ffn(self.ln2(x), identity=x)
        return x


# ----------------------------------------------------------------------------
class Sapiens(nn.Module):
    arch_zoo = {
        **dict.fromkeys(  ## this is vit-large
            ["0.3b", "sapiens_0.3b"],
            {
                "embed_dims": 1024,
                "num_layers": 24,
                "num_heads": 16,
                "feedforward_channels": 1024 * 4,
            },
        ),
        **dict.fromkeys(  ## this is vit-huge
            ["0.6b", "sapiens_0.6b"],
            {
                "embed_dims": 1280,
                "num_layers": 32,
                "num_heads": 16,
                "feedforward_channels": 1280 * 4,
            },
        ),
        **dict.fromkeys(  ## this is vit-g
            ["1b", "sapiens_1b"],
            {
                "embed_dims": 1536,
                "num_layers": 40,
                "num_heads": 24,
                "feedforward_channels": 1536 * 4,
            },
        ),
        **dict.fromkeys(
            ["2b", "sapiens_2b"],
            {
                "embed_dims": 1920,
                "num_layers": 48,
                "num_heads": 32,
                "feedforward_channels": 1920 * 4,
            },
        ),
    }
    num_extra_tokens = 1  # class token
    OUT_TYPES = {"raw", "cls_token", "featmap", "avg_featmap"}

    def __init__(
        self,
        arch="base",
        img_size=224,
        patch_size=16,
        in_channels=3,
        out_indices=-1,
        drop_rate=0.0,
        qkv_bias=True,
        final_norm=True,
        out_type="cls_token",
        with_cls_token=True,
        frozen_stages=-1,
        interpolate_mode="bicubic",
        patch_cfg=dict(),
        layer_cfgs=dict(),
    ):
        super(Sapiens, self).__init__()

        arch = arch.lower()
        assert arch in set(self.arch_zoo), (
            f"Arch {arch} is not in default archs {set(self.arch_zoo)}"
        )
        self.arch_settings = self.arch_zoo[arch]

        self.embed_dims = self.arch_settings["embed_dims"]
        self.num_layers = self.arch_settings["num_layers"]
        self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size

        # Set patch embedding
        _patch_cfg = dict(
            in_channels=in_channels,
            input_size=img_size,
            embed_dims=self.embed_dims,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )
        _patch_cfg.update(patch_cfg)
        self.patch_embed = PatchEmbed(**_patch_cfg)
        self.patch_resolution = self.patch_embed.init_out_size
        num_patches = self.patch_resolution[0] * self.patch_resolution[1]

        # Set out type
        if out_type not in self.OUT_TYPES:
            raise ValueError(
                f"Unsupported `out_type` {out_type}, please "
                f"choose from {self.OUT_TYPES}"
            )
        self.out_type = out_type

        # Set cls token
        self.with_cls_token = with_cls_token
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, self.embed_dims))
        elif out_type != "cls_token":
            self.cls_token = None
            self.num_extra_tokens = 0
        else:
            raise ValueError('with_cls_token must be True when `out_type="cls_token"`.')

        # Set position embedding
        self.interpolate_mode = interpolate_mode
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + self.num_extra_tokens, self.embed_dims)
        )
        self.drop_after_pos = nn.Dropout(p=drop_rate)

        if isinstance(out_indices, int):
            out_indices = [out_indices]
        assert isinstance(out_indices, Sequence), (
            f'"out_indices" must by a sequence or int, get {type(out_indices)} instead.'
        )
        for i, index in enumerate(out_indices):
            if index < 0:
                out_indices[i] = self.num_layers + index
            assert 0 <= out_indices[i] <= self.num_layers, (
                f"Invalid out_indices {index}"
            )
        self.out_indices = out_indices

        self.layers = nn.Sequential()
        if isinstance(layer_cfgs, dict):
            layer_cfgs = [layer_cfgs] * self.num_layers
        for i in range(self.num_layers):
            _layer_cfg = dict(
                embed_dims=self.embed_dims,
                num_heads=self.arch_settings["num_heads"],
                feedforward_channels=self.arch_settings["feedforward_channels"],
                drop_rate=drop_rate,
                qkv_bias=qkv_bias,
            )
            _layer_cfg.update(layer_cfgs[i])
            self.layers.append(TransformerEncoderLayer(**_layer_cfg))

        self.frozen_stages = frozen_stages
        self.pre_norm = nn.Identity()

        self.final_norm = final_norm
        if final_norm:
            self.ln1 = nn.LayerNorm(self.embed_dims, eps=1e-6, elementwise_affine=True)

        # freeze stages only when self.frozen_stages > 0
        if self.frozen_stages > 0:
            self._freeze_stages()

        self._register_load_state_dict_pre_hook(self._prepare_pos_embed)

        return

    def _prepare_pos_embed(self, state_dict, prefix, *args, **kwargs):
        name = prefix + "pos_embed"
        if name not in state_dict.keys():
            return

        ckpt_pos_embed_shape = state_dict[name].shape

        # Handle class token removal if needed
        if not self.with_cls_token:
            if ckpt_pos_embed_shape[1] == self.pos_embed.shape[1] + 1:
                # Remove cls token from state dict if it's not used
                state_dict[name] = state_dict[name][:, 1:]
                ckpt_pos_embed_shape = state_dict[name].shape
            elif ckpt_pos_embed_shape[1] % 2 == 1:
                # Remove class token when interpolation is required
                state_dict[name] = state_dict[name][:, 1:]
                ckpt_pos_embed_shape = state_dict[name].shape

        # Skip if shapes already match
        if self.pos_embed.shape == ckpt_pos_embed_shape:
            return

        # Calculate grid dimensions
        pos_h, pos_w = self.patch_embed.init_out_size
        assert pos_h >= pos_w  # for vertical aspect ratio or square

        # Number of non-extra tokens in checkpoint
        num_vis = ckpt_pos_embed_shape[1] - self.num_extra_tokens

        # Determine original grid shape
        side = int(math.sqrt(num_vis))
        factor = int(math.sqrt((num_vis * self.patch_size * self.patch_size) // 12))

        # Set old grid based on aspect ratio detection
        if side * side == num_vis:
            old_grid = (side, side)  # square grid
        elif 4 * factor * 3 * factor == num_vis * self.patch_size * self.patch_size:
            old_grid = (
                (factor * 4) // self.patch_size,
                (factor * 3) // self.patch_size,
            )  # 4:3 ratio
        else:
            state_dict[name] = self.pos_embed
            return

        # Resize position embedding
        new_grid = (pos_h, pos_w)
        state_dict[name] = resize_pos_embed(
            state_dict[name],
            old_grid,
            new_grid,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens,
        )

    @property
    def norm1(self):
        return self.ln1

    @property
    def norm2(self):
        return self.ln2

    @staticmethod
    def resize_pos_embed(*args, **kwargs):
        """Interface for backward-compatibility."""
        return resize_pos_embed(*args, **kwargs)

    def _freeze_stages(self):
        # freeze position embedding
        if self.pos_embed is not None:
            self.pos_embed.requires_grad = False

        # set dropout to eval model
        self.drop_after_pos.eval()
        # freeze patch embedding
        self.patch_embed.eval()
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        # freeze pre-norm
        for param in self.pre_norm.parameters():
            param.requires_grad = False
        # freeze cls_token
        if self.cls_token is not None:
            self.cls_token.requires_grad = False
        # freeze layers
        for i in range(1, self.frozen_stages + 1):
            m = self.layers[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False
        # freeze the last layer norm
        if self.frozen_stages == len(self.layers):
            if self.final_norm:
                self.ln1.eval()
                for param in self.ln1.parameters():
                    param.requires_grad = False

            if self.out_type == "avg_featmap":
                self.ln2.eval()
                for param in self.ln2.parameters():
                    param.requires_grad = False

    def forward(self, x):
        B = x.shape[0]
        x, patch_resolution = self.patch_embed(x)

        if self.cls_token is not None:
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)

        x = x + resize_pos_embed(
            self.pos_embed,
            self.patch_resolution,
            patch_resolution,
            mode=self.interpolate_mode,
            num_extra_tokens=self.num_extra_tokens,
        )
        x = self.drop_after_pos(x)

        x = self.pre_norm(x)  ## B x (num tokens) x embed_dim

        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)

            if i == len(self.layers) - 1 and self.final_norm:
                x = self.ln1(x)

            if i in self.out_indices:
                outs.append(self._format_output(x, patch_resolution))

        return tuple(outs)

    def _format_output(self, x, hw):
        if self.out_type == "raw":
            return x
        if self.out_type == "cls_token":
            return x[:, 0]

        patch_token = x[:, self.num_extra_tokens :]
        if self.out_type == "featmap":
            B = x.size(0)
            # (B, N, C) -> (B, H, W, C) -> (B, C, H, W)
            return patch_token.reshape(B, *hw, -1).permute(0, 3, 1, 2)
