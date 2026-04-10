"""Turbo VAE Decoder ported to MLX.

Decodes video latents [B, 48, T, H, W] -> pixel video [B, 3, T*4, H*16, W*16].
All operations run on Apple GPU via MLX, bypassing PyTorch MPS int32 limits.

MLX uses channels-last: NTHWC instead of PyTorch's NCTHW.
"""

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def nearest_upsample_2x(x: mx.array) -> mx.array:
    """2x nearest spatial upsample. x: [B, T, H, W, C] or [B, H, W, C]."""
    if x.ndim == 5:
        B, T, H, W, C = x.shape
        x = mx.repeat(x, repeats=2, axis=2)  # H -> H*2
        x = mx.repeat(x, repeats=2, axis=3)  # W -> W*2
    elif x.ndim == 4:
        x = mx.repeat(x, repeats=2, axis=1)  # H -> H*2
        x = mx.repeat(x, repeats=2, axis=2)  # W -> W*2
    return x


def unpatchify(x: mx.array, patch_size: int) -> mx.array:
    """[B, T, H, W, C*p*p] -> [B, T, H*p, W*p, C]

    PyTorch (NCTHW) does: [b, c*p0*p1, f, h, w] -> [b, c, p0, p1, f, h, w]
    -> permute(0,1,4,5,3,6,2) -> [b, c, f, h, p1, w, p0] -> [b, c, f, h*p1, w*p0]

    MLX (NTHWC) equivalent: interleave p1 with H, p0 with W.
    """
    if patch_size == 1:
        return x
    B, T, H, W, Cpp = x.shape
    C = Cpp // (patch_size * patch_size)
    # Split channels into [C, p0, p1] matching PyTorch's view order
    x = x.reshape(B, T, H, W, C, patch_size, patch_size)
    # indices:     0  1  2  3  4  5(p0)      6(p1)
    # Target: [B, T, H, p1, W, p0, C] -> reshape to [B, T, H*p1, W*p0, C]
    x = x.transpose(0, 1, 2, 6, 3, 5, 4)
    x = x.reshape(B, T, H * patch_size, W * patch_size, C)
    return x


def channel_rms_norm(x: mx.array, eps: float = 1e-8, weight: mx.array = None) -> mx.array:
    """RMS normalization over channel (last) dimension."""
    variance = mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True)
    x = x * mx.rsqrt(variance + eps)
    if weight is not None:
        x = x * weight
    return x


# ---------------------------------------------------------------------------
# Core modules
# ---------------------------------------------------------------------------

class ChannelRMSNorm(nn.Module):
    """RMS normalization over channel dimension (last axis in NTHWC)."""

    def __init__(self, dim: int, eps: float = 1e-8, elementwise_affine: bool = True):
        super().__init__()
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = mx.ones((dim,))
        else:
            self.weight = None

    def __call__(self, x: mx.array) -> mx.array:
        return channel_rms_norm(x, self.eps, self.weight)


class CausalConv3d(nn.Module):
    """Conv3d with symmetric temporal edge-replication padding. MLX NTHWC layout."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.kernel_size = kernel_size
        self.time_pad = (kernel_size[0] - 1) // 2
        h_pad = kernel_size[1] // 2
        w_pad = kernel_size[2] // 2

        self.conv = nn.Conv3d(
            in_channels, out_channels, kernel_size,
            stride=stride, padding=(0, h_pad, w_pad), bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        # x: [B, T, H, W, C]
        if self.time_pad > 0:
            pad_left = mx.repeat(x[:, :1], repeats=self.time_pad, axis=1)
            pad_right = mx.repeat(x[:, -1:], repeats=self.time_pad, axis=1)
            x = mx.concatenate([pad_left, x, pad_right], axis=1)
        return self.conv(x)


class DepthwiseSeparableConv3d(nn.Module):
    """Depthwise separable Conv3d with causal temporal padding.

    Note: Not used in current config (decoder_is_dw_conv all False).
    MLX Conv3d lacks groups param, so this would need a manual loop
    over channels for true depthwise conv. Placeholder for future use.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=(3, 3, 3),
        stride=(1, 1, 1),
    ):
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride, stride)

        self.kernel_size = kernel_size
        self.time_pad = (kernel_size[0] - 1) // 2
        h_pad = kernel_size[1] // 2
        w_pad = kernel_size[2] // 2

        # Approximate: use regular conv instead of grouped (not used in current config)
        self.depthwise_conv = nn.Conv3d(
            in_channels, in_channels, kernel_size,
            stride=stride, padding=(0, h_pad, w_pad), bias=True,
        )
        self.pointwise_conv = nn.Conv3d(
            in_channels, out_channels, kernel_size=(1, 1, 1), bias=True,
        )

    def __call__(self, x: mx.array) -> mx.array:
        if self.time_pad > 0:
            pad_left = mx.repeat(x[:, :1], repeats=self.time_pad, axis=1)
            pad_right = mx.repeat(x[:, -1:], repeats=self.time_pad, axis=1)
            x = mx.concatenate([pad_left, x, pad_right], axis=1)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class ResnetBlock3d(nn.Module):
    """3D ResNet block with RMSNorm."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        is_upsampler_modified: bool = False,
        is_dw_conv: bool = False,
        dw_kernel_size: int = 3,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        self.is_upsampler_modified = is_upsampler_modified

        conv_cls = DepthwiseSeparableConv3d if is_dw_conv else CausalConv3d
        ks = dw_kernel_size if is_dw_conv else 3

        self.norm1 = ChannelRMSNorm(in_channels, elementwise_affine=False)
        self.conv1 = conv_cls(in_channels, out_channels, kernel_size=ks)
        self.norm2 = ChannelRMSNorm(out_channels, elementwise_affine=False)
        self.conv2 = conv_cls(out_channels, out_channels, kernel_size=ks)

        if in_channels != out_channels:
            self.norm3 = ChannelRMSNorm(in_channels, elementwise_affine=False)
            self.conv_shortcut = conv_cls(in_channels, out_channels, kernel_size=1)
        else:
            self.norm3 = None
            self.conv_shortcut = None

    def __call__(self, x: mx.array) -> mx.array:
        residual = x

        h = self.norm1(x)
        h = nn.relu(h) if self.is_upsampler_modified else nn.silu(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = nn.silu(h)
        h = self.conv2(h)

        if self.norm3 is not None:
            residual = self.norm3(residual)
        if self.conv_shortcut is not None:
            residual = self.conv_shortcut(residual)

        return h + residual


class WanResample(nn.Module):
    """Spatial (2D) or spatio-temporal (3D) upsampling."""

    def __init__(self, dim: int, mode: str, upsample_out_dim: int = None):
        super().__init__()
        self.mode = mode
        self.dim = dim
        upsample_out_dim = upsample_out_dim or dim // 2

        # Spatial: nearest 2x upsample + Conv2d
        self.spatial_conv = nn.Conv2d(dim, upsample_out_dim, kernel_size=3, padding=1)

        if mode == "upsample3d":
            self.time_conv = CausalConv3d(dim, dim * 2, kernel_size=(3, 1, 1))

    def __call__(self, x: mx.array, is_first_chunk: bool = True) -> mx.array:
        # x: [B, T, H, W, C]
        B, T, H, W, C = x.shape

        if self.mode == "upsample3d":
            # Temporal upsample: conv to 2*C channels, then interleave into time
            x = self.time_conv(x)  # [B, T, H, W, C*2]
            # Split channels and interleave into time
            x = x.reshape(B, T, H, W, 2, C)
            x = x.transpose(0, 1, 4, 2, 3, 5)  # [B, T, 2, H, W, C]
            x = x.reshape(B, T * 2, H, W, C)
            if is_first_chunk:
                x = x[:, 1:]  # Drop first frame
            T = x.shape[1]

        # Spatial upsample: reshape to (B*T, H, W, C), nearest 2x, conv2d
        x = x.reshape(B * T, H, W, C)
        x = x.astype(mx.float32)
        x = nearest_upsample_2x(x)  # (B*T, H*2, W*2, C)
        x = self.spatial_conv(x)
        new_C = x.shape[-1]
        x = x.reshape(B, T, H * 2, W * 2, new_C)
        return x


# ---------------------------------------------------------------------------
# Compound blocks
# ---------------------------------------------------------------------------

class MidBlock3d(nn.Module):
    def __init__(self, in_channels: int, num_layers: int, is_dw_conv: bool = False, dw_kernel_size: int = 3):
        super().__init__()
        self.resnets = [
            ResnetBlock3d(in_channels, in_channels, is_dw_conv=is_dw_conv, dw_kernel_size=dw_kernel_size)
            for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        for resnet in self.resnets:
            x = resnet(x)
        return x


class UpBlock3d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = None,
        num_layers: int = 1,
        spatio_temporal_scale: bool = True,
        is_dw_conv: bool = False,
        dw_kernel_size: int = 3,
        spatio_only: bool = False,
    ):
        super().__init__()
        out_channels = out_channels or in_channels

        if in_channels != out_channels:
            self.conv_in = ResnetBlock3d(in_channels, out_channels, is_dw_conv=is_dw_conv, dw_kernel_size=dw_kernel_size)
        else:
            self.conv_in = None

        if spatio_temporal_scale:
            mode = "upsample2d" if spatio_only else "upsample3d"
            self.upsamplers = [WanResample(out_channels, mode=mode, upsample_out_dim=out_channels)]
        else:
            self.upsamplers = None

        self.resnets = [
            ResnetBlock3d(out_channels, out_channels, is_upsampler_modified=spatio_temporal_scale,
                          is_dw_conv=is_dw_conv, dw_kernel_size=dw_kernel_size)
            for _ in range(num_layers)
        ]

    def __call__(self, x: mx.array, is_first_chunk: bool) -> mx.array:
        if self.conv_in is not None:
            x = self.conv_in(x)
        if self.upsamplers is not None:
            for up in self.upsamplers:
                x = up(x, is_first_chunk=is_first_chunk)
        for resnet in self.resnets:
            x = resnet(x)
        return x


# ---------------------------------------------------------------------------
# Top-level decoder
# ---------------------------------------------------------------------------

class Decoder3d(nn.Module):
    def __init__(
        self,
        in_channels: int = 48,
        out_channels: int = 3,
        block_out_channels=(64, 128, 256, 512),
        spatio_temporal_scaling=(False, True, True, True),
        layers_per_block=(2, 2, 2, 3, 3),
        patch_size: int = 2,
        is_dw_conv=(False, False, False, False, False),
        dw_kernel_size: int = 3,
        spatio_only=(False, False, False, False),
        use_unpatchify: bool = True,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.out_channels = out_channels
        self.use_unpatchify = use_unpatchify

        # Reverse to go from deepest to shallowest
        block_out_channels = tuple(reversed(block_out_channels))
        spatio_temporal_scaling = tuple(reversed(spatio_temporal_scaling))
        layers_per_block = tuple(reversed(layers_per_block))
        is_dw_conv = tuple(reversed(is_dw_conv))
        spatio_only = tuple(reversed(spatio_only))

        output_channel = block_out_channels[0]

        self.conv_in = CausalConv3d(in_channels, output_channel, kernel_size=3)

        self.mid_block = MidBlock3d(
            output_channel, num_layers=layers_per_block[0],
            is_dw_conv=is_dw_conv[0], dw_kernel_size=dw_kernel_size,
        )

        self.up_blocks = []
        for i in range(len(block_out_channels)):
            input_channel = output_channel
            output_channel = block_out_channels[i]
            self.up_blocks.append(UpBlock3d(
                in_channels=input_channel,
                out_channels=output_channel,
                num_layers=layers_per_block[i + 1],
                spatio_temporal_scale=spatio_temporal_scaling[i],
                is_dw_conv=is_dw_conv[i + 1],
                dw_kernel_size=dw_kernel_size,
                spatio_only=spatio_only[i],
            ))

        # Output conv: channels -> out_channels * patch_size^2 for unpatchify
        conv_out_channels = out_channels * patch_size * patch_size if use_unpatchify else out_channels
        self.conv_out = CausalConv3d(output_channel, conv_out_channels, kernel_size=3)

    def __call__(self, x: mx.array, is_first_chunk: bool) -> mx.array:
        x = self.conv_in(x)
        x = self.mid_block(x)

        for up_block in self.up_blocks:
            x = up_block(x, is_first_chunk=is_first_chunk)

        # Inline RMSNorm over channels
        x = channel_rms_norm(x, eps=1e-8)
        x = nn.silu(x)
        x = self.conv_out(x)

        if self.use_unpatchify:
            x = unpatchify(x, self.patch_size)

        return x


class TurboVAED(nn.Module):
    """MLX Turbo VAE Decoder with sliding-window temporal decode."""

    def __init__(
        self,
        latent_channels: int = 48,
        out_channels: int = 3,
        block_out_channels=(64, 128, 256, 512),
        spatio_temporal_scaling=(False, True, True, True),
        layers_per_block=(2, 2, 2, 3, 3),
        patch_size: int = 2,
        is_dw_conv=(False, False, False, False, False),
        dw_kernel_size: int = 3,
        spatio_only=(False, False, False, False),
        use_unpatchify: bool = True,
        first_chunk_size: int = 7,
        step_size: int = 7,
        temporal_compression_ratio: int = 4,
    ):
        super().__init__()
        self.z_dim = latent_channels
        self.first_chunk_size = first_chunk_size
        self.step_size = step_size
        self.temporal_compression_ratio = temporal_compression_ratio

        self.decoder = Decoder3d(
            in_channels=latent_channels,
            out_channels=out_channels,
            block_out_channels=block_out_channels,
            spatio_temporal_scaling=spatio_temporal_scaling,
            layers_per_block=layers_per_block,
            patch_size=patch_size,
            is_dw_conv=is_dw_conv,
            dw_kernel_size=dw_kernel_size,
            spatio_only=spatio_only,
            use_unpatchify=use_unpatchify,
        )

    def decode(self, z: mx.array, mean: mx.array, inv_std: mx.array) -> mx.array:
        """Decode latent video.

        Args:
            z: [B, T, H, W, C] latent (NTHWC)
            mean: [48] channel mean
            inv_std: [48] channel 1/std

        Returns:
            [B, T', H', W', C'] decoded video (NTHWC)
        """
        # Denormalize
        z = z / inv_std + mean

        first_chunk_size = self.first_chunk_size
        step = self.step_size
        num_overlap_pixel_frames = 1 * self.temporal_compression_ratio

        num_frames = z.shape[1]  # T dimension in NTHWC

        # Pad frames for chunking
        num_padding_frames = 0
        if num_frames < first_chunk_size:
            num_padding_frames = first_chunk_size - num_frames
        elif (num_frames - first_chunk_size) % step != 0:
            num_padding_frames = step - (num_frames - first_chunk_size) % step

        if num_padding_frames > 0:
            z = mx.concatenate([z, mx.repeat(z[:, -1:], repeats=num_padding_frames, axis=1)], axis=1)
            num_frames = num_frames + num_padding_frames

        # Sliding window decode
        out_chunks = []

        if num_frames == first_chunk_size:
            out = self.decoder(z, is_first_chunk=True)
            mx.eval(out)
            out_chunks.append(out)
        else:
            # First chunk
            out = self.decoder(z[:, :first_chunk_size + 1], is_first_chunk=True)
            out = out[:, :-num_overlap_pixel_frames]
            mx.eval(out)
            out_chunks.append(out)

            # Middle and last chunks
            for i in range(first_chunk_size, num_frames, step):
                is_last_chunk = i + step == num_frames
                left = i - 1
                right = i + step + 1 if not is_last_chunk else i + step

                out_ = self.decoder(z[:, left:right], is_first_chunk=False)

                if is_last_chunk:
                    out_ = out_[:, num_overlap_pixel_frames:]
                else:
                    out_ = out_[:, num_overlap_pixel_frames:-num_overlap_pixel_frames]

                mx.eval(out_)
                out_chunks.append(out_)

        out = mx.concatenate(out_chunks, axis=1)

        # Remove padding
        if num_padding_frames > 0:
            out = out[:, :-num_padding_frames * self.temporal_compression_ratio]

        return out
