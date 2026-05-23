# Pure-PyTorch reimplementation of the StyleGAN2 / NCSN++ ``upfirdn2d``
# op (no custom CUDA extensions). Matches the semantics
# ``upfirdn2d(input, kernel, up=1, down=1, pad=(pad0, pad1))``:
#   1) zero-insertion upsample, 2) symmetric pad (possibly negative ->
#   cropping), 3) FIR-kernel convolution (kernel is flipped),
#   4) stride downsample.
#
# Replaces the C++/CUDA kernel shipped in yang-song/score_sde_pytorch
# (which itself ports from NVlabs/stylegan2). See
# ``models/ncsnpp/__init__.py`` for the upstream reference.

import torch
import torch.nn.functional as F
from typing import Tuple


def _upfirdn2d_native(
    input: torch.Tensor,
    kernel: torch.Tensor,
    up_x: int,
    up_y: int,
    down_x: int,
    down_y: int,
    pad_x0: int,
    pad_x1: int,
    pad_y0: int,
    pad_y1: int,
) -> torch.Tensor:
    """
    PyTorch port of the original upfirdn2d_native from your op/upfirdn2d.py.

    input : (N, C, H, W)
    kernel: (Kh, Kw)
    up_x, up_y, down_x, down_y: ints
    pad_x0, pad_x1, pad_y0, pad_y1: ints
    """
    assert input.ndim == 4, f"Expected 4D input (N,C,H,W), got {input.shape}"
    assert kernel.ndim == 2, f"Expected 2D kernel (Kh,Kw), got {kernel.shape}"

    device = input.device
    dtype = input.dtype

    # Original logic fuses N and C into a single "batch" dimension and
    # treats the last dimension as a "minor" dimension.
    N, C, in_h, in_w = input.shape
    x = input.reshape(-1, in_h, in_w, 1)  # (N*C, H, W, minor=1)

    _, in_h, in_w, minor = x.shape
    kernel_h, kernel_w = kernel.shape

    # 1) Upsample by inserting zeros between pixels.
    #    This is implemented by reshaping + padding.
    #    out: (N*C, H, 1, W, 1, minor)
    x = x.view(-1, in_h, 1, in_w, 1, minor)
    # F.pad with 8 values pads the last 4 dims: (minor, W, 1, H)
    x = F.pad(x, [0, 0,           # minor dim: no pad
                  0, up_x - 1,    # width: pad right with up_x-1 zeros (insert between)
                  0, 0,           # singleton dim: no pad
                  0, up_y - 1])   # height: pad bottom with up_y-1 zeros
    x = x.view(-1, in_h * up_y, in_w * up_x, minor)  # (N*C, H*up_y, W*up_x, minor)

    # 2) Pad or crop according to pad_x*, pad_y*.
    # F.pad with 6 values pads last 3 dims: (minor, W, H)
    x = F.pad(
        x,
        [
            0, 0,                             # minor: no pad
            max(pad_x0, 0), max(pad_x1, 0),   # W: left, right
            max(pad_y0, 0), max(pad_y1, 0),   # H: top, bottom
        ],
    )

    # Crop for negative padding.
    x = x[
        :,
        max(-pad_y0, 0) : x.shape[1] - max(-pad_y1, 0),
        max(-pad_x0, 0) : x.shape[2] - max(-pad_x1, 0),
        :,
    ]  # (N*C, H', W', minor)

    # 3) Convolve with flipped kernel (FIR filtering).
    # Move "minor" to channel dimension to use conv2d.
    x = x.permute(0, 3, 1, 2)  # (N*C, minor, H', W')
    w = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    w = w.to(device=device, dtype=dtype)

    x = F.conv2d(x, w)  # (N*C, minor, H'' , W'')

    # Reshape back and downsample.
    x = x.reshape(
        -1,
        minor,
        in_h * up_y + pad_y0 + pad_y1 - kernel_h + 1,
        in_w * up_x + pad_x0 + pad_x1 - kernel_w + 1,
    )  # (N*C, minor, H_full, W_full)

    x = x.permute(0, 2, 3, 1)  # (N*C, H_full, W_full, minor)

    # 4) Downsample by striding.
    x = x[:, ::down_y, ::down_x, :]  # keep every down_y / down_x pixel

    out_h = (in_h * up_y + pad_y0 + pad_y1 - kernel_h) // down_y + 1
    out_w = (in_w * up_x + pad_x0 + pad_x1 - kernel_w) // down_x + 1

    x = x.view(-1, C, out_h, out_w)  # (N, C, out_h, out_w)
    return x


def upfirdn2d(
    input: torch.Tensor,
    kernel: torch.Tensor,
    up: int = 1,
    down: int = 1,
    padding: Tuple[int, int] = (0, 0),
) -> torch.Tensor:
    """
    Public interface matching the original op:

        upfirdn2d(input, kernel, up=1, down=1, pad=(pad0, pad1))

    plus compatibility with a newer signature that might use `padding=`.

    - `pad` and `padding` are symmetric in x/y:
        pad_x0 = pad_y0 = padding[0]
        pad_x1 = pad_y1 = padding[1]
    """

    pad_x0, pad_x1 = padding
    pad_y0, pad_y1 = padding

    up_x = up_y = up
    down_x = down_y = down

    return _upfirdn2d_native(
        input,
        kernel,
        up_x,
        up_y,
        down_x,
        down_y,
        pad_x0,
        pad_x1,
        pad_y0,
        pad_y1,
    )
