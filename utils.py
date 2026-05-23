"""
Utilities for SKILD: DCT/IDCT, k-space grids, radial binning,
power-law spectra, logging.

The DCT-II / DCT-III pair is implemented with an O(N log N) FFT-based
algorithm so it is competitive with native ``torch.fft`` on GPU. All
norms (``None``, ``'backward'``, ``'forward'``, ``'ortho'``) are
supported; the SKILD code uses ``norm='forward'`` for both directions to
keep the per-mode amplitude factor explicit.
"""

from __future__ import annotations
import math
import logging
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import numpy as np
import numpy.typing as npt

ArrayF = npt.NDArray[np.float64]


# --------------------------------------------------------------------- #
# Logging                                                                #
# --------------------------------------------------------------------- #
def create_logger(logging_dir):
    """Create a logger that writes to a log file (rank 0 only) and stdout."""
    if dist.get_rank() == 0:
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{logging_dir}/log.txt"),
            ],
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


# --------------------------------------------------------------------- #
# FFT-based DCT-II / DCT-III (1D and 2D)                                 #
# --------------------------------------------------------------------- #
_dct_cache: dict = {}


def _get_dct_twiddle(N: int, device, dtype):
    """exp(-j pi k / (2N)) for k = 0 ... N-1, cached."""
    key = ("tw", N, str(device), dtype)
    if key not in _dct_cache:
        k = torch.arange(N, device=device, dtype=dtype)
        _dct_cache[key] = torch.polar(
            torch.ones(N, device=device, dtype=dtype),
            -math.pi * k / (2 * N),
        )
    return _dct_cache[key]


def _get_idct_twiddle(N: int, device, dtype):
    """exp(+j pi k / (2N)) for k = 0 ... N-1, cached."""
    key = ("tw_inv", N, str(device), dtype)
    if key not in _dct_cache:
        k = torch.arange(N, device=device, dtype=dtype)
        _dct_cache[key] = torch.polar(
            torch.ones(N, device=device, dtype=dtype),
            math.pi * k / (2 * N),
        )
    return _dct_cache[key]


def _get_idct_weights(N: int, norm, device, dtype):
    """Weight vector c[k] for the DCT-III sum, cached."""
    key = ("w", N, norm, str(device), dtype)
    if key not in _dct_cache:
        if norm is None or norm == "backward":
            w = torch.full((N,), 2.0 / N, device=device, dtype=dtype)
            w[0] = 1.0 / N
        elif norm == "ortho":
            w = torch.full((N,), math.sqrt(2.0 / N), device=device, dtype=dtype)
            w[0] = math.sqrt(1.0 / N)
        elif norm == "forward":
            w = torch.ones(N, device=device, dtype=dtype)
            w[0] = 0.5
        else:
            raise ValueError(f"Unsupported norm: {norm}")
        _dct_cache[key] = w
    return _dct_cache[key]


def _real_dtype_for_fft(dtype: torch.dtype) -> torch.dtype:
    """torch.fft only supports float32/64; promote low-precision reals for AMP."""
    if dtype == torch.float64:
        return torch.float64
    return torch.float32


def dct_1d(x: torch.Tensor, norm=None) -> torch.Tensor:
    """1D DCT-II along the last dimension (FFT-based, O(N log N))."""
    N = x.size(-1)
    if N <= 1:
        if norm == "forward":
            return x * (2.0 / max(N, 1))
        if norm == "ortho":
            return x * math.sqrt(1.0 / max(N, 1))
        return x.clone()

    work = _real_dtype_for_fft(x.dtype)
    v = torch.cat((x[..., ::2], x[..., 1::2].flip(-1)), dim=-1).to(dtype=work)
    Vc = torch.fft.fft(v, dim=-1)
    tw = _get_dct_twiddle(N, x.device, work)
    X = (Vc * tw).real

    if norm is None or norm == "backward":
        out = X
    elif norm == "forward":
        out = X * (2.0 / N)
    elif norm == "ortho":
        out = X.clone()
        out[..., 0] *= math.sqrt(1.0 / N)
        out[..., 1:] *= math.sqrt(2.0 / N)
    else:
        raise ValueError(f"Unsupported norm: {norm}")
    return out.to(x.dtype)


def idct_1d(X: torch.Tensor, norm=None) -> torch.Tensor:
    """1D inverse DCT (DCT-III) along the last dimension (FFT-based, O(N log N))."""
    N = X.size(-1)
    if N <= 1:
        if norm == "forward":
            return X * 0.5
        if norm is None or norm == "backward":
            return X.clone()
        if norm == "ortho":
            return X * math.sqrt(max(N, 1))
        return X.clone()

    work = _real_dtype_for_fft(X.dtype)
    w = _get_idct_weights(N, norm, X.device, work)
    tw_inv = _get_idct_twiddle(N, X.device, work)

    cdtype = torch.complex128 if work == torch.float64 else torch.complex64
    Xw = X.to(work) * w
    V = Xw.to(cdtype) * tw_inv.to(cdtype)
    v = torch.fft.ifft(V, dim=-1, norm="forward").real

    x = torch.empty(v.shape, device=v.device, dtype=work)
    half = (N + 1) // 2
    x[..., ::2] = v[..., :half]
    x[..., 1::2] = v[..., half:].flip(-1)
    return x.to(X.dtype)


def dct_2d(x: torch.Tensor, norm=None) -> torch.Tensor:
    """2D DCT-II over the last two dimensions."""
    X = dct_1d(x, norm=norm)
    X = dct_1d(X.transpose(-1, -2), norm=norm).transpose(-1, -2)
    return X


def idct_2d(X: torch.Tensor, norm=None) -> torch.Tensor:
    """2D inverse DCT (DCT-III pair) over the last two dimensions."""
    x = idct_1d(X, norm=norm)
    x = idct_1d(x.transpose(-1, -2), norm=norm).transpose(-1, -2)
    return x


# --------------------------------------------------------------------- #
# k-space spectrum utilities                                             #
# --------------------------------------------------------------------- #
def make_k_grid_dct(
    H: int, W: int,
    device=None, dtype=torch.float32, use_radians: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """DCT-II modal frequencies (Neumann boundary conditions).

    With ``use_radians=True`` (paper convention), ``k = (pi * u, pi * v)``
    so mode magnitudes range in ``[0, sqrt(2) * pi * (H - 1)]``.

    Returns ``(k, k2)`` each of shape ``(H, W)``.
    """
    m = torch.arange(H, device=device, dtype=dtype)
    n = torch.arange(W, device=device, dtype=dtype)
    fac = math.pi if use_radians else 1.0
    ky = m * fac
    kx = n * fac
    k2 = ky[:, None] ** 2 + kx[None, :] ** 2
    return torch.sqrt(k2), k2


def powerlaw_psd(
    k2: torch.Tensor, alpha: float, C: float = 1.0,
    k02: float = 0.0, eps: float = 1e-12,
) -> torch.Tensor:
    """Power-law variance spectrum ``S0(k) = C * (k^2 + k0^2)^(-alpha)``.

    The paper fits this form to per-mode empirical variances for CIFAR-10,
    ImageNet-128, ImageNet-256, and the critical Ising dataset; recovered
    parameters are listed in Table 2 (Appendix B.1).
    """
    denom = k2 + (k02 if k02 > 0.0 else eps)
    return C * torch.pow(denom, -alpha)


# --------------------------------------------------------------------- #
# Radial binning (DCT domain)                                            #
# --------------------------------------------------------------------- #
def radial_freqs(
    H: int, W: int, norm: Optional[str] = "forward",
) -> Tuple[ArrayF, npt.NDArray[np.int64]]:
    """Return ``(freqs, inv_idx)`` for radial binning of square 2D DCT spectra.

    Assumes ``H == W``; DC sits at ``(0, 0)``. ``inv_idx`` is the flat
    bin index of each mode (suitable for ``np.bincount``); ``freqs`` is
    the per-bin frequency magnitude scaled to match ``dct_2d``'s ``norm``.
    Used together with :func:`radial_profile` to reduce a per-mode 2D PSD
    to a 1D radial profile for the power-law fit
    (paper Appendix B.1, Figure 2).
    """
    if H != W:
        raise ValueError("radial_freqs assumes square images (H == W).")
    ky, kx = np.indices((H, W))
    r2 = (kx ** 2 + ky ** 2).ravel()
    radii2, inv_idx = np.unique(r2, return_inverse=True)
    if norm == "forward":
        freqs = np.sqrt(radii2) * np.pi
    elif norm == "backward":
        freqs = np.sqrt(radii2) * np.pi / (H * H)
    elif norm == "ortho":
        freqs = np.sqrt(radii2) * np.pi / H
    else:
        raise ValueError(f"Unsupported norm: {norm}")
    return freqs.astype(np.float64), inv_idx.astype(np.int64)


def radial_profile(
    img2d: npt.NDArray[np.floating],
    inv_idx: npt.NDArray[np.int64],
    avg: bool = True,
) -> ArrayF:
    """Bin a 2D image into radial shells using ``inv_idx`` from :func:`radial_freqs`.

    With ``avg=True`` (default) returns the shell mean; with ``avg=False``
    returns the shell sum.
    """
    vals = img2d.ravel()
    sum_vals = np.bincount(inv_idx, weights=vals)
    if not avg:
        return sum_vals.astype(np.float64)
    count_vals = np.bincount(inv_idx)
    return (sum_vals / np.maximum(count_vals, 1)).astype(np.float64)


def safe_div(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-20) -> torch.Tensor:
    return a / (b + eps)
