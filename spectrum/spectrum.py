"""
Dataset power-spectrum measurement and offset-power-law fitting.

SKILD's noise schedule is parameterized in terms of the data power
spectrum ``S_0(k)``, which is fit to a single offset power law

    S(k) approx C * (k**2 + k0**2)**(-alpha)

per dataset (paper Sec. 3 and Appendix B.1). This module provides:

* ``compute_dct_stats_parallel`` — streaming first / second moment of
  ``|DCT(x)|`` across a dataloader.
* ``fit_powerlaw_offset`` — least-squares fit to the radially binned
  spectrum.
* ``plot_spectrum_fit`` — diagnostic plots used to produce Figure 2 of
  the paper.
"""

from __future__ import annotations

from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import torch
from scipy.optimize import curve_fit

from utils import dct_2d

ArrayF = npt.NDArray[np.float64]


def compute_dct_stats_parallel(
    loader,
    max_batches: Optional[int] = None,
    device: Optional[str | int | torch.device] = None,
    norm: Optional[str] = "forward",
    dtype: torch.dtype = torch.float32,
) -> Tuple[ArrayF, ArrayF, ArrayF]:
    """Streaming computation of dataset DCT statistics.

    Returns ``(mean_abs, var, mean_psd)`` of shape ``(C, H, W)`` each.
    With a unitary DCT, ``sum_k mean_psd[c, k]`` equals
    ``E[mean(x_c**2)]`` via Parseval, so ``mean_psd`` is the per-mode PSD.
    """

    torch.set_grad_enabled(False)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sum_abs = None
    sum_abs2 = None
    count = 0

    for bidx, batch in enumerate(loader):
        x = batch[0] if isinstance(batch, (list, tuple)) else batch
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = x.to(device=device, dtype=dtype, non_blocking=True)
        B = x.shape[0]

        F = dct_2d(x, norm=norm)
        M = F.abs()
        batch_sum = M.sum(dim=0)
        batch_sumsq = (M * M).sum(dim=0)

        if sum_abs is None:
            sum_abs = torch.zeros_like(batch_sum)
            sum_abs2 = torch.zeros_like(batch_sumsq)
        sum_abs += batch_sum
        sum_abs2 += batch_sumsq
        count += B

        if max_batches and (bidx + 1) >= max_batches:
            break

    if count == 0:
        raise ValueError("Empty loader or max_batches=0.")

    mean_abs = sum_abs / count
    mean_psd = sum_abs2 / count
    var = mean_psd - mean_abs * mean_abs
    return mean_abs.cpu().numpy(), var.cpu().numpy(), mean_psd.cpu().numpy()


def _powerlaw(k, k02, a, C):
    return C * (k ** 2 + k02) ** (-a)


def fit_powerlaw_offset(
    freqs: npt.NDArray[np.floating],
    spectrum: npt.NDArray[np.floating],
    head_trim: int | None = None,
    tail_trim: int | None = None,
) -> Tuple[ArrayF, ArrayF, Tuple[float, float, float]]:
    """Fit ``S(k) approx C * (k**2 + k0**2)**(-alpha)`` to a radial spectrum.

    Returns ``(k_full, S_fit_full, (k0, alpha, C))``. The returned curve is
    evaluated on the full frequency range even when ``head_trim`` / ``tail_trim``
    restrict the parameter estimation to an interior window.
    """

    f_full = np.asarray(freqs, dtype=np.float64)
    s_full = np.asarray(spectrum, dtype=np.float64)
    f_fit, s_fit = f_full, s_full
    if head_trim:
        f_fit = f_fit[head_trim:]
        s_fit = s_fit[head_trim:]
    if tail_trim:
        f_fit = f_fit[:-tail_trim]
        s_fit = s_fit[:-tail_trim]
    if f_fit.size < 5:
        raise ValueError("Not enough points after trimming to fit spectrum.")

    p0 = [2.0, 1.0, float(s_fit.min())]
    bounds = ([np.pi / 2, 0.0, 0.0], [3.0, 10.0, np.inf])
    params, _ = curve_fit(_powerlaw, f_fit, s_fit, p0=p0, bounds=bounds, maxfev=20000)
    k0_fit, a_fit, C_fit = map(float, params)
    print(f"Fit: S(k) ~ {C_fit:.4e} / (k**2 + {k0_fit:.6f})^{a_fit:.6f}")

    spec_fit = _powerlaw(f_full, k0_fit, a_fit, C_fit)
    return f_full, spec_fit, (k0_fit, a_fit, C_fit)


def plot_spectrum_fit(
    freqs,
    rad_spectra,
    sum_spectra,
    fit_freqs,
    fit_spectra,
    dset_name: str,
    labels=("R", "G", "B"),
    method: str = "PSD",
    save: bool = False,
    head_trim: int | None = None,
    tail_trim: int | None = None,
):
    """Diagnostic plot of measured spectrum overlaid with the fitted curve.

    Produces two panels (log-log and lin-x / log-y) per call. Set ``save=True``
    to dump PDFs to the current working directory.
    """

    colors = ["tab:red", "tab:green", "tab:blue"]
    markers = [0, 1, 3]

    freqs = np.asarray(freqs, dtype=np.float64)
    fit_freqs = np.asarray(fit_freqs, dtype=np.float64)
    fit_spectra = np.asarray(fit_spectra, dtype=np.float64)

    n_fit = fit_freqs.size
    lo = max(int(head_trim or 0), 0)
    hi = min(max(n_fit - int(tail_trim or 0), lo), n_fit)
    in_fit_mask = np.zeros(n_fit, dtype=bool)
    if hi > lo:
        in_fit_mask[lo:hi] = True
    out_fit_mask = ~in_fit_mask

    pos_mask = freqs > 0
    fit_pos_mask = fit_freqs > 0
    if np.any(pos_mask):
        plt.figure(figsize=(4, 4))
        if len(labels) > 1:
            for c, lab in enumerate(labels):
                plt.scatter(
                    freqs[pos_mask],
                    np.asarray(rad_spectra[c])[pos_mask],
                    label=str(lab),
                    color=colors[c],
                    marker=markers[c],
                )
        plt.plot(freqs[pos_mask], np.asarray(sum_spectra)[pos_mask], lw=2, label="All", color="tab:orange")
        fit_in_pos = fit_pos_mask & in_fit_mask
        fit_out_pos = fit_pos_mask & out_fit_mask
        if np.any(fit_in_pos):
            plt.plot(fit_freqs[fit_in_pos], fit_spectra[fit_in_pos], "-", lw=2.2, label="Fit (used)", color="aquamarine")
        if np.any(fit_out_pos):
            plt.plot(fit_freqs[fit_out_pos], fit_spectra[fit_out_pos], "--", lw=2, label="Fit (extrap.)", color="deepskyblue")
        plt.xscale("log"); plt.yscale("log")
        fmin_pos = freqs[pos_mask].min()
        fmax_pos = freqs[pos_mask].max()
        plt.xlim(fmin_pos * 0.6, fmax_pos * 1.2)
        plt.xlabel("Radial frequency")
        plt.ylabel(method)
        plt.title(f"{dset_name.upper()} - {method} (log-log)")
        plt.legend(); plt.grid()
        plt.gca().set_facecolor("whitesmoke")
        plt.tight_layout()
        if save:
            plt.savefig(f"{dset_name}_{method}_spectrum_fit_loglog.pdf", dpi=300)
        else:
            plt.show()

    plt.figure(figsize=(4, 4))
    if len(labels) > 1:
        for c, lab in enumerate(labels):
            plt.scatter(freqs, rad_spectra[c], label=str(lab), color=colors[c], marker=markers[c])
    plt.plot(freqs, sum_spectra, lw=2, label="All", color="tab:orange")
    if np.any(in_fit_mask):
        plt.plot(fit_freqs[in_fit_mask], fit_spectra[in_fit_mask], "-", lw=2.2, label="Fit (used)", color="aquamarine")
    if np.any(out_fit_mask):
        plt.plot(fit_freqs[out_fit_mask], fit_spectra[out_fit_mask], "--", lw=2, label="Fit (extrap.)", color="deepskyblue")
    plt.yscale("log")
    if freqs.size > 0:
        xmin = float(freqs.min()); xmax = float(freqs.max())
        span = max(xmax - xmin, 1.0)
        plt.xlim(xmin - 0.04 * span, xmax + 0.02 * span)
    plt.xlabel("Radial frequency")
    plt.ylabel(method)
    plt.title(f"{dset_name.upper()} - {method} (lin-x)")
    plt.legend(); plt.grid()
    plt.gca().set_facecolor("whitesmoke")
    plt.tight_layout()
    if save:
        plt.savefig(f"{dset_name}_{method}_spectrum_fit_linx.pdf", dpi=300)
    else:
        plt.show()
