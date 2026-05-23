"""
Sampling entry point for SKILD.

Two modes are supported:

  * ``full`` (default) -- unconditional generation from pure noise. Used
    for the CIFAR-10 numbers in Table 1.
  * ``viz`` -- save intermediate states of the reverse process to
    visualize how an image is built up scale by scale.

Paper-faithful super-resolution evaluation lives in ``sr_eval.py``: it
initializes the reverse chain from the exact forward marginal of the
held-out HR image (paired across baselines via a cached noise
realization) and reports per-image quality metrics.

Examples::

    # Unconditional CIFAR-10
    python sample.py --config configs.specifics.cifar10_linear \\
        --ckpt /path/to/ckpt.pt --output_dir samples/cifar10

    # Visualize the reverse process
    python sample.py --config configs.specifics.cifar10_linear \\
        --ckpt /path/to/ckpt.pt --mode viz --output_dir samples/viz_cifar10
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os

import numpy as np
import torch
from torch import amp as torch_amp
from torchvision.utils import make_grid, save_image
from PIL import Image
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

import models.mutils as mutils
from diffusion import SKILD
from utils import make_k_grid_dct


# --------------------------------------------------------------------- #
#  AMP helpers                                                          #
# --------------------------------------------------------------------- #
def resolve_sampling_amp(config, device, no_amp: bool = False):
    """Mirror train.py AMP policy at inference (no GradScaler needed)."""
    if no_amp or str(device) == "cpu" or not getattr(config.training, "amp", True):
        return False, torch.float32
    ad = getattr(config.training, "amp_dtype", "auto")
    if ad == "auto":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    elif ad in ("bfloat16", "bf16"):
        dtype = torch.bfloat16
    else:
        dtype = torch.float16
    return True, dtype


@contextlib.contextmanager
def sampling_autocast(device, use_amp: bool, amp_dtype: torch.dtype):
    if not use_amp or str(device) == "cpu":
        yield
        return
    with torch_amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
        yield


def load_config_from_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return checkpoint.get("config")


# --------------------------------------------------------------------- #
#  Sampling routines                                                     #
# --------------------------------------------------------------------- #
def sample_full_generation(
    config, model, ddpm, output_dir: str,
    num_samples: int = 64,
    batch_size: int = 16,
    prediction_type: str = "eps",
    seed: int = 42,
    start_idx: int = 0,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
):
    """Unconditional generation from pure noise; resumes from existing samples on disk."""
    torch.manual_seed(seed)
    dev = next(model.parameters()).device

    os.makedirs(output_dir, exist_ok=True)
    samples_dir = os.path.join(output_dir, "samples")
    grids_dir = os.path.join(output_dir, "grids")
    os.makedirs(samples_dir, exist_ok=True)
    os.makedirs(grids_dir, exist_ok=True)

    H = W = config.data.image_size
    C = config.data.num_channels

    sample_idx = start_idx
    if start_idx == 0:
        existing = [f for f in os.listdir(samples_dir) if f.endswith(".png")]
        sample_idx = len(existing)
        if sample_idx > 0:
            print(f"Found {sample_idx} existing samples, resuming from index {sample_idx}")
    if sample_idx >= num_samples:
        print(f"Already have {sample_idx} >= {num_samples} samples. Nothing to do.")
        return

    num_batches = (num_samples + batch_size - 1) // batch_size
    batches_to_skip = sample_idx // batch_size

    print(f"Generating {num_samples} samples at {H}x{W} (starting from {sample_idx})...")
    pbar = tqdm(total=num_samples - sample_idx, desc="Sampling")
    for batch_idx in range(num_batches):
        if sample_idx >= num_samples:
            break
        if batch_idx < batches_to_skip:
            continue
        current_batch = min(batch_size, num_samples - sample_idx)
        shape = (current_batch, C, H, W)
        torch.manual_seed(seed + batch_idx)

        with torch.inference_mode():
            with sampling_autocast(dev, use_amp, amp_dtype):
                samples = ddpm.sample(
                    config, model, shape,
                    prediction_type=prediction_type, log_path=False,
                )

        for i in range(samples.shape[0]):
            if sample_idx >= num_samples:
                break
            sample = torch.clamp((samples[i].cpu() + 1.0) / 2.0, 0.0, 1.0)
            sample_np = (sample * 255).numpy().astype(np.uint8).transpose(1, 2, 0)
            Image.fromarray(sample_np).save(os.path.join(samples_dir, f"{sample_idx:05d}.png"))
            sample_idx += 1
            pbar.update(1)

        if batch_idx % 5 == 0 or batch_idx == num_batches - 1:
            nrow = min(10, int(np.ceil(np.sqrt(current_batch))))
            grid = make_grid(samples, nrow=nrow, normalize=True, value_range=(-1, 1))
            save_image(grid, os.path.join(grids_dir, f"grid_{batch_idx:04d}.png"))

    pbar.close()
    print(f"Generated {sample_idx} samples; saved to {samples_dir}")


def sample_with_visualization(
    config, model, ddpm, output_dir: str,
    num_samples: int = 4,
    prediction_type: str = "eps",
    save_intermediates: bool = True,
    use_amp: bool = False,
    amp_dtype: torch.dtype = torch.float32,
):
    """Generate samples while logging intermediate reverse-process states."""
    dev = next(model.parameters()).device
    os.makedirs(output_dir, exist_ok=True)

    H = W = config.data.image_size
    C = config.data.num_channels
    shape = (num_samples, C, H, W)

    print(f"Generating {num_samples} samples with intermediate visualization...")
    with torch.inference_mode():
        with sampling_autocast(dev, use_amp, amp_dtype):
            path = ddpm.sample(
                config, model, shape,
                prediction_type=prediction_type, log_path=True,
            )

    final_samples = path[-1]
    grid = make_grid(final_samples, nrow=2, normalize=True, value_range=(-1, 1))
    save_image(grid, os.path.join(output_dir, "final_samples.png"))

    if save_intermediates:
        T = path.shape[0]
        idxs = sorted({min(i, T - 1) for i in [0, T // 4, T // 2, 3 * T // 4, T - 1]})
        for idx in idxs:
            grid = make_grid(path[idx], nrow=2, normalize=True, value_range=(-1, 1))
            save_image(grid, os.path.join(output_dir, f"path_{idx:04d}.png"))

    print(f"Results saved to: {output_dir}")


# --------------------------------------------------------------------- #
#  CLI                                                                  #
# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(description="SKILD sampling")
    parser.add_argument("--config", type=str,
                        help="Config module path (optional if checkpoint contains config)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="results/samples",
                        help="Output directory")
    parser.add_argument("--num_samples", type=int, default=50000,
                        help="Number of samples to generate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for sampling")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="full", choices=["full", "viz"],
                        help="full | viz")
    parser.add_argument("--start_idx", type=int, default=0,
                        help="Starting sample index (for resuming interrupted generation)")
    parser.add_argument("--no_amp", action="store_true",
                        help="Disable AMP autocast (overrides config.training.amp)")
    parser.add_argument("--abar_cutoff", type=float, default=1e-6,
                        help="Override config.model.abar_cutoff")
    args = parser.parse_args()

    if args.config:
        config = importlib.import_module(args.config).get_config()
    else:
        config = load_config_from_checkpoint(args.ckpt)
        if config is None:
            raise ValueError("Config not found in checkpoint; pass --config.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    use_amp, amp_dtype = resolve_sampling_amp(config, device, no_amp=args.no_amp)
    print(f"Sampling AMP: {amp_dtype if use_amp else 'disabled (fp32)'}")

    image_size = config.data.image_size
    _, k2 = make_k_grid_dct(image_size, image_size, use_radians=True)
    k2 = k2.expand(1, image_size, image_size).to(device)
    S_0 = config.model.S_0
    if isinstance(S_0, torch.Tensor):
        S_0 = S_0.to(device)

    prediction_type = getattr(config.model, "prediction_type", "eps")
    kc = getattr(config.model, "kc", 0.0)
    abar_cutoff = getattr(config.model, "abar_cutoff", 1e-6)
    if args.abar_cutoff != abar_cutoff:
        print(f"Overriding abar_cutoff {abar_cutoff} -> {args.abar_cutoff}")
        abar_cutoff = args.abar_cutoff
    lambda_sched = getattr(config.model, "lambda_schedule", "log_linear")
    theta_sched = getattr(config.model, "theta", 3.0)

    ddpm = SKILD(
        N=config.model.num_scales,
        k2=k2, S_0=S_0,
        lambda_i=config.model.lambda_i,
        lambda_f=config.model.lambda_f,
        kc=kc, abar_cutoff=abar_cutoff,
        device=device, schedule=lambda_sched, theta=theta_sched,
    )

    lam_embed = ddpm.lam
    config.model.skild_sigmas = torch.sqrt(lam_embed).cpu().numpy().tolist()
    config.model.skild_lambda = lam_embed.cpu().numpy().tolist()

    print(f"Loading model from {args.ckpt}...")
    model = mutils.load_model(config, args.ckpt, device)
    model.eval()

    use_compile = getattr(config.training, "compile", False)
    compile_mode = getattr(config.training, "compile_mode", "default")
    if use_compile:
        print(f"Compiling model with torch.compile(mode='{compile_mode}')...")
        model = torch.compile(model, mode=compile_mode)

    print(f"Image size: {image_size}")
    print(f"kc={ddpm.kc:.4f}, kc^2={ddpm.kc2:.4f}")
    print(f"abar_cutoff={ddpm.abar_cutoff:.4e}")
    print(f"lambda range: [{ddpm.lam[0]:.6e}, {ddpm.lam[-1]:.6e}]")
    print(f"Prediction type: {prediction_type}")

    if args.mode == "full":
        sample_full_generation(
            config, model, ddpm, args.output_dir,
            num_samples=args.num_samples, batch_size=args.batch_size,
            prediction_type=prediction_type, seed=args.seed,
            start_idx=args.start_idx, use_amp=use_amp, amp_dtype=amp_dtype,
        )
    elif args.mode == "viz":
        sample_with_visualization(
            config, model, ddpm, args.output_dir,
            num_samples=min(4, args.num_samples),
            prediction_type=prediction_type,
            save_intermediates=True,
            use_amp=use_amp, amp_dtype=amp_dtype,
        )


if __name__ == "__main__":
    main()
