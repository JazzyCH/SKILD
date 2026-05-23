"""
Paired super-resolution evaluation against ground-truth forward marginals.

This script reproduces the ImageNet super-resolution evaluation protocol
of the paper (Section 5.2, Appendix C): for each held-out HR image we use
its **exact forward marginal** at ``init_timestep`` as the reverse-process
initialization, run the reverse chain end-to-end, and score the resulting
HR reconstruction with per-image quality metrics.

Cached forward marginals (``--gt_forward_k_path``) are required so that
the same noise realization is shared across baselines for paired
comparison. The matching HR folder (``--gt_hr_dir``) supplies the ground
truth used by the full-reference metrics.

Reported per-image metrics:
  * MSE / PSNR / SSIM / MS-SSIM / LPIPS (full reference)
  * MUSIQ / CLIPIQA (no reference, can be disabled with ``--no_iqa``)

Multi-GPU (one rank per GPU)::

    torchrun --nproc_per_node=8 sr_eval.py \\
        --config configs.specifics.imagenet256_4x \\
        --ckpt /path/to/ckpt.pt \\
        --gt_forward_k_path /path/to/forward_cache.pt \\
        --gt_hr_dir /path/to/hr_images \\
        --output_dir results/imagenet256_4x_eval \\
        --num_samples 3000 --batch_size 32
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.distributed as dist
from PIL import Image
from torch import amp as torch_amp
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

import models.mutils as mutils
from datasets import center_crop_arr
from diffusion import SKILD
from utils import make_k_grid_dct


# --------------------------------------------------------------------- #
#  Distributed helpers                                                   #
# --------------------------------------------------------------------- #
def setup_distributed() -> Tuple[int, int, int]:
    """Initialize distributed env; returns ``(rank, local_rank, world_size)``."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        return dist.get_rank(), int(os.environ.get("LOCAL_RANK", 0)), dist.get_world_size()
    return 0, 0, 1


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process() -> bool:
    return not dist.is_initialized() or dist.get_rank() == 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


# --------------------------------------------------------------------- #
#  AMP                                                                   #
# --------------------------------------------------------------------- #
def resolve_sampling_amp(config, device, no_amp: bool = False):
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


def load_config_from_checkpoint(ckpt_path: str):
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return checkpoint.get("config")


# --------------------------------------------------------------------- #
#  Cache loading + GT dataset                                            #
# --------------------------------------------------------------------- #
def load_gt_pack(path: str, init_timestep_cfg: int, num_scales: int):
    """Load a ``.pt`` cache produced by ``build_gt_forward_cache`` (or equivalent).

    Expected keys: ``x_init_k`` (B, C, H, W), ``init_timestep``, ``num_scales``.
    """
    pack = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(pack, dict) or "x_init_k" not in pack:
        raise ValueError(f"Invalid cache format in {path}: expected dict with x_init_k")
    if int(pack.get("init_timestep", -1)) != int(init_timestep_cfg):
        raise ValueError(
            f"gt pack init_timestep={pack.get('init_timestep')} != expected {init_timestep_cfg}"
        )
    if int(pack.get("num_scales", num_scales)) != int(num_scales):
        raise ValueError("Cached num_scales does not match model num_scales; rebuild cache.")
    return pack


def build_gt_dataset(gt_hr_dir: str, image_size: int):
    transform = transforms.Compose([
        transforms.Lambda(lambda p: center_crop_arr(p, image_size)),
        transforms.ToTensor(),
    ])
    return ImageFolder(gt_hr_dir, transform=transform)


def denorm_to_uint8(x01: torch.Tensor) -> np.ndarray:
    x = torch.clamp(x01, 0.0, 1.0)
    return (x * 255.0).round().to(torch.uint8).cpu().numpy().transpose(1, 2, 0)


# --------------------------------------------------------------------- #
#  Metric models                                                         #
# --------------------------------------------------------------------- #
def get_metric_models(device: torch.device, image_size: int):
    try:
        import lpips
    except Exception as e:
        raise RuntimeError("Missing dependency 'lpips'. pip install lpips") from e
    try:
        from pytorch_msssim import ms_ssim, ssim
    except Exception as e:
        raise RuntimeError("Missing dependency 'pytorch-msssim'. pip install pytorch-msssim") from e

    lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    # pytorch-msssim defaults need image side > 160 with win_size=11; for 128x128
    # we reduce win_size to 7 so MS-SSIM remains available.
    ms_ssim_win_size = 11 if int(image_size) > 160 else 7
    return ssim, ms_ssim, lpips_model, ms_ssim_win_size


def get_iqa_models(device: torch.device):
    try:
        import pyiqa
    except Exception as e:
        raise RuntimeError("Missing dependency 'pyiqa'. pip install pyiqa") from e
    musiq = pyiqa.create_metric("musiq", device=device, as_loss=False).eval()
    clipiqa = pyiqa.create_metric("clipiqa", device=device, as_loss=False).eval()
    return musiq, clipiqa


def compute_metrics_batch(
    pred01: torch.Tensor, gt01: torch.Tensor,
    ssim_fn, ms_ssim_fn, lpips_model,
    ms_ssim_win_size: int,
    musiq_model=None, clipiqa_model=None,
) -> Dict[str, torch.Tensor]:
    """Full-reference (MSE, PSNR, SSIM, MS-SSIM, LPIPS) plus optional MUSIQ/CLIPIQA."""
    pred01 = pred01.float()
    gt01 = gt01.float()

    mse = ((pred01 - gt01) ** 2).mean(dim=(1, 2, 3))
    psnr = -10.0 * torch.log10(torch.clamp(mse, min=1e-12))
    ssim_v = ssim_fn(pred01, gt01, data_range=1.0, size_average=False)
    try:
        ms_ssim_v = ms_ssim_fn(
            pred01, gt01, data_range=1.0,
            size_average=False, win_size=int(ms_ssim_win_size),
        )
    except AssertionError:
        ms_ssim_v = torch.full_like(ssim_v, float("nan"))

    pred11 = pred01 * 2.0 - 1.0
    gt11 = gt01 * 2.0 - 1.0
    with torch.inference_mode():
        lpips_v = lpips_model(pred11, gt11).view(-1)

    result = {"mse": mse, "psnr": psnr, "ssim": ssim_v, "ms_ssim": ms_ssim_v, "lpips": lpips_v}

    if musiq_model is not None and clipiqa_model is not None:
        with torch.inference_mode():
            result["musiq"] = musiq_model(pred01).view(-1)
            result["clipiqa"] = clipiqa_model(pred01).view(-1)
    return result


# --------------------------------------------------------------------- #
#  Aggregation                                                           #
# --------------------------------------------------------------------- #
def summarize_metrics(csv_path: Path, summary_json_path: Path) -> Dict[str, float]:
    metric_keys = ["mse", "psnr", "ssim", "ms_ssim", "lpips", "musiq", "clipiqa"]
    values: Dict[str, List[float]] = {k: [] for k in metric_keys}
    with csv_path.open("r", newline="") as f:
        for row in csv.DictReader(f):
            for k in metric_keys:
                if not row.get(k):
                    continue
                try:
                    v = float(row[k])
                    if math.isfinite(v):
                        values[k].append(v)
                except ValueError:
                    pass

    summary: Dict[str, float] = {}
    for k, v in values.items():
        arr = np.asarray(v, dtype=np.float64)
        if arr.size == 0:
            continue
        summary[f"{k}_mean"] = float(arr.mean())
        summary[f"{k}_std"] = float(arr.std())
        summary[f"{k}_min"] = float(arr.min())
        summary[f"{k}_max"] = float(arr.max())
        summary[f"{k}_p10"] = float(np.quantile(arr, 0.10))
        summary[f"{k}_p50"] = float(np.quantile(arr, 0.50))
        summary[f"{k}_p90"] = float(np.quantile(arr, 0.90))

    summary_json_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_json_path.open("w") as f:
        json.dump(summary, f, indent=2)
    return summary


def merge_rank_csvs(output_dir: Path, world_size: int, fieldnames: List[str]) -> Path:
    metrics_dir = output_dir / "metrics"
    final_csv = metrics_dir / "per_image_metrics.csv"
    all_rows = []
    for r in range(world_size):
        rank_csv = metrics_dir / f"per_image_metrics_rank{r}.csv"
        if rank_csv.exists():
            with rank_csv.open("r", newline="") as f:
                for row in csv.DictReader(f):
                    all_rows.append(row)
    all_rows.sort(key=lambda x: int(x.get("output_index", 0)))
    with final_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)
    for r in range(world_size):
        rank_csv = metrics_dir / f"per_image_metrics_rank{r}.csv"
        if rank_csv.exists():
            rank_csv.unlink()
    return final_csv


# --------------------------------------------------------------------- #
#  Main                                                                  #
# --------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Paired SR sampling from forward-k cache + per-image quality metrics"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Config module path (e.g. configs.specifics.imagenet256_4x)")
    parser.add_argument("--ckpt", type=str, required=True, help="Checkpoint path")
    parser.add_argument("--gt_forward_k_path", type=str, default=None,
                        help="Path to cache .pt with x_init_k "
                             "(defaults to config.training.gt_forward_k_path)")
    parser.add_argument("--gt_init_timestep", type=int, default=None,
                        help="Expected cache init_timestep (default: config.training.gt_init_timestep)")
    parser.add_argument("--gt_hr_dir", type=str, default=None,
                        help="GT HR image directory (default: config.data.eval_path)")
    parser.add_argument("--output_dir", type=str, default="results/sr_eval")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shuffle_indices", action="store_true",
                        help="Deterministic shuffled order over cache rows")
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--abar_cutoff", type=float, default=1e-6,
                        help="Override config.model.abar_cutoff")
    parser.add_argument("--no_iqa", action="store_true",
                        help="Skip MUSIQ/CLIPIQA (full-ref metrics only)")
    parser.add_argument("--save_samples", action="store_true", default=True,
                        help="Save generated PNG samples (default: True)")
    parser.add_argument("--no_save_samples", action="store_true",
                        help="Disable saving PNG samples (metrics-only mode)")
    args = parser.parse_args()
    if args.no_save_samples:
        args.save_samples = False

    rank, local_rank, world_size = setup_distributed()
    is_main = is_main_process()
    try:
        _run(args, rank, local_rank, world_size, is_main)
    finally:
        cleanup_distributed()


def _run(args, rank: int, local_rank: int, world_size: int, is_main: bool):
    if args.config:
        config = importlib.import_module(args.config).get_config()
    else:
        config = load_config_from_checkpoint(args.ckpt)
        if config is None:
            raise ValueError("Could not load config; pass --config")

    gt_path = args.gt_forward_k_path or getattr(config.training, "gt_forward_k_path", None)
    gt_n = (
        args.gt_init_timestep if args.gt_init_timestep is not None
        else getattr(config.training, "gt_init_timestep", None)
    )
    gt_hr_dir = args.gt_hr_dir or getattr(config.data, "eval_path", None)

    if gt_path is None or gt_n is None:
        raise ValueError(
            "Need --gt_forward_k_path and --gt_init_timestep "
            "(or set them in config.training)"
        )
    if gt_hr_dir is None:
        raise ValueError("Need --gt_hr_dir or config.data.eval_path")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(gt_path)
    if not os.path.isdir(gt_hr_dir):
        raise FileNotFoundError(gt_hr_dir)

    if torch.cuda.is_available():
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")
    torch.manual_seed(args.seed + rank)

    use_amp, amp_dtype = resolve_sampling_amp(config, device, no_amp=args.no_amp)
    if is_main:
        print(f"[Rank {rank}/{world_size}] Sampling AMP: "
              f"{amp_dtype if use_amp else 'disabled (fp32)'}")
        print(f"[Rank {rank}/{world_size}] Device: {device}")

    image_size = int(config.data.image_size)
    _, k2 = make_k_grid_dct(image_size, image_size, use_radians=True)
    k2 = k2.expand(1, image_size, image_size).to(device)
    S_0 = config.model.S_0
    if isinstance(S_0, torch.Tensor):
        S_0 = S_0.to(device)

    prediction_type = getattr(config.model, "prediction_type", "eps")
    kc = getattr(config.model, "kc", 0.0)
    abar_cutoff = getattr(config.model, "abar_cutoff", 1e-6)
    if args.abar_cutoff != abar_cutoff:
        if is_main:
            print(f"Overriding abar_cutoff {abar_cutoff} -> {args.abar_cutoff}")
        abar_cutoff = args.abar_cutoff
    lambda_sched = getattr(config.model, "lambda_schedule", "log_linear")
    theta_sched = getattr(config.model, "theta", 3.0)

    ddpm = SKILD(
        N=config.model.num_scales,
        k2=k2, S_0=S_0,
        lambda_i=config.model.lambda_i, lambda_f=config.model.lambda_f,
        kc=kc, abar_cutoff=abar_cutoff,
        device=device, schedule=lambda_sched, theta=theta_sched,
    )
    lam_embed = ddpm.lam
    config.model.skild_sigmas = torch.sqrt(lam_embed).cpu().numpy().tolist()
    config.model.skild_lambda = lam_embed.cpu().numpy().tolist()

    gt_pack = load_gt_pack(gt_path, int(gt_n), ddpm.N)
    x_stored = gt_pack["x_init_k"]
    gt_ds = build_gt_dataset(gt_hr_dir, image_size)

    if len(gt_ds) != int(x_stored.shape[0]):
        if is_main:
            print(
                f"Warning: GT dataset ({len(gt_ds)}) and cache ({x_stored.shape[0]}) differ; "
                "using min length for paired evaluation."
            )
    n_total = min(len(gt_ds), int(x_stored.shape[0]))
    if args.num_samples <= 0:
        raise ValueError("--num_samples must be > 0")

    output_dir = Path(args.output_dir)
    samples_dir = output_dir / "samples"
    metrics_dir = output_dir / "metrics"
    final_csv_path = metrics_dir / "per_image_metrics.csv"
    summary_path = metrics_dir / "summary.json"
    rank_csv_path = (
        metrics_dir / f"per_image_metrics_rank{rank}.csv" if world_size > 1 else final_csv_path
    )

    if is_main:
        samples_dir.mkdir(parents=True, exist_ok=True)
        metrics_dir.mkdir(parents=True, exist_ok=True)
    barrier()

    n_target = min(args.num_samples, n_total)
    per_rank = (n_target + world_size - 1) // world_size
    rank_start = rank * per_rank
    rank_end = min(rank_start + per_rank, n_target)
    if rank_start >= n_target:
        if is_main:
            print(f"Rank {rank}: no work (start={rank_start} >= target={n_target})")
        barrier()
        return

    order = torch.arange(n_total, dtype=torch.long)
    if args.shuffle_indices:
        g = torch.Generator().manual_seed(int(args.seed))
        order = torch.randperm(n_total, generator=g)

    if is_main:
        print(f"Loading model from {args.ckpt}...")
    model = mutils.load_model(config, args.ckpt, device)
    model.eval()

    use_compile = getattr(config.training, "compile", False)
    compile_mode = getattr(config.training, "compile_mode", "default")
    if use_compile:
        if is_main:
            print(f"Compiling model torch.compile(mode='{compile_mode}')...")
        model = torch.compile(model, mode=compile_mode)

    ssim_fn, ms_ssim_fn, lpips_model, ms_ssim_win_size = get_metric_models(
        device, image_size=image_size,
    )
    if is_main:
        print(f"MS-SSIM win_size={ms_ssim_win_size}")

    musiq_model, clipiqa_model = None, None
    if not args.no_iqa:
        if is_main:
            print("Loading IQA models (MUSIQ, CLIPIQA)...")
        musiq_model, clipiqa_model = get_iqa_models(device)

    fieldnames = [
        "output_index", "cache_index", "gt_relpath", "sample_path",
        "mse", "psnr", "ssim", "ms_ssim", "lpips",
    ]
    if not args.no_iqa:
        fieldnames += ["musiq", "clipiqa"]

    barrier()
    if is_main:
        print(f"Starting evaluation: {n_target} samples across {world_size} GPU(s)")
        print(f"  Rank {rank}: indices [{rank_start}, {rank_end}) = {rank_end - rank_start} samples")

    with rank_csv_path.open("w", newline="") as fcsv:
        writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
        writer.writeheader()
        pbar = tqdm(
            total=rank_end - rank_start, desc=f"[Rank {rank}] SR sample+eval",
            disable=(not is_main and world_size > 1), position=rank,
        )
        out_idx = rank_start
        while out_idx < rank_end:
            bsz = min(args.batch_size, rank_end - out_idx)
            out_ids = torch.arange(out_idx, out_idx + bsz, dtype=torch.long)
            cache_idx = order[out_ids]
            x_init_k = x_stored[cache_idx].to(device=device, dtype=ddpm.S_0_sqrt.dtype)

            gt_batch, gt_relpaths = [], []
            for idx in cache_idx.tolist():
                x_gt, _ = gt_ds[idx]
                gt_batch.append(x_gt)
                gt_relpaths.append(os.path.relpath(gt_ds.samples[idx][0], gt_hr_dir))
            gt01 = torch.stack(gt_batch, dim=0).to(device=device, dtype=torch.float32)

            with torch.inference_mode():
                with sampling_autocast(device, use_amp, amp_dtype):
                    samples = ddpm.sample(
                        config, model, shape=tuple(gt01.shape),
                        prediction_type=prediction_type, log_path=False,
                        x_init_k=x_init_k, init_timestep=int(gt_n),
                    )
            pred01 = torch.clamp((samples.float() + 1.0) / 2.0, 0.0, 1.0)
            m = compute_metrics_batch(
                pred01, gt01,
                ssim_fn, ms_ssim_fn, lpips_model,
                ms_ssim_win_size=ms_ssim_win_size,
                musiq_model=musiq_model, clipiqa_model=clipiqa_model,
            )

            for i in range(bsz):
                this_idx = out_idx + i
                sample_name = f"{this_idx:05d}.png"
                sample_abs = samples_dir / sample_name
                if args.save_samples:
                    Image.fromarray(denorm_to_uint8(pred01[i])).save(sample_abs)
                row = {
                    "output_index": this_idx,
                    "cache_index": int(cache_idx[i].item()),
                    "gt_relpath": gt_relpaths[i],
                    "sample_path": f"samples/{sample_name}" if args.save_samples else "",
                    "mse": float(m["mse"][i].item()),
                    "psnr": float(m["psnr"][i].item()),
                    "ssim": float(m["ssim"][i].item()),
                    "ms_ssim": float(m["ms_ssim"][i].item()),
                    "lpips": float(m["lpips"][i].item()),
                }
                if not args.no_iqa:
                    row["musiq"] = float(m["musiq"][i].item())
                    row["clipiqa"] = float(m["clipiqa"][i].item())
                writer.writerow(row)
            fcsv.flush()
            out_idx += bsz
            pbar.update(bsz)
        pbar.close()

    barrier()
    if is_main:
        if world_size > 1:
            print("Merging per-rank CSV files...")
            final_csv_path = merge_rank_csvs(output_dir, world_size, fieldnames)
        else:
            final_csv_path = rank_csv_path
        summary = summarize_metrics(final_csv_path, summary_path)
        print("\nDone. Metric summary:")
        print(json.dumps(summary, indent=2))
        print(f"Samples: {samples_dir}")
        print(f"CSV: {final_csv_path}")
        print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
