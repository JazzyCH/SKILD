"""
FID and Inception Score evaluation for generated samples.

Built on top of ``torch-fidelity`` (https://github.com/toshas/torch-fidelity).
Used to compute the CIFAR-10 numbers in Table 1 of the paper. Generated
samples should be PNGs in ``--samples_dir``; the reference dataset can be
the torch-fidelity built-in ``cifar10-train`` (default), a folder of
images (``--ref_dir``), or extracted from a local torchvision install
(``--cifar10_path``).

Example::

    python evaluate.py --samples_dir results/cifar10/samples \\
        --cifar10_path /path/to/cifar10_root
"""

from __future__ import annotations
import argparse
import json
import os
import shutil
import tempfile

import torch
# Workaround: torch-fidelity caches FID stats containing numpy arrays, but
# PyTorch 2.6+ defaults weights_only=True and rejects numpy globals.
_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    if "weights_only" not in kwargs:
        kwargs["weights_only"] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

from torch_fidelity import calculate_metrics  # noqa: E402
from torchvision import transforms              # noqa: E402
from torchvision.datasets import CIFAR10        # noqa: E402
from PIL import Image                           # noqa: E402
from tqdm import tqdm                           # noqa: E402


class CIFAR10FolderFromTorchvision:
    """Extract CIFAR-10 training images to a temporary folder of PNGs."""

    def __init__(self, root: str, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        dataset = CIFAR10(root=root, train=True, download=True)
        print(f"Extracting {len(dataset)} CIFAR-10 training images to {output_dir}...")
        for idx in tqdm(range(len(dataset)), desc="Extracting CIFAR-10"):
            img, _ = dataset[idx]
            img.save(os.path.join(output_dir, f"{idx:05d}.png"))
        print(f"Extracted {len(dataset)} images.")


def resize_samples(src_dir: str, dst_dir: str, size: int):
    """Resize all PNG images in ``src_dir`` to ``(size, size)``, save to ``dst_dir``."""
    os.makedirs(dst_dir, exist_ok=True)
    resize = transforms.Resize(
        (size, size), interpolation=transforms.InterpolationMode.BILINEAR,
    )
    files = sorted(f for f in os.listdir(src_dir) if f.lower().endswith(".png"))
    print(f"Resizing {len(files)} images to {size}x{size}...")
    for fname in tqdm(files, desc="Resizing"):
        img = Image.open(os.path.join(src_dir, fname)).convert("RGB")
        resize(img).save(os.path.join(dst_dir, fname))


def main():
    parser = argparse.ArgumentParser(description="FID/IS evaluation via torch-fidelity")
    parser.add_argument("--samples_dir", type=str, required=True,
                        help="Directory of generated PNG samples")
    parser.add_argument("--ref_dir", type=str, default=None,
                        help="Folder of reference images "
                             "(if omitted, falls back to --cifar10_path or built-in cifar10-train)")
    parser.add_argument("--cifar10_path", type=str, default=None,
                        help="Path to torchvision CIFAR-10 root to extract reference from")
    parser.add_argument("--resize", type=int, default=None,
                        help="Resize generated samples to this size before evaluation (e.g. 32)")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for Inception feature extraction")
    parser.add_argument("--save_json", type=str, default=None,
                        help="Path to save metrics as JSON")
    parser.add_argument("--kid", action="store_true",
                        help="Also compute KID (Kernel Inception Distance)")
    parser.add_argument("--deep", action="store_true",
                        help="Recursively search samples_dir (needed for class-based folders)")
    parser.add_argument("--no_cache", action="store_true",
                        help="Disable torch-fidelity Inception-stats caching")
    parser.add_argument("--cache_root", type=str, default=None,
                        help="Custom root for torch-fidelity Inception-stats cache")
    args = parser.parse_args()

    sample_files = [f for f in os.listdir(args.samples_dir) if f.lower().endswith(".png")]
    print(f"Found {len(sample_files)} generated samples in {args.samples_dir}")

    # Optional resize of generated samples
    samples_input = args.samples_dir
    tmp_resized_dir = None
    if args.resize is not None:
        tmp_resized_dir = tempfile.mkdtemp(prefix="eval_resized_")
        resize_samples(args.samples_dir, tmp_resized_dir, args.resize)
        samples_input = tmp_resized_dir

    # Reference dataset
    tmp_cifar10_dir = None
    if args.ref_dir is not None:
        ref_input, ref_label = args.ref_dir, args.ref_dir
    elif args.cifar10_path is not None:
        tmp_cifar10_dir = tempfile.mkdtemp(prefix="eval_cifar10_")
        CIFAR10FolderFromTorchvision(args.cifar10_path, tmp_cifar10_dir)
        ref_input = tmp_cifar10_dir
        ref_label = f"CIFAR-10 train (from {args.cifar10_path})"
    else:
        ref_input, ref_label = "cifar10-train", "cifar10-train (built-in)"

    print(f"\n{'=' * 50}")
    print("Evaluation Settings:")
    print(f"  Generated samples : {args.samples_dir} ({len(sample_files)} images)")
    if args.resize:
        print(f"  Resized to        : {args.resize}x{args.resize}")
    print(f"  Reference dataset : {ref_label}")
    print(f"  Batch size        : {args.batch_size}")
    print(f"  Metrics           : FID, IS" + (", KID" if args.kid else ""))
    print(f"{'=' * 50}\n")

    calc_kwargs = dict(
        input1=samples_input, input2=ref_input,
        cuda=torch.cuda.is_available(),
        isc=True, fid=True, kid=args.kid,
        batch_size=args.batch_size, verbose=True,
    )
    if args.deep:
        calc_kwargs["samples_find_deep"] = True
    if args.no_cache:
        calc_kwargs["cache"] = False
    if args.cache_root is not None:
        calc_kwargs["cache_root"] = args.cache_root

    try:
        metrics = calculate_metrics(**calc_kwargs)
    finally:
        if tmp_resized_dir and os.path.exists(tmp_resized_dir):
            shutil.rmtree(tmp_resized_dir)
        if tmp_cifar10_dir and os.path.exists(tmp_cifar10_dir):
            shutil.rmtree(tmp_cifar10_dir)

    print(f"\n{'=' * 50}\nEvaluation Results:\n{'=' * 50}")
    for key, value in sorted(metrics.items()):
        print(f"  {key}: {value:.4f}")
    print(f"{'=' * 50}")

    save_path = args.save_json or os.path.join(
        os.path.dirname(args.samples_dir.rstrip("/")), "metrics.json"
    )
    save_path = os.path.abspath(save_path)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nMetrics saved to: {save_path}")
    return metrics


if __name__ == "__main__":
    main()
