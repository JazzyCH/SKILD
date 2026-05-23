"""
Dataset utilities for SKILD: distributed ``ImageFolder`` / ``CIFAR10``
loaders with the ADM-style center-crop preprocessing, plus an optional
in-memory variant that caches JPEG bytes per rank to amortize disk I/O
on large datasets.
"""

from __future__ import annotations
import io
from functools import partial

import numpy as np
import torch.distributed as dist
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder


def center_crop_arr(pil_image, image_size):
    """ADM-style center crop with progressive area halving + bicubic resize.

    Adapted from
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size),
            resample=Image.Resampling.BOX,
        )
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size),
        resample=Image.Resampling.BICUBIC,
    )
    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


def _make_transform(image_size: int, train: bool):
    """[-1, 1]-normalized 3-channel transform with optional random flip."""
    ops = [transforms.Lambda(partial(center_crop_arr, image_size=image_size))]
    if train:
        ops.append(transforms.RandomHorizontalFlip())
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
    ]
    return transforms.Compose(ops)


def get_dataloader(
    data_path,
    image_size=256,
    global_batch_size=64,
    num_workers=4,
    global_seed=42,
    train=True,
    logger=None,
    persistent_workers=False,
    prefetch_factor=None,
):
    """Distributed ImageFolder DataLoader (used for ImageNet)."""
    transform = _make_transform(image_size, train)
    dataset = ImageFolder(data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=True,
        seed=global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    if logger is not None:
        logger.info(
            f"Dataset ({'train' if train else 'test'}) contains "
            f"{len(dataset):,} images ({data_path})"
        )
    return loader, sampler


class InMemoryImageFolder(ImageFolder):
    """ImageFolder that caches raw JPEG/PNG bytes in RAM (per-rank).

    Avoids repeated disk reads at the cost of memory; useful when the
    dataset fits in RAM and disk I/O is the bottleneck.
    """

    def __init__(self, root, transform=None, **kwargs):
        super().__init__(root, transform=transform, **kwargs)
        self.imgs_bytes = []
        print(f"Loading {len(self.samples)} images into memory from {root}...")
        total, report_every = len(self.samples), 5000
        for idx, (path, _) in enumerate(self.samples, start=1):
            with open(path, "rb") as f:
                self.imgs_bytes.append(f.read())
            if idx % report_every == 0 or idx == total:
                print(f"  Loaded {idx}/{total} images...")
        print("Loading complete.")

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = Image.open(io.BytesIO(self.imgs_bytes[index])).convert("RGB")
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target


def get_dataloader_in_memory(
    data_path,
    image_size=256,
    global_batch_size=64,
    num_workers=4,
    global_seed=42,
    train=True,
    logger=None,
    persistent_workers=False,
    prefetch_factor=None,
):
    """Distributed in-memory ImageFolder DataLoader."""
    transform = _make_transform(image_size, train)
    dataset = InMemoryImageFolder(data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=train,
        seed=global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    if logger is not None:
        logger.info(
            f"In-Memory Dataset ({'train' if train else 'test'}) contains "
            f"{len(dataset):,} images ({data_path})"
        )
    return loader, sampler


def get_dataloader_cifar10(
    data_root,
    image_size=32,
    global_batch_size=64,
    num_workers=4,
    global_seed=42,
    train=True,
    logger=None,
    persistent_workers=False,
    prefetch_factor=None,
):
    """Distributed CIFAR-10 DataLoader using ``torchvision.datasets.CIFAR10``.

    Uses the standard pickle files (``cifar-10-batches-py/``); the entire
    dataset lives in RAM natively.
    """
    cifar_transforms = [transforms.Resize(image_size)]
    if train:
        cifar_transforms.append(transforms.RandomHorizontalFlip())
    cifar_transforms += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
    transform = transforms.Compose(cifar_transforms)

    dataset = CIFAR10(root=data_root, train=train, transform=transform, download=False)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=dist.get_rank(),
        shuffle=train,
        seed=global_seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=int(global_batch_size // dist.get_world_size()),
        shuffle=False,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=train,
        persistent_workers=persistent_workers and num_workers > 0,
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )
    if logger is not None:
        logger.info(
            f"CIFAR10 ({'train' if train else 'test'}) loaded: "
            f"{len(dataset):,} images from {data_root}"
        )
    return loader, sampler
