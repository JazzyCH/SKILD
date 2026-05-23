"""
Model and optimizer construction helpers.

The score backbone is NCSN++ (Song et al., 2021); see ``models/ncsnpp/``
for the architecture. Checkpoints save both the live model weights and
the EMA copy used for sampling.
"""

from __future__ import annotations
import os
from copy import deepcopy

import numpy as np
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.utils import clip_grad_norm_

from models.ncsnpp.models import Unet_score_models


def _requires_grad(module, flag: bool = True):
    for p in module.parameters():
        p.requires_grad = flag


def create_model(config, rank, device, logger):
    """Build the score U-Net, wrap with DDP, and create a frozen EMA copy.

    Returns ``(ddp_model, ema)``. The EMA copy is stepped manually in the
    training loop via ``train.update_ema``.
    """
    model = Unet_score_models[config.model.name](config)
    ema = deepcopy(model).to(device)
    _requires_grad(ema, False)
    model = DDP(model.to(device), device_ids=[rank])
    logger.info(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, ema


def load_model(config, ckpt_path: str, device):
    """Load EMA weights from a checkpoint into a fresh NCSN++ model."""
    model = Unet_score_models[config.model.name](config).to(device)
    assert os.path.isfile(ckpt_path), f"Could not find checkpoint at {ckpt_path}"
    checkpoint = torch.load(
        ckpt_path, map_location=lambda storage, loc: storage, weights_only=False,
    )
    if "ema" in checkpoint:
        checkpoint = checkpoint["ema"]
    model.load_state_dict(checkpoint)
    model.eval()
    return model


def get_optimizer(config, params):
    """Build an Adam or AdamW optimizer per ``config.optim``."""
    if config.optim.optimizer == "Adam":
        return optim.Adam(
            params, lr=config.optim.lr,
            betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
    if config.optim.optimizer == "AdamW":
        return optim.AdamW(
            params, lr=config.optim.lr,
            betas=(config.optim.beta1, 0.999), eps=config.optim.eps,
            weight_decay=config.optim.weight_decay,
        )
    raise NotImplementedError(f"Optimizer {config.optim.optimizer} not supported")


def optimization_manager(config):
    """Return a closure that applies LR warmup, grad-clipping, and ``optimizer.step``."""

    def optimize_fn(
        optimizer, params, step,
        lr=config.optim.lr,
        warmup=config.optim.warmup,
        grad_clip=config.optim.grad_clip,
    ):
        params = [p for p in params if p.grad is not None]
        if warmup > 0:
            for g in optimizer.param_groups:
                g["lr"] = lr * np.minimum(step / warmup, 1.0)
        if params:
            if grad_clip >= 0:
                grad_norm = clip_grad_norm_(params, max_norm=grad_clip)
            else:
                grad_norm = torch.norm(
                    torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2,
                )
        else:
            grad_norm = torch.tensor(0.0)
        optimizer.step()
        return grad_norm

    return optimize_fn
