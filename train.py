"""
Distributed training entry point for SKILD.

Trains a frequency-space scale-invariant diffusion model on CIFAR-10,
ImageNet (128 or 256), or the critical Ising dataset, using NCSN++ as
the score backbone. Mid-training, the script periodically samples from
the model for visualization:

  * **Unconditional generation** (default). Used to monitor CIFAR-10
    runs; intermediate samples are drawn from pure noise.
  * **Super-resolution preview**. Triggered when
    ``config.training.sr_step`` is set. The latest training images are
    used as ground-truth HR references; the reverse chain is
    initialized from the exact forward marginal at timestep
    ``sr_step`` (see ``SKILD.ground_truth_init_x_k``), which matches
    the protocol used for paper SR evaluation.

Run with::

    torchrun --nproc_per_node=N train.py \\
        --config configs.specifics.cifar10_linear

Resume with ``--resume_from /path/to/checkpoint.pt``.
"""

from __future__ import annotations

# Enable TF32 / high-precision tensor-core paths on Ampere+ GPUs.
import torch
from torch import amp as torch_amp
from torch.nn.utils import clip_grad_norm_
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.set_float32_matmul_precision("high")

import argparse
import importlib
import json
import os
from glob import glob
from time import time

import numpy as np
import torch.distributed as dist
from torchvision.utils import make_grid, save_image

import models.mutils as mutils
from datasets import (
    get_dataloader,
    get_dataloader_cifar10,
    get_dataloader_in_memory,
)
from diffusion import SKILD
from schedule_sampler import LossAwareSampler, create_schedule_sampler
from utils import create_logger, make_k_grid_dct


# --------------------------------------------------------------------- #
#  Training helpers                                                     #
# --------------------------------------------------------------------- #
@torch.no_grad()
def update_ema(ema_model, model, decay: float = 0.9999):
    """Step the EMA weights towards the current model parameters (vectorized)."""
    ema_p = [p for p in ema_model.parameters()]
    model_p = [p.data for p in model.parameters()]
    torch._foreach_mul_(ema_p, decay)
    torch._foreach_add_(ema_p, model_p, alpha=1 - decay)


def _ddpm_loss(ddpm, config, model, x, model_kwargs, labels, prediction_type):
    """Dispatch on ``prediction_type`` to the matching SKILD loss."""
    if prediction_type == "eps":
        return ddpm.losses_eps(config, model, x, model_kwargs, labels=labels)
    if prediction_type == "xprev":
        return ddpm.losses_xprev(config, model, x, model_kwargs, labels=labels)
    if prediction_type == "x0":
        return ddpm.losses_x0(config, model, x, model_kwargs, labels=labels)
    if prediction_type == "w":
        return ddpm.losses_w(config, model, x, model_kwargs, labels=labels)
    if prediction_type == "v":
        return ddpm.losses_v(config, model, x, model_kwargs, labels=labels)
    raise ValueError(f"Unknown prediction_type: {prediction_type}")


# --------------------------------------------------------------------- #
#  Training loop                                                         #
# --------------------------------------------------------------------- #
def train(config, result_dir, resume_from=None, experiment_index=None):
    assert torch.cuda.is_available(), "Training requires at least one GPU."

    # DDP setup
    dist.init_process_group("nccl")
    assert config.training.batch_size % dist.get_world_size() == 0
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = config.seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")

    # Experiment directories
    if rank == 0:
        os.makedirs(result_dir, exist_ok=True)
        if resume_from is not None:
            experiment_dir = os.path.dirname(os.path.dirname(resume_from))
        else:
            if experiment_index is None:
                experiment_index = len(glob(f"{result_dir}/*"))
            model_string = config.model.name.replace("/", "-")
            experiment_dir = f"{result_dir}/{experiment_index:03d}-{model_string}"
        checkpoint_dir = f"{experiment_dir}/checkpoints"
        eval_dir = f"{experiment_dir}/eval"
        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(eval_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(
            f"{'Resuming into' if resume_from else 'Created'} experiment directory: {experiment_dir}"
        )
        with open(f"{experiment_dir}/config.json", "w") as f:
            json.dump(config.to_dict(), f, indent=2, default=str)
    else:
        logger = create_logger(None)

    # k-space grid and S_0
    image_size = config.data.image_size
    _, k2 = make_k_grid_dct(image_size, image_size, use_radians=True)
    k2 = k2.expand(1, image_size, image_size).to(device)
    S_0 = config.model.S_0
    if isinstance(S_0, torch.Tensor):
        S_0 = S_0.to(device)

    # Diffusion process
    prediction_type = getattr(config.model, "prediction_type", "eps")
    kc = getattr(config.model, "kc", 0.0)
    abar_cutoff = getattr(config.model, "abar_cutoff", 1e-6)
    lambda_sched = getattr(config.model, "lambda_schedule", "log_linear")
    theta_sched = getattr(config.model, "theta", 3.0)
    logger.info(
        f"SKILD: prediction_type={prediction_type}, kc={kc}, abar_cutoff={abar_cutoff}, "
        f"schedule={lambda_sched}, theta={theta_sched}"
    )
    ddpm = SKILD(
        N=config.model.num_scales,
        k2=k2, S_0=S_0,
        lambda_i=config.model.lambda_i,
        lambda_f=config.model.lambda_f,
        kc=kc, abar_cutoff=abar_cutoff,
        device=device, schedule=lambda_sched, theta=theta_sched,
    )

    # NCSN++ uses sqrt(lambda) as its noise-level embedding.
    lam_embed = ddpm.lam
    config.model.skild_sigmas = torch.sqrt(lam_embed).cpu().numpy().tolist()
    config.model.skild_lambda = lam_embed.cpu().numpy().tolist()

    sampler_name = getattr(config.training, "schedule_sampler", "uniform")
    schedule_sampler = create_schedule_sampler(sampler_name, ddpm.N)
    logger.info(f"Schedule sampler: {sampler_name}")
    logger.info(f"Image size: {image_size}")
    logger.info(f"kc={ddpm.kc:.4f}, kc^2={ddpm.kc2:.4f}")
    logger.info(f"lambda range: [{ddpm.lam[0]:.6e}, {ddpm.lam[-1]:.6e}]")

    # Super-resolution preview during training (uses ground-truth forward marginal).
    sr_step_cfg = getattr(config.training, "sr_step", None)
    use_sr_preview = sr_step_cfg is not None
    if rank == 0 and use_sr_preview:
        logger.info(
            f"SR preview: sr_step={sr_step_cfg} "
            "(reverse chain initialized from exact forward marginal of HR batch)"
        )

    # Model + EMA
    model, ema = mutils.create_model(config, rank, device, logger)

    # torch.compile
    use_compile = getattr(config.training, "compile", False)
    compile_mode = getattr(config.training, "compile_mode", "default")
    if use_compile:
        logger.info(f"Compiling model with torch.compile(mode='{compile_mode}') (~60s first step)...")
        model = torch.compile(model, mode=compile_mode)

    # Mixed precision (bf16 preferred; fp16 + GradScaler on older GPUs)
    use_amp = bool(getattr(config.training, "amp", False))
    amp_dtype = torch.float32
    scaler = None
    if use_amp:
        ad = getattr(config.training, "amp_dtype", "auto")
        if ad == "auto":
            if torch.cuda.is_bf16_supported():
                amp_dtype = torch.bfloat16
            else:
                amp_dtype = torch.float16
                try:
                    scaler = torch_amp.GradScaler("cuda")
                except TypeError:
                    scaler = torch_amp.GradScaler()
                if rank == 0:
                    logger.info("AMP: float16 + GradScaler (bf16 not supported on this GPU).")
        elif ad in ("bfloat16", "bf16"):
            amp_dtype = torch.bfloat16
        elif ad in ("float16", "fp16"):
            amp_dtype = torch.float16
            try:
                scaler = torch_amp.GradScaler("cuda")
            except TypeError:
                scaler = torch_amp.GradScaler()
        else:
            if rank == 0:
                logger.warning(f"Unknown training.amp_dtype={ad!r}; disabling AMP.")
            use_amp = False
        if use_amp and rank == 0 and amp_dtype == torch.bfloat16:
            logger.info("AMP: bfloat16 autocast (no GradScaler).")

    # Optimizer
    optimizer = mutils.get_optimizer(config, model.parameters())
    optimize_fn = mutils.optimization_manager(config)

    # Resume from checkpoint
    start_epoch = 0
    train_steps = 0
    if resume_from is not None:
        if rank == 0:
            logger.info(f"Resuming training from checkpoint: {resume_from}")
        checkpoint = torch.load(
            resume_from,
            map_location=lambda storage, loc: storage.cuda(device),
            weights_only=False,
        )
        model.module.load_state_dict(checkpoint["model"])
        ema.load_state_dict(checkpoint["ema"])
        optimizer.load_state_dict(checkpoint["opt"])
        start_epoch = checkpoint.get("epoch", 0)
        train_steps = checkpoint.get("train_steps", 0)
        if scaler is not None and "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        if rank == 0:
            logger.info(f"Resumed from epoch {start_epoch}, step {train_steps}")
        dist.barrier()

    # Data
    dataset_name = config.data.dataset.lower()
    persistent_workers = getattr(config.data, "persistent_workers", False)
    prefetch_factor = getattr(config.data, "prefetch_factor", None)
    if "cifar10" in dataset_name:
        cifar10_root = getattr(config.data, "cifar10_root", config.data.train_path)
        train_loader, train_sampler = get_dataloader_cifar10(
            data_root=cifar10_root, image_size=image_size,
            global_batch_size=config.training.batch_size,
            num_workers=config.data.num_workers, global_seed=config.seed,
            train=True, logger=logger,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
        )
        eval_loader, _ = get_dataloader_cifar10(
            data_root=cifar10_root, image_size=image_size,
            global_batch_size=config.training.batch_size,
            num_workers=config.data.num_workers, global_seed=config.seed,
            train=False, logger=logger,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
        )
    else:
        use_in_memory = getattr(config.data, "in_memory", False)
        imagefolder_loader = (
            get_dataloader_in_memory if use_in_memory else get_dataloader
        )
        if rank == 0 and use_in_memory:
            logger.info("data.in_memory=True: ImageFolder bytes cached per rank.")
        train_loader, train_sampler = imagefolder_loader(
            data_path=config.data.train_path, image_size=image_size,
            global_batch_size=config.training.batch_size,
            num_workers=config.data.num_workers, global_seed=config.seed,
            train=True, logger=logger,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
        )
        eval_loader, _ = imagefolder_loader(
            data_path=config.data.eval_path, image_size=image_size,
            global_batch_size=config.training.batch_size,
            num_workers=config.data.num_workers, global_seed=config.seed,
            train=False, logger=logger,
            persistent_workers=persistent_workers, prefetch_factor=prefetch_factor,
        )

    # Prepare for training
    if resume_from is None:
        update_ema(ema, model.module, decay=0)
    model.train()
    ema.eval()

    log_steps = 0
    running_loss = 0.0
    running_grad_norm = 0.0
    start_time = time()
    logger.info(f"Training for {config.training.epochs} epochs...")

    # Microbatching: split each local batch and accumulate gradients.
    local_batch_size = config.training.batch_size // dist.get_world_size()
    microbatch = getattr(config.training, "microbatch", 0)
    if microbatch <= 0 or microbatch > local_batch_size:
        microbatch = local_batch_size
    accum_steps = local_batch_size // microbatch
    if accum_steps > 1:
        logger.info(
            f"Local batch={local_batch_size}, microbatch={microbatch}, accum_steps={accum_steps}"
        )

    # Rank-0 buffer of the most recent HR batch for SR-preview sampling.
    sr_hr_buffer = None

    for epoch in range(start_epoch, config.training.epochs):
        train_sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            if rank == 0 and use_sr_preview:
                sr_hr_buffer = x[: min(16, x.shape[0])].detach().clone()

            model_kwargs = dict(y=y) if hasattr(config.model, "num_classes") else {}
            optimizer.zero_grad()

            batch_loss = 0.0
            for i in range(0, x.shape[0], microbatch):
                micro_x = x[i : i + microbatch]
                micro_kwargs = {k: v[i : i + microbatch] for k, v in model_kwargs.items()}
                ts, weights = schedule_sampler.sample(micro_x.shape[0], device)

                with torch_amp.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                    per_sample_losses, labels = _ddpm_loss(
                        ddpm, config, model, micro_x, micro_kwargs, ts, prediction_type,
                    )
                    loss = (per_sample_losses * weights).mean() / accum_steps

                if isinstance(schedule_sampler, LossAwareSampler):
                    schedule_sampler.update_with_local_losses(labels, per_sample_losses.detach())

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                batch_loss += loss.item()

            # Optimizer step
            if scaler is not None:
                scaler.unscale_(optimizer)
                warmup = getattr(config.optim, "warmup", 0)
                lr = config.optim.lr
                if warmup > 0:
                    for g in optimizer.param_groups:
                        g["lr"] = lr * np.minimum(train_steps / warmup, 1.0)
                params = [p for p in model.parameters() if p.grad is not None]
                grad_clip = getattr(config.optim, "grad_clip", -1)
                if params:
                    if grad_clip >= 0:
                        grad_norm = clip_grad_norm_(params, max_norm=grad_clip)
                    else:
                        grad_norm = torch.norm(
                            torch.stack([torch.norm(p.grad.detach(), 2) for p in params]), 2,
                        )
                else:
                    grad_norm = torch.tensor(0.0, device=device)
                scaler.step(optimizer)
                scaler.update()
                grad_norm = float(grad_norm)
            else:
                grad_norm = float(optimize_fn(optimizer, model.parameters(), train_steps))

            update_ema(ema, model.module, decay=config.model.ema_rate)

            running_loss += batch_loss
            running_grad_norm += grad_norm
            log_steps += 1
            train_steps += 1

            # Logging
            if train_steps % config.training.log_every == 0:
                torch.cuda.synchronize()
                steps_per_sec = log_steps / (time() - start_time)

                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()

                avg_grad_norm = torch.tensor(running_grad_norm / log_steps, device=device)
                dist.all_reduce(avg_grad_norm, op=dist.ReduceOp.SUM)
                avg_grad_norm = avg_grad_norm.item() / dist.get_world_size()

                logger.info(
                    f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, "
                    f"Grad Norm: {avg_grad_norm:.4f}, Steps/Sec: {steps_per_sec:.2f}"
                )
                running_loss = running_grad_norm = 0.0
                log_steps = 0
                start_time = time()

            # Checkpoint
            if train_steps % config.training.ckpt_every == 0 and train_steps > 0:
                if rank == 0:
                    checkpoint = {
                        "model": model.module.state_dict(),
                        "ema": ema.state_dict(),
                        "opt": optimizer.state_dict(),
                        "epoch": epoch,
                        "train_steps": train_steps,
                        "config": config,
                    }
                    if scaler is not None:
                        checkpoint["scaler"] = scaler.state_dict()
                    path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, path)
                    logger.info(f"Saved checkpoint to {path}")
                dist.barrier()

            # Eval-loss evaluation
            if train_steps % config.training.eval_every == 0 and train_steps > 0:
                model.eval()
                ema.eval()
                eval_loss, eval_steps = 0.0, 0
                eval_batches = getattr(config.training, "eval_nbatch", 4)
                with torch.no_grad():
                    if eval_loader is not None:
                        for i, (x_eval, y_eval) in enumerate(eval_loader):
                            if i >= eval_batches:
                                break
                            x_eval = x_eval.to(device, non_blocking=True)
                            y_eval = y_eval.to(device, non_blocking=True)
                            eval_kwargs = (
                                dict(y=y_eval) if hasattr(config.model, "num_classes") else {}
                            )
                            with torch_amp.autocast(
                                device_type="cuda", dtype=amp_dtype, enabled=use_amp,
                            ):
                                eval_losses, _ = _ddpm_loss(
                                    ddpm, config, ema, x_eval, eval_kwargs, None, prediction_type,
                                )
                            eval_loss += eval_losses.mean().item()
                            eval_steps += 1
                        if eval_steps > 0:
                            tensor = torch.tensor(eval_loss / eval_steps, device=device)
                            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                            avg = tensor.item() / dist.get_world_size()
                            logger.info(f"(step={train_steps:07d}) Eval Loss: {avg:.4f}")
                model.train()
                dist.barrier()

            # Sampling preview
            sample_every = getattr(config.training, "sample_every", config.training.eval_every)
            if train_steps % sample_every == 0 and train_steps > 0:
                model.eval()
                ema.eval()
                if rank == 0:
                    with torch.no_grad():
                        B_sample = min(16, config.training.batch_size)
                        sample_shape = (
                            B_sample, config.data.num_channels, image_size, image_size,
                        )
                        with torch_amp.autocast(
                            device_type="cuda", dtype=amp_dtype, enabled=use_amp,
                        ):
                            if use_sr_preview and sr_hr_buffer is not None:
                                B_eff = min(B_sample, sr_hr_buffer.shape[0])
                                sample_shape = (
                                    B_eff, config.data.num_channels, image_size, image_size,
                                )
                                x_init_k = ddpm.ground_truth_init_x_k(
                                    sr_hr_buffer[:B_eff],
                                    init_timestep=int(sr_step_cfg),
                                )
                                samples = ddpm.sample(
                                    config, ema, sample_shape,
                                    prediction_type=prediction_type,
                                    log_path=False,
                                    x_init_k=x_init_k,
                                    init_timestep=int(sr_step_cfg),
                                )
                            else:
                                samples = ddpm.sample(
                                    config, ema, sample_shape,
                                    prediction_type=prediction_type,
                                    log_path=False,
                                )
                        grid = make_grid(samples, nrow=4, normalize=True, value_range=(-1, 1))
                        save_image(grid, f"{eval_dir}/samples_{train_steps:07d}.png")
                        logger.info(f"Saved samples to {eval_dir}/samples_{train_steps:07d}.png")
                model.train()
                dist.barrier()

    model.eval()
    logger.info("Training complete!")
    dist.destroy_process_group()


# --------------------------------------------------------------------- #
#  CLI                                                                  #
# --------------------------------------------------------------------- #
def load_config(module_path: str):
    """Import a config module and call its ``get_config()``."""
    module = importlib.import_module(module_path)
    assert hasattr(module, "get_config"), f"Module {module_path!r} must define get_config()."
    return module.get_config()


def parse_args():
    parser = argparse.ArgumentParser(description="SKILD training")
    parser.add_argument("--config", required=True,
                        help="Python module path to config (e.g. configs.specifics.cifar10_linear)")
    parser.add_argument("--result_dir", default="results", help="Top-level results directory")
    parser.add_argument("--resume_from", help="Path to checkpoint to resume from")
    parser.add_argument("--experiment_index", type=int, default=None,
                        help="Force experiment subdirectory index (e.g. 0 -> 000-ModelName)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    train(config, args.result_dir, resume_from=args.resume_from,
          experiment_index=args.experiment_index)


if __name__ == "__main__":
    main()
