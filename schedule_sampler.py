"""
Timestep schedule samplers.

Adapted from guided-diffusion's resample.py:
https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/resample.py

Importance sampling over diffusion timesteps. The expected gradient is
unchanged (unbiased); variance is reduced by sampling timesteps with
larger losses more often. SKILD uses 1-indexed timesteps ``n in [1, N]``;
internal arrays are 0-indexed of length ``N`` with the offset handled
transparently.
"""

from abc import ABC, abstractmethod
import numpy as np
import torch
import torch.distributed as dist


class ScheduleSampler(ABC):
    """Distribution over diffusion timesteps for variance reduction."""

    @abstractmethod
    def weights(self):
        """Numpy array of unnormalized positive weights, one per timestep."""

    def sample(self, batch_size, device):
        """Importance-sample timesteps for one batch.

        Returns ``(timesteps, weights)`` where ``timesteps`` are 1-indexed in
        ``[1, N]`` and ``weights`` are the corresponding importance weights.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        timesteps = torch.from_numpy(indices_np).long().to(device) + 1
        weights_np = 1.0 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return timesteps, weights


class UniformSampler(ScheduleSampler):
    """Uniform sampling over ``[1, N]``. Importance weights are all 1."""

    def __init__(self, num_timesteps):
        self.num_timesteps = num_timesteps
        self._weights = np.ones([num_timesteps], dtype=np.float64)

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    """Base class for samplers that adapt based on observed losses."""

    def update_with_local_losses(self, local_ts, local_losses):
        """Update reweighting from per-rank losses, with DDP synchronization."""
        batch_sizes = [
            torch.tensor([0], dtype=torch.int32, device=local_ts.device)
            for _ in range(dist.get_world_size())
        ]
        dist.all_gather(
            batch_sizes,
            torch.tensor([len(local_ts)], dtype=torch.int32, device=local_ts.device),
        )
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [torch.zeros(max_bs).to(local_ts) for _ in batch_sizes]
        loss_batches = [torch.zeros(max_bs).to(local_losses) for _ in batch_sizes]
        dist.all_gather(timestep_batches, local_ts)
        dist.all_gather(loss_batches, local_losses)
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [
            x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]
        ]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """Update reweighting from gathered (all-rank) losses; called identically on every rank."""


class LossSecondMomentResampler(LossAwareSampler):
    """Sample timesteps proportionally to ``sqrt(E[L_t^2])`` over a sliding window.

    Until every timestep has been observed at least ``history_per_term``
    times (warmup), falls back to uniform sampling. A small ``uniform_prob``
    floor prevents any timestep from being completely starved of samples.
    """

    def __init__(self, num_timesteps, history_per_term=10, uniform_prob=0.001):
        self.num_timesteps = num_timesteps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros([num_timesteps, history_per_term], dtype=np.float64)
        self._loss_counts = np.zeros([num_timesteps], dtype=np.int64)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            idx = int(t) - 1
            if self._loss_counts[idx] == self.history_per_term:
                self._loss_history[idx, :-1] = self._loss_history[idx, 1:]
                self._loss_history[idx, -1] = loss
            else:
                self._loss_history[idx, self._loss_counts[idx]] = loss
                self._loss_counts[idx] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()


def create_schedule_sampler(name, num_timesteps):
    """Factory: ``'uniform'`` (paper default) or ``'loss-second-moment'``."""
    if name == "uniform":
        return UniformSampler(num_timesteps)
    elif name == "loss-second-moment":
        return LossSecondMomentResampler(num_timesteps)
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")
