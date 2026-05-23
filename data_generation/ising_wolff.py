"""
2D Ising-model sample generator at the critical temperature using the Wolff
cluster algorithm.

Used to produce the Ising training/evaluation set referenced in paper
Sec. 5.3 / Appendix D. Each generated PNG is a single configuration of an
L x L lattice sampled at beta = beta_c = log(1 + sqrt(2)) / 2 with spins
mapped to {0, 255}.

Example
-------
::

    python -m data_generation.ising_wolff \
        --L 1024 --n_samples 50000 --batch 8 \
        --out_dir /path/to/data/ising_1024/train
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class WolffSampler2D:
    """Batched 2D Ising Wolff sampler with single-cluster updates."""

    L: int
    beta: float
    J: float = 1.0
    device: str = "cuda"
    spin_dtype: torch.dtype = torch.int8
    seed: int | None = None

    def __post_init__(self):
        self.p = float(1.0 - torch.exp(torch.tensor(-2.0 * self.beta * self.J)).item())
        self.N = self.L * self.L
        self._rng = torch.Generator(device=self.device)
        if self.seed is not None:
            self._rng.manual_seed(self.seed)
        else:
            self._rng.seed()
        # Lookup table q_k = 1 - (1 - p)^k for k in {0, 1, 2, 3, 4}.
        op = 1.0 - self.p
        self.q_lut = torch.tensor(
            [0.0, 1.0 - op, 1.0 - op ** 2, 1.0 - op ** 3, 1.0 - op ** 4],
            device=self.device,
            dtype=torch.float32,
        )

    def _rand(self, *shape, device=None) -> torch.Tensor:
        return torch.rand(*shape, device=device or self.device, generator=self._rng)

    def _randint(self, low: int, high: int, shape: tuple, device=None) -> torch.Tensor:
        return torch.randint(low, high, shape, device=device or self.device, generator=self._rng)

    def init_spins(self, batch: int) -> torch.Tensor:
        r = self._randint(0, 2, (batch, self.L, self.L))
        return (2 * r - 1).to(self.spin_dtype)

    @torch.no_grad()
    def wolff_update(self, spins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """One Wolff single-cluster update per batch element.

        Implements independent bond activation: a candidate site with ``k``
        frontier neighbors joins the cluster with probability
        ``1 - (1 - p)^k``.
        """

        B, L, _ = spins.shape
        device = spins.device

        sy = self._randint(0, L, (B,), device=device)
        sx = self._randint(0, L, (B,), device=device)

        batch_idx = torch.arange(B, device=device)
        seed_mask = torch.zeros((B, L, L), device=device, dtype=torch.bool)
        seed_mask[batch_idx, sy, sx] = True

        seed_spin = spins[batch_idx, sy, sx].view(B, 1, 1)
        same = spins == seed_spin

        cluster = seed_mask.clone()
        frontier = seed_mask
        active = torch.ones((B,), device=device, dtype=torch.bool)

        while active.any():
            k = (
                torch.roll(frontier, +1, 1).to(torch.int8)
                + torch.roll(frontier, -1, 1).to(torch.int8)
                + torch.roll(frontier, +1, 2).to(torch.int8)
                + torch.roll(frontier, -1, 2).to(torch.int8)
            )
            candidates = (k > 0) & same & (~cluster)
            q = self.q_lut[k.to(torch.long)]
            new_sites = candidates & (self._rand(B, L, L, device=device) < q)

            cluster |= new_sites
            frontier = new_sites
            active = frontier.flatten(1).any(dim=1)

        spins = torch.where(cluster, -spins, spins)
        cs = cluster.sum(dim=(1, 2), dtype=torch.int32)
        return spins, cs

    @torch.no_grad()
    def thermalize(self, spins: torch.Tensor, wolff_flips: int) -> torch.Tensor:
        for _ in range(wolff_flips):
            spins, _ = self.wolff_update(spins)
        return spins


if __name__ == "__main__":
    import argparse
    import os

    from PIL import Image

    parser = argparse.ArgumentParser(description="Generate critical 2D Ising configurations (Wolff).")
    parser.add_argument("--L", type=int, default=1024, help="Lattice side length L (image is L x L).")
    parser.add_argument("--n_samples", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=4, help="Number of parallel Markov chains.")
    parser.add_argument("--therm_flips", type=int, default=1000, help="Wolff flips per chain for warm-up.")
    parser.add_argument(
        "--target_sweeps_between",
        type=float,
        default=2.0,
        help="Lattice sweeps worth of spin flips between saved samples (per chain).",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    beta_c = 0.5 * torch.log(torch.tensor(1.0 + 2 ** 0.5)).item()
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Device: {device}  L={args.L}  beta_c={beta_c:.6f}  n_samples={args.n_samples}")

    sampler = WolffSampler2D(L=args.L, beta=beta_c, device=device, seed=args.seed)
    spins = sampler.init_spins(args.batch)
    print(f"Thermalizing ({args.therm_flips} Wolff flips)...")
    spins = sampler.thermalize(spins, args.therm_flips)

    target = int(args.target_sweeps_between * sampler.N)
    progressed = torch.zeros((args.batch,), device=device, dtype=torch.int64)

    saved = 0
    while saved < args.n_samples:
        spins, cs = sampler.wolff_update(spins)
        progressed += cs.to(progressed.dtype)

        if bool((progressed >= target).all()):
            for j in range(args.batch):
                if saved >= args.n_samples:
                    break
                img_np = ((spins[j].cpu().numpy().astype("int16") + 1) * 127).astype("uint8")
                Image.fromarray(img_np, mode="L").save(os.path.join(args.out_dir, f"ising_{saved:05d}.png"))
                saved += 1
            progressed.zero_()
            print(f"Saved {saved}/{args.n_samples}", flush=True)

    print(f"Done. {saved} images saved to {args.out_dir}")
