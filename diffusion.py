"""
SKILD: Scale-invariant K-Space Image Learning Diffusion.

Implements the frequency-space scale-invariant diffusion process from
the paper. The forward marginal is

    X(k, t) = sqrt(abar_t(k)) . X0(k)
            + sqrt(1 - abar_t(k)) . sqrt(S0(k)) . eps,

where ``abar_t(k) = exp(-k^2 . lambda(t))`` damps high-frequency modes
before low-frequency ones, and the injected noise carries the dataset
variance spectrum ``S0(k)``.

A low-frequency cutoff ``kc`` regularizes the k -> 0 limit by replacing
``k`` with ``max(|k|, kc)`` in the damping exponent, which preserves the
algebra above while controlling the otherwise-trivial dynamics of the
near-DC modes (see Section 4.2 of the paper).

The class exposes:
  - the forward marginal ``q_sample``;
  - ``ground_truth_init_x_k``, the exact-forward-marginal initializer used
    for all super-resolution experiments in the paper;
  - the ancestral DDPM-style sampler in k-space;
  - eps / x0 / xprev / w / v prediction losses (the paper uses 'eps').
"""

from __future__ import annotations
from typing import Literal, Tuple

import torch
from torch import Tensor

from utils import safe_div, dct_2d, idct_2d

LambdaScheduleKind = Literal["log_linear", "linear"]


class SKILD:
    """
    Scale-invariant K-space diffusion process.

    Parameters
    ----------
    N : int
        Number of discrete diffusion steps.
    k2 : Tensor
        Squared-frequency grid ``k^2`` of shape ``(1, H, W)`` produced by
        ``utils.make_k_grid_dct``.
    S_0 : Tensor
        Per-mode variance spectrum ``S0(k)`` of shape ``(1, H, W)`` (or
        broadcastable to it). Typically a power-law fit from
        ``utils.powerlaw_psd``.
    lambda_i, lambda_f : float
        Schedule endpoints (paper Eq. 8). ``lambda_i`` primarily controls
        the high-frequency, early-time behavior of ``lambda(t)``;
        ``lambda_f`` (with ``kc`` and ``theta``) primarily controls the
        low-frequency, late-time behavior.
    kc : float
        Low-frequency cutoff (radians). Modes with ``|k| <= kc`` use
        ``kc`` in the damping exponent.
    abar_cutoff : float
        Numerical floor on ``sqrt(abar)`` used in the eps -> x0 inversion
        to keep reverse updates bounded for highly attenuated modes
        (paper Section 4.2).
    schedule : {'log_linear', 'linear'}
        Form of ``lambda(t)`` (paper Section 4.3).
    theta : float
        Overall noise scale for the linear schedule.
    """

    def __init__(
        self,
        N: int,
        k2: Tensor,
        S_0: Tensor,
        lambda_i: float | None = None,
        lambda_f: float | None = None,
        kc: float = 0.0,
        abar_cutoff: float = 1e-6,
        global_seed: int = 42,
        device: str | int = "cuda",
        schedule: LambdaScheduleKind = "log_linear",
        theta: float = 5.0,
    ):
        if lambda_i is None or lambda_f is None:
            raise TypeError(
                "SKILD requires lambda_i and lambda_f (see docstring for their "
                "meaning under each schedule)."
            )
        if schedule not in ("log_linear", "linear"):
            raise ValueError(
                f"Unknown schedule {schedule!r}; use 'log_linear' or 'linear'."
            )

        self.N = int(N)
        self.lambda_i = float(lambda_i)
        self.lambda_f = float(lambda_f)
        self.schedule = schedule
        self.theta = float(theta)
        self.kc = float(kc)
        self.kc2 = self.kc ** 2
        self.abar_cutoff = float(abar_cutoff)

        self.device = device
        self.k2 = k2.to(self.device)
        self.S_0 = S_0.to(self.device)
        self.S_0_sqrt = torch.sqrt(torch.clamp(self.S_0, min=0.0))
        self.k2_eff = torch.clamp(self.k2, min=self.kc2)

        self.rng = torch.Generator(device=self.device).manual_seed(global_seed)

        _, lam = self.compute_lambda_schedule(
            self.N, self.lambda_i, self.lambda_f,
            schedule=self.schedule, theta=self.theta, device=self.device,
        )
        self.lam = lam

        (
            self.betas,
            self.alphas,
            self.alphas_cumprod,
            self.sqrt_alphas_cumprod,
            self.sqrt_1m_alphas_cumprod,
        ) = self._make_ddpm_params(self.lam, self.k2_eff)
        self.sqrt_alphas = torch.sqrt(self.alphas)

        (
            self.posterior_variance,
            self.posterior_mean_c1,
            self.posterior_mean_c2,
        ) = self._make_posterior_params(
            self.alphas_cumprod, self.sqrt_alphas,
            self.sqrt_alphas_cumprod, self.S_0,
        )

        # Shell-local weights for the v-space loss: Delta abar[n] = abar[n] - abar[n+1].
        # Used to weight per-mode errors by the amount of signal removed at step n.
        self.shell_weight = self.alphas_cumprod[:-1] - self.alphas_cumprod[1:]
        with torch.no_grad():
            invs = []
            for n in range(self.N):
                w = self.shell_weight[n : n + 1]
                m = w.mean(dim=tuple(range(1, w.ndim)), keepdim=True).clamp(min=1e-20)
                invs.append(1.0 / m)
            self._shell_weight_norm_inv = torch.cat(invs, dim=0)

    # ------------------------------------------------------------------ #
    #  lambda(t) schedule                                                 #
    # ------------------------------------------------------------------ #
    @staticmethod
    @torch.no_grad()
    def compute_lambda_schedule(
        N: int,
        lambda_i: float,
        lambda_f: float,
        *,
        schedule: LambdaScheduleKind = "log_linear",
        theta: float = 5.0,
        device=None,
    ) -> Tuple[Tensor, Tensor]:
        """Discrete schedule ``lambda(t)`` on ``t in {0, 1/N, ..., 1}``.

        log_linear:  lambda(t) = t * 10^(lambda_i + (lambda_f - lambda_i) * t)
        linear:      lambda(t) = theta * t / (lambda_i*(1 - t) + lambda_f)^2

        The leading factor of ``t`` enforces ``lambda(0) = 0`` so that the
        forward process has a smooth onset (paper Appendix B.2).
        """
        if N < 1:
            raise ValueError("N must be >= 1")
        t = torch.linspace(0.0, 1.0, N + 1, device=device)
        if schedule == "log_linear":
            s = lambda_i + (lambda_f - lambda_i) * t
            lam = t * torch.pow(10.0, s)
        elif schedule == "linear":
            kf = lambda_i * (1.0 - t) + lambda_f
            lam = theta * t / (kf * kf)
        else:
            raise ValueError(f"Unknown schedule {schedule!r}")
        return t, lam

    # ------------------------------------------------------------------ #
    #  DDPM coefficients                                                  #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _make_ddpm_params(self, lam: Tensor, k2_eff: Tensor):
        N = lam.shape[0] - 1
        k2_eff = k2_eff.to(self.device)

        d_lam = lam[1:] - lam[:-1]
        view_shape = (N,) + (1,) * k2_eff.ndim
        d_sig = d_lam.view(view_shape)
        k2_view = k2_eff.unsqueeze(0)

        alphas = torch.exp(-k2_view * d_sig)
        betas = 1.0 - alphas

        lam_view = lam.view(N + 1, *([1] * k2_eff.ndim))
        alphas_cumprod = torch.exp(-k2_view[0] * lam_view)

        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_1m_alphas_cumprod = torch.sqrt(torch.clamp(1.0 - alphas_cumprod, min=0.0))
        return betas, alphas, alphas_cumprod, sqrt_alphas_cumprod, sqrt_1m_alphas_cumprod

    @torch.no_grad()
    def _make_posterior_params(self, alphas_cumprod, sqrt_alphas, sqrt_alphas_cumprod, S_0):
        S_0_view = S_0
        while S_0_view.ndim < alphas_cumprod.ndim:
            S_0_view = S_0_view.unsqueeze(0)
        Sigma = S_0_view * (1.0 - alphas_cumprod)

        Sigma_t = Sigma[:-1]
        Sigma_tp1 = Sigma[1:]
        A = sqrt_alphas

        G_all = safe_div(A * Sigma_t, Sigma_tp1)
        posterior_var = Sigma_t - safe_div((A * Sigma_t) ** 2, Sigma_tp1)
        posterior_var = torch.clamp(posterior_var, min=0.0)

        sqrt_ab_t = sqrt_alphas_cumprod[:-1]
        sqrt_ab_tp1 = sqrt_alphas_cumprod[1:]
        C2_all = G_all
        C1_all = sqrt_ab_t - G_all * sqrt_ab_tp1
        return posterior_var, C1_all, C2_all

    # ------------------------------------------------------------------ #
    #  Forward process                                                    #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def q_sample(self, x0_k: Tensor, t: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Closed-form forward marginal in k-space.

        Returns ``(x_t_k, eps, signal)`` where
        ``x_t_k = sqrt(abar_t) . x0_k + sqrt(S_0 . (1 - abar_t)) . eps``.
        """
        B = x0_k.shape[0]
        assert t.shape[0] == B
        assert 0 <= t.min() and t.max() <= self.N

        sqrt_ab_t = self.sqrt_alphas_cumprod[t]
        sqrt_1m_ab_t = self.sqrt_1m_alphas_cumprod[t]

        eps = torch.randn(x0_k.shape, generator=self.rng, device=self.device, dtype=x0_k.dtype)
        signal = sqrt_ab_t * x0_k
        noise = self.S_0_sqrt * sqrt_1m_ab_t * eps
        return signal + noise, eps, signal

    @torch.no_grad()
    def ground_truth_init_x_k(
        self,
        x_pixel: Tensor,
        init_timestep: int,
        eps: Tensor | None = None,
    ) -> Tensor:
        """k-space initializer from the **exact forward marginal** (paper protocol).

        Used in all ImageNet and Ising super-resolution experiments
        (Section 5.2 and Appendix D.2). Pass ``init_timestep = n0`` to
        ``sample`` / ``ancestral_sampler`` so the reverse
        chain runs ``n = n0, ..., 1``.

        If ``eps`` is supplied (matching ``x0_k`` in shape), it is used for
        the noise term; otherwise noise is drawn from the internal RNG.
        """
        if x_pixel.dim() != 4:
            raise ValueError(f"x_pixel must be (B, C, H, W), got shape {tuple(x_pixel.shape)}")
        x_pixel = x_pixel.to(device=self.device, dtype=self.S_0_sqrt.dtype)
        B = x_pixel.shape[0]
        n0 = int(init_timestep)
        if not (1 <= n0 <= self.N):
            raise ValueError(f"init_timestep must be in [1, {self.N}], got {n0}")

        x0_k = dct_2d(x_pixel, norm="forward")
        t = torch.full((B,), n0, device=self.device, dtype=torch.long)
        if eps is None:
            x_init_k, _, _ = self.q_sample(x0_k, t)
            return x_init_k

        if eps.shape != x0_k.shape:
            raise ValueError(f"eps must match x0_k shape {tuple(x0_k.shape)}, got {tuple(eps.shape)}")
        eps = eps.to(device=self.device, dtype=x0_k.dtype)
        sqrt_ab_t = self.sqrt_alphas_cumprod[t]
        sqrt_1m_ab_t = self.sqrt_1m_alphas_cumprod[t]
        return sqrt_ab_t * x0_k + self.S_0_sqrt * sqrt_1m_ab_t * eps

    # ------------------------------------------------------------------ #
    #  Prediction targets                                                 #
    # ------------------------------------------------------------------ #
    def compute_w_target(self, x0_k: Tensor, eps: Tensor, labels: Tensor) -> Tensor:
        """Unwhitened v-prediction target: ``w_t(k) = a_t.sqrt(S_0).eps - b_t.x0(k)``."""
        a = self.sqrt_alphas_cumprod[labels]
        b = self.sqrt_1m_alphas_cumprod[labels]
        return a * self.S_0_sqrt * eps - b * x0_k

    def compute_v_target(self, x0_k: Tensor, eps: Tensor, labels: Tensor) -> Tensor:
        """Whitened v-prediction target: ``v_t(k) = a_t.eps - b_t.y0(k)`` with ``y0 = x0 / sqrt(S_0)``."""
        a = self.sqrt_alphas_cumprod[labels]
        b = self.sqrt_1m_alphas_cumprod[labels]
        y0 = safe_div(x0_k, self.S_0_sqrt)
        return a * eps - b * y0

    # ------------------------------------------------------------------ #
    #  Inverse maps from network predictions                              #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def w_to_x0(self, x_n_k: Tensor, w_hat_k: Tensor, n: Tensor) -> Tensor:
        """x0_hat from w-prediction: ``a_t . x_t - b_t . w_hat``."""
        a = self.sqrt_alphas_cumprod[n]
        b = self.sqrt_1m_alphas_cumprod[n]
        return a * x_n_k - b * w_hat_k

    @torch.no_grad()
    def w_to_z(self, x_n_k: Tensor, w_hat_k: Tensor, n: Tensor) -> Tensor:
        """Colored noise estimate ``z_hat = sqrt(S_0) . eps_hat`` from w-prediction."""
        a = self.sqrt_alphas_cumprod[n]
        b = self.sqrt_1m_alphas_cumprod[n]
        return b * x_n_k + a * w_hat_k

    @torch.no_grad()
    def eps_to_x0(self, x_n_k: Tensor, eps_hat_k: Tensor, n: Tensor) -> Tensor:
        """x0_hat from eps-prediction (uses ``abar_cutoff`` floor for stability)."""
        sqrt_abar = self.sqrt_alphas_cumprod[n]
        sqrt_1m_abar = self.sqrt_1m_alphas_cumprod[n]
        sqrt_abar_safe = torch.clamp(sqrt_abar, min=self.abar_cutoff)
        return (x_n_k - self.S_0_sqrt * sqrt_1m_abar * eps_hat_k) / sqrt_abar_safe

    # ------------------------------------------------------------------ #
    #  Reverse posterior                                                  #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def q_posterior_mean_variance(
        self,
        x_next: Tensor,
        x0_k: Tensor,
        t: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Exact posterior ``q(x_t | x_{t+1}, x0)`` in Kalman form (0-indexed ``t``)."""
        B = x0_k.shape[0]
        assert t.shape[0] == B
        assert 0 <= t.min() and t.max() <= self.N - 1

        C1 = self.posterior_mean_c1[t]
        C2 = self.posterior_mean_c2[t]
        var = self.posterior_variance[t]
        return C1 * x0_k + C2 * x_next, var

    @torch.no_grad()
    def p_sample(
        self,
        x_n_k: Tensor,
        pred_k: Tensor,
        n: int,
        prediction_type: str = "eps",
    ) -> Tensor:
        """One reverse step ``p(x_{n-1} | x_n)`` with the chosen prediction parameterization."""
        B = x_n_k.shape[0]
        labels = torch.full((B,), n, device=self.device, dtype=torch.long)
        t = labels - 1

        if prediction_type == "eps":
            x0_hat_k = self.eps_to_x0(x_n_k, pred_k, labels)
            mu, var = self.q_posterior_mean_variance(x_n_k, x0_hat_k, t)
        elif prediction_type == "w":
            x0_hat_k = self.w_to_x0(x_n_k, pred_k, labels)
            mu, var = self.q_posterior_mean_variance(x_n_k, x0_hat_k, t)
        elif prediction_type == "v":
            w_hat_k = self.S_0_sqrt * pred_k
            x0_hat_k = self.w_to_x0(x_n_k, w_hat_k, labels)
            mu, var = self.q_posterior_mean_variance(x_n_k, x0_hat_k, t)
        elif prediction_type == "x0":
            mu, var = self.q_posterior_mean_variance(x_n_k, pred_k, t)
        elif prediction_type == "xprev":
            mu = pred_k
            var = self.posterior_variance[t]
        else:
            raise ValueError(f"Unknown prediction_type: {prediction_type}")

        noise = torch.randn(x_n_k.shape, generator=self.rng, device=self.device, dtype=x_n_k.dtype)
        return mu + torch.sqrt(var) * noise if n > 1 else mu

    # ------------------------------------------------------------------ #
    #  Training losses                                                    #
    # ------------------------------------------------------------------ #
    def losses_eps(self, config, model, x0, model_kwargs={}, labels=None):
        """eps-prediction loss (paper default; converged fastest in ablations)."""
        B = x0.shape[0]
        if labels is None:
            labels = torch.randint(low=1, high=self.N + 1, size=(B,),
                                   device=self.device, dtype=torch.long)
        x0_k = dct_2d(x0, norm="forward")
        x_t_k, eps, _ = self.q_sample(x0_k, labels)
        x_t = idct_2d(x_t_k, norm="forward")

        eps_hat = model(x_t, labels, **model_kwargs)
        eps_hat_k = dct_2d(eps_hat, norm="forward")

        losses = torch.square(eps_hat_k - eps)
        losses = torch.mean(losses.reshape(B, -1), dim=-1)
        return losses, labels

    def losses_x0(self, config, model, x0, model_kwargs={}, labels=None):
        """x0-prediction loss."""
        B = x0.shape[0]
        if labels is None:
            labels = torch.randint(low=1, high=self.N + 1, size=(B,),
                                   device=self.device, dtype=torch.long)
        x0_k = dct_2d(x0, norm="forward")
        x_t_k, _, _ = self.q_sample(x0_k, labels)
        x_t = idct_2d(x_t_k, norm="forward")

        x0_hat = model(x_t, labels, **model_kwargs)
        x0_hat_k = dct_2d(x0_hat, norm="forward")

        losses = torch.square(x0_hat_k - x0_k)
        losses = torch.mean(losses.reshape(B, -1), dim=-1)
        return losses, labels

    def losses_xprev(self, config, model, x0, model_kwargs={}, labels=None):
        """Posterior-mean (x_{prev}) prediction loss."""
        B = x0.shape[0]
        if labels is None:
            labels = torch.randint(low=1, high=self.N + 1, size=(B,),
                                   device=self.device, dtype=torch.long)
        x0_k = dct_2d(x0, norm="forward")
        x_t_k, _, _ = self.q_sample(x0_k, labels)

        t = labels - 1
        C1 = self.posterior_mean_c1[t]
        C2 = self.posterior_mean_c2[t]
        mu_true_k = C1 * x0_k + C2 * x_t_k

        x_t = idct_2d(x_t_k, norm="forward")
        mu_hat = model(x_t, labels, **model_kwargs)
        mu_hat_k = dct_2d(mu_hat, norm="forward")

        losses = torch.square(mu_hat_k - mu_true_k)
        losses = torch.mean(losses.reshape(B, -1), dim=-1)
        return losses, labels

    def losses_w(self, config, model, x0, model_kwargs={}, labels=None):
        """Unwhitened v-prediction (w) loss in k-space.

        With ``config.training.shell_weighted = True`` (default), the loss is
        evaluated in whitened v-space with per-mode shell weighting
        ``lambda_t(k) ~ abar_{t-1} - abar_t``; otherwise it is plain MSE on w.
        """
        B = x0.shape[0]
        shell_weighted = getattr(config.training, "shell_weighted", True)
        if labels is None:
            labels = torch.randint(low=1, high=self.N + 1, size=(B,),
                                   device=self.device, dtype=torch.long)

        x0_k = dct_2d(x0, norm="forward")
        x_t_k, eps, _ = self.q_sample(x0_k, labels)
        w_target_k = self.compute_w_target(x0_k, eps, labels)

        x_t = idct_2d(x_t_k, norm="forward")
        w_hat = model(x_t, labels, **model_kwargs)
        w_hat_k = dct_2d(w_hat, norm="forward")

        if shell_weighted:
            v_diff = safe_div(w_hat_k - w_target_k, self.S_0_sqrt)
            weight = self.shell_weight[labels - 1] * self._shell_weight_norm_inv[labels - 1]
            losses = weight * v_diff.square()
        else:
            losses = (w_hat_k - w_target_k).square()

        losses = torch.mean(losses.reshape(B, -1), dim=-1)
        return losses, labels

    def losses_v(self, config, model, x0, model_kwargs={}, labels=None):
        """Whitened v-prediction loss (same shell-weighting option as ``losses_w``)."""
        B = x0.shape[0]
        shell_weighted = getattr(config.training, "shell_weighted", True)
        if labels is None:
            labels = torch.randint(low=1, high=self.N + 1, size=(B,),
                                   device=self.device, dtype=torch.long)

        x0_k = dct_2d(x0, norm="forward")
        x_t_k, eps, _ = self.q_sample(x0_k, labels)
        v_target_k = self.compute_v_target(x0_k, eps, labels)

        x_t = idct_2d(x_t_k, norm="forward")
        v_hat = model(x_t, labels, **model_kwargs)
        v_hat_k = dct_2d(v_hat, norm="forward")

        if shell_weighted:
            diff = v_hat_k - v_target_k
            weight = self.shell_weight[labels - 1] * self._shell_weight_norm_inv[labels - 1]
            losses = weight * diff.square()
        else:
            losses = (v_hat_k - v_target_k).square()

        losses = torch.mean(losses.reshape(B, -1), dim=-1)
        return losses, labels

    # ------------------------------------------------------------------ #
    #  Sampler                                                            #
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def sample(
        self,
        config,
        model,
        shape: Tuple[int, int, int, int],
        prediction_type: str = "eps",
        log_path: bool = False,
        model_kwargs: dict = {},
        x_init_k: Tensor | None = None,
        init_timestep: int | None = None,
    ) -> Tensor:
        """Run the ancestral reverse process (paper protocol)."""
        return self.ancestral_sampler(
            config, model, shape,
            prediction_type=prediction_type, log_path=log_path,
            model_kwargs=model_kwargs,
            x_init_k=x_init_k, init_timestep=init_timestep,
        )

    @torch.no_grad()
    def ancestral_sampler(
        self,
        config,
        model,
        shape: Tuple[int, int, int, int],
        prediction_type: str = "eps",
        log_path: bool = False,
        model_kwargs: dict = {},
        x_init_k: Tensor | None = None,
        init_timestep: int | None = None,
    ) -> Tensor:
        """Ancestral sampling over all k modes (one step per n).

        From pure noise: one reverse step for each n = N, N-1, ..., 1.
        With ``(x_init_k, init_timestep = n0)``: starts at x_{n0} and runs only
        n = n0, n0-1, ..., 1 (used for super-resolution).
        """
        std = self.S_0_sqrt * self.sqrt_1m_alphas_cumprod[self.N]
        if x_init_k is not None:
            if init_timestep is None:
                raise ValueError("init_timestep is required when x_init_k is provided.")
            n0 = int(init_timestep)
            if not (1 <= n0 <= self.N):
                raise ValueError(f"init_timestep must be in [1, {self.N}], got {n0}.")
            x_t_k = x_init_k.to(device=self.device, dtype=std.dtype)
            n_range = range(1, n0 + 1)
        else:
            if init_timestep is not None:
                raise ValueError("init_timestep is only used together with x_init_k.")
            x_t_k = std * torch.randn(shape, generator=self.rng, device=self.device)
            n_range = range(1, self.N + 1)

        path = [x_t_k.clone()] if log_path else None
        B = shape[0]
        for n in reversed(n_range):
            labels = torch.full((B,), n, device=self.device, dtype=torch.long)
            x_t = idct_2d(x_t_k, norm="forward")
            pred = model(x_t, labels, **model_kwargs)
            pred_k = dct_2d(pred, norm="forward")
            x_t_k = self.p_sample(x_t_k, pred_k, n, prediction_type=prediction_type)
            if log_path:
                path.append(x_t_k.clone())

        x0 = idct_2d(x_t_k, norm="forward")
        if log_path:
            return torch.stack([idct_2d(p, norm="forward") for p in path], dim=0)
        return x0
