"""Default config for CIFAR-10 SKILD training."""

import ml_collections
from utils import make_k_grid_dct, powerlaw_psd


def get_default_configs():
    config = ml_collections.ConfigDict()
    config.seed = 42

    # ---- training ----
    config.training = training = ml_collections.ConfigDict()
    training.mode = "ddpm"
    training.batch_size = 128
    training.epochs = 1000
    training.ckpt_every = 10000
    training.log_every = 100
    training.eval_every = 1000
    training.sample_every = 5000
    training.sample_shape = (8, 3, 32, 32)
    training.compile = True
    training.compile_mode = "default"
    training.amp = True
    training.amp_dtype = "auto"  # 'auto' | 'bfloat16' | 'float16'
    training.shell_weighted = True  # w/v losses: per-mode shell weighting
    training.sr_step = None  # if set, enables GT-init SR preview during training

    # ---- sampling ----
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.shape = (64, 3, 32, 32)
    sampling.num_samples = 50000
    sampling.batch_size = 64

    config.eval = ml_collections.ConfigDict()

    # ---- data ----
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "CIFAR10"
    data.image_size = 32
    data.num_channels = 3
    data.num_classes = 10
    data.train_path = "/path/to/data/cifar10/train"
    data.eval_path = "/path/to/data/cifar10/val"
    data.cifar10_root = "/path/to/data/cifar10"
    data.num_workers = 4
    data.persistent_workers = True
    data.prefetch_factor = 2

    # ---- model / diffusion ----
    config.model = model = ml_collections.ConfigDict()
    model.num_scales = 1000
    model.lambda_i = -4.75
    model.lambda_f = -0.5
    _, k2 = make_k_grid_dct(data.image_size, data.image_size)
    k2_expanded = k2.expand(1, data.image_size, data.image_size)
    model.k2 = k2_expanded
    # Empirical CIFAR-10 spectrum fit (paper Table 2 / Appendix B.1).
    model.S_0 = powerlaw_psd(k2_expanded, alpha=1.051304, C=9.1001e-01, k02=1.940556)
    model.abar_cutoff = 1e-6

    # ---- optimization ----
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "AdamW"

    return config
