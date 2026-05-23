"""Default config for ImageNet 256 x 256 SKILD super-resolution."""

import ml_collections
from utils import make_k_grid_dct, powerlaw_psd


def get_default_configs():
    config = ml_collections.ConfigDict()
    config.seed = 42

    # ---- training ----
    config.training = training = ml_collections.ConfigDict()
    training.mode = "ddpm"
    training.batch_size = 128
    training.epochs = 100
    training.ckpt_every = 10000
    training.log_every = 100
    training.eval_every = 1000
    training.sample_every = 5000
    training.sample_shape = (8, 3, 256, 256)
    training.compile = True
    training.compile_mode = "default"
    training.amp = True
    training.amp_dtype = "auto"
    training.shell_weighted = True
    training.sr_step = None  # if set, enables GT-init SR preview during training

    # ---- sampling ----
    config.sampling = sampling = ml_collections.ConfigDict()
    sampling.shape = (16, 3, 256, 256)
    sampling.num_samples = 50000
    sampling.batch_size = 16

    config.eval = ml_collections.ConfigDict()

    # ---- data ----
    config.data = data = ml_collections.ConfigDict()
    data.dataset = "IMAGENET256"
    data.image_size = 256
    data.num_channels = 3
    data.num_classes = 1000
    data.train_path = "/path/to/data/imagenet256/train"
    data.eval_path = "/path/to/data/imagenet256/val"
    data.num_workers = 8
    data.persistent_workers = True
    data.prefetch_factor = 2

    # ---- model / diffusion ----
    config.model = model = ml_collections.ConfigDict()
    model.num_scales = 1000
    model.lambda_i = -6.0
    model.lambda_f = -0.5
    _, k2 = make_k_grid_dct(data.image_size, data.image_size)
    k2_expanded = k2.expand(1, data.image_size, data.image_size)
    model.k2 = k2_expanded
    # ImageNet-256 spectrum fit (paper Table 2).
    model.S_0 = powerlaw_psd(k2, alpha=1.059750, C=9.3221e-01, k02=1.570796)
    model.abar_cutoff = 1e-6

    # ---- optimization ----
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "AdamW"
    return config
