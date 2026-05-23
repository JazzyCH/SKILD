"""
ImageNet 256 / 4 x super-resolution (64 -> 256) / NCSN++.

Schedule and architecture for the 4 x SR row of Table 4 (paper Sec. 5.2)
at 256 x 256 resolution.
"""

import ml_collections
from utils import make_k_grid_dct, powerlaw_psd
from configs.imagenet256_defaults import get_default_configs


def get_config():
    config = get_default_configs()

    # ---- training ----
    training = config.training
    training.epochs = 100
    training.batch_size = 256
    training.microbatch = 16
    training.ckpt_every = 10000
    training.log_every = 100
    training.eval_every = 10000
    training.eval_nbatch = 1
    training.sample_every = 5000
    training.sample_shape = (8, 3, 256, 256)

    training.compile = True
    training.compile_mode = "default"
    training.amp = True
    training.amp_dtype = "auto"
    training.shell_weighted = True
    training.schedule_sampler = "uniform"

    # SR preview during training (GT-init from exact forward marginal at sr_step).
    training.sr_step = 1000

    # Paired SR evaluation cache (consumed by sr_eval.py).
    training.gt_forward_k_path = "/path/to/data/imagenet256_64_val.pt"
    training.gt_init_timestep = 1000

    # ---- sampling ----
    sampling = config.sampling
    sampling.shape = (16, 3, 256, 256)
    sampling.num_samples = 50000
    sampling.batch_size = 16

    # ---- data ----
    data = config.data
    data.dataset = "IMAGENET256"
    data.image_size = 256
    data.in_memory = False

    # ---- model (NCSN++) ----
    model = config.model
    model.name = "NCSNpp-S"
    model.ema_rate = 0.9999

    model.dropout = 0.1
    model.fourier_scale = 16
    model.scale_by_sigma = False
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 128
    model.ch_mult = (1, 1, 2, 2, 2, 2)
    model.num_res_blocks = 6
    model.attn_resolutions = (32, 16, 8)
    model.resamp_with_conv = True
    model.conditional = True
    model.fir = True
    model.fir_kernel = [1, 3, 3, 1]
    model.skip_rescale = True
    model.resblock_type = "biggan"
    model.progressive = "none"
    model.progressive_input = "residual"
    model.progressive_combine = "sum"
    model.attention_type = "ddpm"
    model.embedding_type = "positional"
    model.init_scale = 0.0
    model.conv_size = 3

    # ---- diffusion schedule ----
    model.num_scales = 1000
    model.lambda_i = 1132.9352
    model.lambda_f = 550.8723
    model.snr_threshold = 1e-1
    model.kc = 0
    model.theta = 9.0
    model.lambda_schedule = "linear"

    model.prediction_type = "eps"
    model.abar_cutoff = 1e-6

    _, k2 = make_k_grid_dct(data.image_size, data.image_size)
    k2_expanded = k2.expand(1, data.image_size, data.image_size)
    model.k2 = k2_expanded
    model.S_0 = powerlaw_psd(k2, alpha=1.059750, C=9.3221e-01, k02=1.570796)

    # ---- optimization ----
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "AdamW"
    optim.lr = 1e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.weight_decay = 1e-5
    optim.warmup = 1000
    optim.grad_clip = 1.0
    return config
