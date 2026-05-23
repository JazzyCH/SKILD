"""
ImageNet 128 / 4 x super-resolution (32 -> 128) / NCSN++.

Schedule and architecture used by the continuous-SR figure on
ImageNet-128 (paper Sec. 5.2). The effective resolution at sampling
start is set so that all frequencies up to k = 32 are above the SNR
threshold tau = 0.1.
"""

import ml_collections
from utils import make_k_grid_dct, powerlaw_psd
from configs.imagenet128_defaults import get_default_configs


def get_config():
    config = get_default_configs()

    # ---- training ----
    training = config.training
    training.epochs = 100
    training.batch_size = 256
    training.microbatch = 128
    training.ckpt_every = 10000
    training.log_every = 100
    training.eval_every = 10000
    training.eval_nbatch = 1
    training.sample_every = 5000
    training.sample_shape = (8, 3, 128, 128)

    training.compile = True
    training.compile_mode = "default"
    training.amp = True
    training.amp_dtype = "auto"
    training.shell_weighted = True
    training.schedule_sampler = "uniform"

    # SR preview during training (qualitative diagnostic only).
    # GT-init: reverse chain starts from the exact forward marginal at sr_step.
    training.sr_step = 1000

    # Paired SR evaluation cache (consumed by sr_eval.py).
    training.gt_forward_k_path = "/path/to/data/imagenet128_32_val.pt"
    training.gt_init_timestep = 1000

    # ---- sampling ----
    sampling = config.sampling
    sampling.shape = (16, 3, 128, 128)
    sampling.num_samples = 50000
    sampling.batch_size = 16

    # ---- data ----
    data = config.data
    data.dataset = "IMAGENET128"
    data.image_size = 128
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
    model.lambda_i = 564.2461
    model.lambda_f = 275.4361
    model.snr_threshold = 1e-1
    model.kc = 0
    model.theta = 9.0
    model.lambda_schedule = "linear"

    model.prediction_type = "eps"
    model.abar_cutoff = 1e-6

    _, k2 = make_k_grid_dct(data.image_size, data.image_size)
    k2_expanded = k2.expand(1, data.image_size, data.image_size)
    model.k2 = k2_expanded
    model.S_0 = powerlaw_psd(k2, alpha=1.061318, C=9.0617e-01, k02=1.570796)

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
