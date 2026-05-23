"""
CIFAR-10 / NCSN++ / linear schedule.

The schedule and hyperparameters here match the linear-schedule result
reported in Table 1 of the paper (FID 2.65 / IS 9.63).
"""

import ml_collections
from utils import make_k_grid_dct, powerlaw_psd
from configs.cifar10_defaults import get_default_configs


def get_config():
    config = get_default_configs()

    # ---- training ----
    training = config.training
    training.epochs = 1000
    training.batch_size = 128
    training.ckpt_every = 50000
    training.log_every = 100
    training.eval_every = 50000
    training.eval_nbatch = 4
    training.sample_every = 5000
    training.sample_shape = (8, 3, 32, 32)
    training.compile = True
    training.compile_mode = "default"
    training.amp = True
    training.amp_dtype = "auto"
    training.shell_weighted = True
    training.schedule_sampler = "uniform"
    training.sr_step = None

    # ---- sampling ----
    sampling = config.sampling
    sampling.shape = (64, 3, 32, 32)
    sampling.num_samples = 50000
    sampling.batch_size = 32

    data = config.data

    # ---- model (NCSN++) ----
    model = config.model
    model.name = "NCSNpp-S"
    model.ema_rate = 0.999

    model.dropout = 0.1
    model.fourier_scale = 16
    model.scale_by_sigma = False
    model.normalization = "GroupNorm"
    model.nonlinearity = "swish"
    model.nf = 256
    model.ch_mult = (1, 2, 2, 2)
    model.num_res_blocks = 8
    model.attn_resolutions = (16,)
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

    # ---- diffusion schedule (linear; paper Appendix B.2) ----
    model.num_scales = 1000           # N
    model.lambda_i = 137.7294
    model.lambda_f = 1.57
    model.snr_threshold = 1e-4        # SNR cutoff used to define effective res.
    model.kc = 3                      # near-DC cutoff
    model.theta = 5.0                 # overall noise scale
    model.lambda_schedule = "linear"

    model.prediction_type = "eps"
    model.abar_cutoff = 1e-6

    _, k2 = make_k_grid_dct(data.image_size, data.image_size)
    k2_expanded = k2.expand(1, data.image_size, data.image_size)
    model.k2 = k2_expanded
    model.S_0 = powerlaw_psd(k2_expanded, alpha=1.051304, C=9.1001e-01, k02=1.940556)

    # ---- optimization ----
    config.optim = optim = ml_collections.ConfigDict()
    optim.optimizer = "AdamW"
    optim.lr = 2e-4
    optim.beta1 = 0.9
    optim.eps = 1e-8
    optim.weight_decay = 0.0
    optim.warmup = 5000
    optim.grad_clip = 1.0
    return config
