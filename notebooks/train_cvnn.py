# ════════════════════════════════════════════════════════════════════════════
# train_cvnn.py · last edited 03 May 2025
#
# End-to-end training / evaluation harness for three CNN variants:
#   1. Complex-valued frequency-domain CNN      (ComplexValuedNN)
#   2. Real-valued   frequency-domain CNN       (RealValuedNN)
#   3. Real-valued   time-domain CNN/LSTM       (RealValuedNNtd)
#
# This script *glues* data-gen, models, schedulers and TensorBoard together.
# Heavy lifting lives in:
#   · cvnn_models.py  – all network definitions / residual blocks
#   · cvnn_data.py    – synthetic glitch data + caching
#
# New in this revision
# ────────────────────
# • Switched to the std-lib `logging` module → unified, filterable output.
# • Added high-level docstrings & inline comments in “hot” sections.
# • Replaced ad-hoc prints in hooks/train-loop with   logger.debug/info().
# • Guarded CUDA memory dump (`memsum`) behind `logger.debug` to reduce noise.
# ════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────── std-lib ────────────────────────────────────
import os, sys, random, re, json, math, shutil, time, logging
from pathlib import Path
from typing import Iterator, Sized, List, Tuple

# ────────────────────── third-party / data-science ──────────────────────────
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt
# change to agg
plt.switch_backend("agg")  # no X11 on server
import complextorch.nn as cvnn          # pip install complextorch
import complextorch.nn.functional as cvF
from functools import wraps
import wandb
# ───────────────────────── project-local imports ────────────────────────────
from cvnn_models import (
    ComplexValuedNN,
    RealValuedNN,
    RealValuedNNtd,
    GlitchNetRes,
    GlitchNetCV,
    GlitchRegressor,
)
from cvnn_data import get_data, GlitchDataset
from cvnn_models import _CVBlock
from loggers import DualLogger

# ════════════════════════════════════════════════════════════════════════════
#                               LOGGING SET-UP
# ════════════════════════════════════════════════════════════════════════════
LOG_LEVEL = os.getenv("CVNN_LOG_LEVEL", "INFO").upper()  # DEBUG / INFO / …
logging.basicConfig(
    stream=sys.stdout,
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)-7s | %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger("cvnn.main")  # use child-loggers in helpers
logging.getLogger('matplotlib').setLevel(logging.INFO)  # quieten matplotlib

# ═════════════════════ reproducibility & device detection ═══════════════════
torch.backends.cuda.matmul.allow_tf32 = True            # TF32 on Ampere+
torch.autograd.set_detect_anomaly(False)                 # flag NaNs/Infs
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
if torch.cuda.is_available():
    device = torch.device("cuda")
# elif torch.backends.mps.is_available():
    # device = torch.device("mps")
else:
    device = torch.device("cpu")
logger.info(f"Using device: {device}")

# Global RNGs for deterministic-ish runs
rng   = np.random.default_rng(seed=0)
t_rng = torch.Generator().manual_seed(0)

_grad_history = []          # grows to [epochs × n_layers]
_capture_acts: bool = False
_latest_acts: list = []

# ═══════════════════════════ complextorch PATCH ═════════════════════════════
# # Quietly patch an upstream bug in _whiten2x2_batch_norm (only once)
if not getattr(cvF, "_is_whiten2x2_bn_patched", False):
    _orig_whiten = cvF._whiten2x2_batch_norm

    @wraps(_orig_whiten)
    def _whiten2x2_batch_norm_fixed(
            x: torch.Tensor,
            training: bool = True,
            running_mean=None,
            running_cov=None,
            momentum: float = 0.1,
            eps: float = 1e-5):

        if (not training) and (running_mean is not None):
            # bring running_mean to shape [2, 1, F, 1, …]
            tail = (1, x.shape[2], *([1] * (x.dim() - 3)))
            running_mean = running_mean.real.view(2, *tail)

        return _orig_whiten(
            x, training, running_mean, running_cov, momentum, eps
        )

    cvF._whiten2x2_batch_norm = _whiten2x2_batch_norm_fixed
    cvF._is_whiten2x2_bn_patched = True
    logger.debug("Patched complextorch.nn.functional._whiten2x2_batch_norm")


# ══════════════════════ tiny diagnostic helper (CUDA mem) ═══════════════════
def memsum() -> None:
    """Dump a *full* CUDA memory summary (DEBUG-only)."""
    if device.type == "cuda":
        logger.debug("\n" + torch.cuda.memory_summary(None, abbreviated=False))

# ═════════════════════════════ HYPER-PARAMETERS ═════════════════════════════
NTRAIN, NTEST     = 70_000, 30_000
BATCH             = 2096
AMP               = False            # mixed precision ⇢ set True if desired

NUM_EPOCHS        = 800
WARMUP_EPOCHS     = 5
BASE_LR, PEAK_LR  = 2e-3, 3e-3

conv_filters      = [16, 32, 64]     # channels per conv stage
n_conv_layers     = 3
conv_kernel_size  = [4, 8, 16]       # (large→small) receptive fields
acc_steps = 1

ROOT      = Path("/Users/xangm/OneDrive/repos/antiglitch/")
# ROOT = Path(
    # "/Users/xangma/Library/CloudStorage/OneDrive-Personal/repos/antiglitch/")
DATADIR   = ROOT / "data"
LOGDIR    = ROOT / "runs"
MODEL_DIR = ROOT / "notebooks"

# --------------------------------------------------------------------------
# 8-d ground-truth              ↔  10-d prediction
# 0  modA                       ↔  0  μ_modA
# 1  sin_phi                    ↔  8  sin_phi
# 2  cos_phi                    ↔  9  cos_phi
# 3  Δt                         ↔  2  μ_Δt
# 4  f0                         ↔  4  μ_f0
# 5  gbw                        ↔  6  μ_gbw          (log σ_* at odd indices)
# 6  snr
# 7  residual
# --------------------------------------------------------------------------
field_names = ("modA", "sin_phi", "cos_phi", "t", "f0", "gbw", "snr", "residual")

# |A|, sin, cos, t, f0, gbw, snr, resid
pred_index_for_gt = [0, 12, 13, 2, 4, 6, 8, 10]

field_names14 = (
    "modA", "log_sigma_modA", # 1, 2
    "Δt",   "log_sigma_Δt", # 3, 4
    "f0",   "log_sigma_f0", # 5, 6
    "gbw",  "log_sigma_gbw", # 7, 8
    "snr",  "log_sigma_snr", # 9, 10
    "residual", "log_sigma_residual", # 11, 12
    "sin_phi", "cos_phi" # 13, 14
)

for p in (DATADIR, LOGDIR, MODEL_DIR):
    p.mkdir(parents=True, exist_ok=True)

# ─────────────── architecture “switchboard” (train/test flags) ──────────────
train_complex, test_complex = True,  True
train_real,    test_real    = True,  True
train_real_td, test_real_td = False, False   # LSTM is slow; default off

# ═════════════════════════════ DATA LOADING ═════════════════════════════════
logger.info("Preparing synthetic glitch datasets … (one-off cache build)")
dists, glitches, ifos, keys, dist_amp = get_data(DATADIR)
logger.info(f"Detectors: {ifos} | Glitch classes: {keys}")

# Datasets are cached under DATADIR/*.pt after first run
train_set = GlitchDataset(
    datadir=DATADIR, ifos=ifos, glitches=glitches, distributions=dists, dist_amp=dist_amp,
    tr_size=NTRAIN,  te_size=NTEST,
    device=device,   split="train",  outtype="complex",
    noise=True,      aug_phase=True, aug_time=True
)
test_set  = GlitchDataset(
    datadir=DATADIR, ifos=ifos, glitches=glitches, distributions=dists, dist_amp=dist_amp,
    tr_size=NTRAIN,  te_size=NTEST,
    device=device,   split="test",   outtype="complex",
    noise=True,      aug_phase=True, aug_time=True
)

def collate_identity(batch): return batch

# ═════════════════════════════ LOSS HELPERS ═════════════════════════════════
sigmas = train_set.Y_raw.std(0)
vars_ = torch.as_tensor(sigmas, device=device).pow(2)

def weighted_mse(pred: torch.Tensor, tgt: torch.Tensor,
                 var: torch.Tensor = vars_) -> torch.Tensor:
    """MSE with per-dimension variance normalisation."""
    d = tgt.shape[1]
    return ((pred - tgt).square() / var[:d]).mean()


# ═══════════════════════ LEARNING RATE SCHEDULE ═════════════════════════════
def lr_lambda(ep: int) -> float:
    """Linear warm-up: BASE_LR ➜ PEAK_LR, then constant."""
    if ep < WARMUP_EPOCHS:
        return (PEAK_LR / BASE_LR) * (ep + 1) / WARMUP_EPOCHS
    return 1.0

# ═══════════════════════ DATALOADERS ═══════════════════════

dl_kwargs = dict(drop_last=False, collate_fn=collate_identity)

train_loader = DataLoader(train_set, batch_size=BATCH,
                          shuffle=True, **dl_kwargs)
test_loader = DataLoader(test_set,  batch_size=BATCH,
                         shuffle=False, **dl_kwargs)

print("Sample tensor shapes:", next(iter(train_loader))[0][0].shape)

# ═══════════════════════════ TRAINING UTILITIES ══════════════════════════════


def make_writer(tag: str) -> SummaryWriter:
    """Create a TensorBoard writer under runs/<tag>/."""
    return SummaryWriter(os.path.join(LOGDIR, tag))


def load_latest_ckpt(model: nn.Module, stem: str) -> int:
    """
    Resume from the newest checkpoint matching <stem>_????.pt.
    Returns
    -------
    int : epoch index to *start from* (i.e. previous + 1).
    """
    ckpts = [f for f in os.listdir(MODEL_DIR)
             if f.startswith(stem) and f.endswith(".pt")
             and re.match(r".*_\d+\.pt$", f)]
    if not ckpts:
        return 0
    ckpts.sort(key=lambda s: int(s.split("_")[-1].split(".")[0]))
    best = os.path.join(MODEL_DIR, ckpts[-1])
    print(f"Resuming from checkpoint {best}")
    model.load_state_dict(torch.load(best, map_location=device))
    return int(best.split("_")[-1].split(".")[0]) + 1

# ─────────────────────── metrics for cyclic quantities ──────────────────────


def mse_angle(pred, true) -> torch.Tensor:
    """
    MSE on a wrapped angle in radians (-π…π).
    Equivalent to MSE in (cos, sin) space → removes discontinuity at ±π.
    """
    return ((torch.sin(pred) - torch.sin(true))**2 +
            (torch.cos(pred) - torch.cos(true))**2).mean()


def mse_wrapped(pred, true, period) -> torch.Tensor:
    """
    MSE on a scalar that is periodic (e.g. Δt mod 0.2 s).
    """
    return torch.square(
        torch.remainder(pred - true + 0.5*period, period) - 0.5*period
    ).mean()

# ═════════════════════════ OPTIMISER HELPER ═════════════════════════
def _build_optimizer(model: nn.Module,
                     weight_decay: float,
                     base_lr: float = BASE_LR) -> torch.optim.Optimizer:
    """
    AdamW with *two* parameter groups:

      • **decay**     – all conv/linear weights
      • **no-decay**  – Batch/Layer-Norm γ/β, explicit bias terms,
                        and the special ``weight_matrix_`` kernels
                        used by weight-fractional-moment (wFM) blocks.

    The LR is set to ``2 × base_lr`` to reproduce the historical 2 e-3
    used in the original code.  Tweak if you prefer another value.
    """
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if "weight_matrix_" in name:             # wFM kernel → no decay
            no_decay.append(p)
        elif p.ndim == 1 or "norm" in name.lower() or "bias" in name.lower():
            no_decay.append(p)                   # γ/β and biases
        else:
            decay.append(p)                      # ordinary weights

    return torch.optim.AdamW(
        [
            {"params": decay,    "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=base_lr * 2,          # 2 × base mirrors the old hard-coded 2 e-3
        betas=(0.9, 0.99)
    )

    # ═════════════════════════ LOSS / CRITERION HELPER ═════════════════════════


def _criterion(pred: torch.Tensor,
               target: torch.Tensor,
               log_vars: torch.Tensor | None = None,
               clamp: tuple[float, float] = (-6.0, 6.0),  # e-6  ≤ σ ≤  e6
               eps: float = 1e-6) -> torch.Tensor:
    """
    Handles three heads:
      · 6-D    (μ)                    + global log_vars
      · 10-D   (μ, logσ) ×4           + sinφ, cosφ
      · 14-D   (μ, logσ) ×6           + sinφ, cosφ
    Returns: scalar loss that is numerically stable.
    """
    D = pred.shape[1]

    # ─── phase loss (shared) ───────────────────────────────────────────────
    v_pred, v_true = pred[:, -2:], target[:, 1:3]          # (B,2)
    loss_phi = (v_pred - v_true).pow(2).sum(1).mean()
    loss_phi += 0.1 * (v_pred.norm(dim=1).sub(1).pow(2)).mean()

    # ----------------------------------------------------------------------


    def _safe_nll(mu, logσ, tgt, λ_reg=1e-4):
        """
        Gaussian negative log-likelihood with
            • clamped logσ to avoid exp overflow
            • ε added to variance for numerical safety
            • optional L2 regulariser on logσ to stop σ-inflation
        """
        logσ = torch.clamp(logσ, *clamp)
        var = torch.exp(2 * logσ) + eps            # σ² + ε
        nll = ((tgt - mu).pow(2) / var + 2 * logσ).mean()

        # ── NEW: shrink logσ towards 0 (i.e. σ ≈ 1) ───────────────────────────
        nll += λ_reg * (logσ ** 2).mean()
        return nll

    # ─── heteroscedastic heads ─────────────────────────────────────────────
    if D == 10:                                           # (μ, logσ) ×4
        mu, logσ = pred[:, :8].reshape(-1, 4, 2).unbind(-1)
        tgt = target[:, (0, 3, 4, 5)]
        nll = _safe_nll(mu, logσ, tgt)
        return loss_phi + nll

    elif D == 14:                                         # (μ, logσ) ×6
        mu, logσ = pred[:, :12].reshape(-1, 6, 2).unbind(-1)
        tgt = target[:, (0, 3, 4, 5, 6, 7)]
        nll = _safe_nll(mu, logσ, tgt)
        return loss_phi + nll

    # ─── legacy 6-D head with global log_vars ──────────────────────────────
    elif D == 6 and log_vars is not None:
        idx = torch.as_tensor([0, 3, 4, 5], device=pred.device)
        var = torch.exp(log_vars[idx]).clamp_min(eps)
        nll = ((pred[:, idx] - target[:, idx]).pow(2) /
               var + log_vars[idx]).mean()
        return loss_phi + nll

    else:
        raise ValueError(f"Unsupported output dim {D}")


# ═══════════════════════ VALIDATION PASS HELPER ═════════════════════════
def _validation_pass(model: nn.Module,
                     loader: torch.utils.data.DataLoader,
                     out_dim: int) -> tuple[float, torch.Tensor, torch.Tensor]:
    """
    Run one forward sweep over *loader* (no grad) and return:

        val_loss : float
        all_pred : (N, out_dim) tensor on CPU
        all_y    : (N, out_dim) tensor on CPU
    """
    model.eval()
    val_loss: float = 0.0
    preds, tgts = [], []
    loss_fn = torch.nn.MSELoss()                # after your μ/σ scaling
    with torch.no_grad(), torch.amp.autocast(
            device_type=device.type, dtype=torch.float16, enabled=AMP):
        for step, (xb, yb) in enumerate(loader):
            _capture_acts = (step == 0)
            xb = xb.unsqueeze(1)                         # (B,1,F)
            p = model(xb)
            _capture_acts = False
            val_loss += _criterion(p, yb[:, :out_dim], getattr(model, "log_vars", None)).item()
            preds.append(p.cpu())
            tgts.append(yb[:, :out_dim].cpu())


    val_loss /= len(loader)

    return val_loss, torch.cat(preds), torch.cat(tgts)

# ═══════════════════════ TENSORBOARD VIZ HELPER ══════════════════════════
def _tensorboard_viz(writer: SummaryWriter,
                     model: nn.Module,
                     epoch: int,
                     dummy_x: torch.Tensor,
                     preds: torch.Tensor,
                     tgts: torch.Tensor,
                     where=pred_index_for_gt,
                     projector_every: int = 50) -> None:
    """
    Group all heavyweight TensorBoard logging in one call.

    Parameters
    ----------
    writer  : SummaryWriter currently in use.
    model   : network being trained (needed for weight/grad hists & projector).
    epoch   : integer epoch index.
    tag     : run-specific tag string (e.g. 'complex_2025-05-03').
    dummy_x : 1×1×513 complex tensor for graph / projector tracing.
    preds   : (N, out_dim) validation-set predictions (CPU tensor).
    tgts    : (N, out_dim) validation-set targets      (CPU tensor).
    projector_every : export cadence for the embedding projector.
    """
    tgts = tgts.numpy()
    preds = preds.numpy()
    # ─── once-only graph export ───────────────────────────────────────────
    tb_add_graph_once(writer, model, dummy_x)

    # ─── weight & gradient histograms ─────────────────────────────────────
    tb_weight_and_grad_histograms(writer, model, epoch)

    # ─── scatter / hexbin target vs prediction plots ─────────────────────
    tb_scatter_preds(writer, epoch,
                     tgts, preds,
                     where=pred_index_for_gt,
                     field_names=field_names[:preds.shape[1]])

    # ─── optional feature-embedding projector ────────────────────────────
    if epoch % projector_every == 0:
        tb_project_embedding(writer, model, dummy_x, epoch)

    # --- NEW plots --------------------------------------------------------
    tb_polar_phi(writer, epoch, preds, tgts)                 # 〈—— add
    tb_pred_vs_target_dist(writer, epoch, preds, tgts,
                           names=field_names[:preds.shape[1]])  # 〈—— add

# ════════════════════════ GENERIC TRAINING LOOP ═════════════════════════════
def train_generic(model_fn,
                  tag: str,
                  out_dim: int,
                  train_loader: DataLoader,
                  test_loader:  DataLoader,
                  num_epochs:   int = NUM_EPOCHS,
                  base_lr:      float = BASE_LR,
                  dropout:      float = 0.15,
                  weight_decay: float = 1e-4,
                  log_every:    int = 1,
                  viz_every:    int = 10,
                  projector_every: int = 50) -> Tuple[List[float], List[float]]:
    """
    Universal training wrapper with:
    · AdamW + warm-up + cosine anneal
    · Mixed precision (optional)
    · TensorBoard scalars / hists / figs / projector
    · 10-epoch auto-checkpoint
    """
    logger.info(f"[{tag}] Initialising model")
    # Replace the bare SummaryWriter with the dual wrapper
    tb_writer = SummaryWriter(LOGDIR / tag)

    model: nn.Module = model_fn().to(device)
    model.compile()

    # ── W&B init ──────────────────────────────────────────────
    run = wandb.init(
        project="cvnn",          # change to whatever project name you like
        name   = tag,
        dir    = LOGDIR,               # keeps run-dirs beside TensorBoard logs
        config = {
            "model": model_fn.__name__,
            "epochs": num_epochs,
            "batch": BATCH,
            "lr_base": BASE_LR,
            "lr_peak": PEAK_LR,
            "amp": AMP,
            "device": str(device),
        },
        sync_tensorboard=True,         # auto-streams TB figures/plots to W&B
        save_code=True                 # snapshot the source files
    )

    writer = DualLogger(run, tb_writer)

    # Track gradients & parameters every epoch (like TB’s histograms)
    wandb.watch(model, log="all", log_freq=100, log_graph=False)
    # Kaiming init for (complex) conv kernels  -------------------------------
    for m in model.modules():
        if isinstance(m, cvnn.Conv1d):
            for name, p in m.named_parameters(recurse=False):
                if "weight" in name and p.is_complex():
                    nn.init.kaiming_uniform_(p.data.real, a=math.sqrt(5))
                    nn.init.kaiming_uniform_(p.data.imag, a=math.sqrt(5))

        # Add global log-σ² **only** when the model wants it (6-D head)
        if hasattr(model, "log_vars"):
            model.log_vars.data[:] = torch.log(vars_)
            model.log_vars.requires_grad_(True)

    # Forward-hook (prints a tiny sample and NaN guard)
    hook_handle = _register_output_hook(model, print_every=100)
    act_handles = _register_activation_hooks(model)

    # optimizer = _build_optimizer(model, weight_decay, base_lr)
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr)
    warmup    = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=num_epochs - WARMUP_EPOCHS, eta_min=1e-4
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=200,  # first cycle
        T_mult=1,  # equal-length cycles
        eta_min=1e-5,
    )


    loss_fn = torch.nn.MSELoss()                # after your μ/σ scaling


    scaler   = torch.amp.GradScaler(enabled=AMP)
    ckpt_stem = f"{tag}_model"
    start_ep = load_latest_ckpt(model, ckpt_stem)

    dummy_x = torch.zeros((1, 1, 513), dtype=torch.complex64, device=device)

    train_hist, val_hist = [], []
    # ──────────────────────────────── EPOCH LOOP ────────────────────────────
    for epoch in tqdm(range(start_ep, num_epochs), desc=f"{tag}"):
        # —— training pass ——
        model.train()
        running = 0.0
        for step, (xb, yb) in enumerate(train_loader):
            xb = xb.unsqueeze(1)

            # zero grads **only** at the start of an accumulation window
            if step % acc_steps == 0:
                optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast(device_type=device.type,
                                    dtype=torch.float16, enabled=AMP):
                pred  = model(xb)
                loss  = _criterion(pred, yb[:, :out_dim],
                                getattr(model, "log_vars", None))

                scaler.scale(loss).backward()
            if (step + 1) % acc_steps == 0 or (step + 1) == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()

            running += loss.item()

        train_loss = running / len(train_loader)
        train_hist.append(train_loss)

        # —— validation pass ——
        val_loss, all_pred, all_y = _validation_pass(
            model, test_loader, out_dim
        )
        val_hist.append(val_loss)

        # —— scheduler(s) ——
        (warmup if epoch < WARMUP_EPOCHS else scheduler).step()

        # —— minimal console/TB logging ——
        if epoch % log_every == 0:
            logger.info(
                f"[{tag}] ep {epoch:>3d} | train {train_loss:.4e} | "
                f"val {val_loss:.4e} | lr {optimizer.param_groups[0]['lr']:.2e}"
            )

        writer.add_scalars(
            "loss", {"train": train_loss, "val": val_loss}, epoch)

        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        with torch.no_grad():
            _grad_history.append(
                torch.tensor([p.grad.norm().item()
                              if p.grad is not None else 0.
                              for p in model.parameters()])
            )
        ### NEW – snapshot the activations captured by hooks
        acts_snapshot = list(_latest_acts)   # shallow copy
        _latest_acts.clear()                 # free the tensors

        # —— heavier TB logging ——
        # if epoch % viz_every == 0:
        _tensorboard_viz(writer, model, epoch, dummy_x,
                        all_pred, all_y, where=pred_index_for_gt)

        _tb_epoch_metrics(writer, epoch, all_pred, all_y, model, where=pred_index_for_gt, tag=tag)

        phi_pred = torch.atan2(all_pred[:,pred_index_for_gt[1]], all_pred[:,pred_index_for_gt[2]])
        phi_true = torch.atan2(all_y   [:,1], all_y   [:,2])
        dt = (phi_pred - phi_true).cpu().numpy()

        writer.add_histogram("phi/angle_error", dt, epoch)

        for n,p in model.named_parameters():
            if p.grad is not None:
                ratio = (p.grad.norm()/p.data.norm()).item()
                writer.add_scalar(f"grad2weight/{n}", ratio, epoch)

        with torch.no_grad():
            log_sigma = model.head.bias[1::2]      # length-4 tensor
            for k, name in enumerate(("A", "t", "f0", "gbw")):
                writer.add_scalar(f"log_sigma/{name}", log_sigma[k], epoch)

        with torch.no_grad():
            last = model.head.weight      # (out_dim, hidden)
            gnorm = last.grad.detach().norm(p=2, dim=1).cpu()
            gnorm_dict = {f"grad_norm/{name}": gnorm[k] for k, name in enumerate(field_names[:pred.shape[1]])}
            writer.add_scalars("grad_norm", gnorm_dict, epoch)

        if hasattr(model, "log_vars"):
            writer.add_scalars("log_vars",
                {f"lv_{i}": v for i,v in enumerate(model.log_vars.cpu())},
                epoch)

        # ── extra diagnostics ───────────────────────────────────────────
        tb_grad_flow(writer, epoch)
        tb_logit_histograms(writer, epoch, all_pred)
        tb_activation_histograms(writer, epoch, acts_snapshot)
        tb_residual_stats(writer, epoch, all_pred, all_y)

        # —— checkpoint every 10 ep ——
        if epoch and epoch % 10 == 0:
            ckpt_path = MODEL_DIR / f"{ckpt_stem}_{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)

    # —— clean-up ——
    hook_handle.remove()
    for h in act_handles:        # remove act hooks
        h.remove()
    writer.close()
    logger.info(f"[{tag}] Training complete")
    run.finish()
    return train_hist, val_hist

# ═════════════ TensorBoard helper functions (graph / hists / figs) ═══════════


def tb_add_graph_once(writer: SummaryWriter,
                      model: nn.Module,
                      example_input: torch.Tensor):
    """
    Try to export the model graph exactly once.
    Complex ops sometimes break the tracer, so we swallow exceptions.
    """
    if getattr(writer, "_graph_logged", False):
        return
    try:
        writer.add_graph(model, example_input, use_strict_trace=False)
    except Exception as err:
        print(f"[TensorBoard] graph export failed – skipping ({err})")
    finally:
        writer._graph_logged = True


def tb_weight_and_grad_histograms(writer, model, epoch):
    """
    |W| and |∇W| histograms for every parameter.
    Magnitude is logged for complex tensors to placate TensorBoard.
    """
    for name, p in model.named_parameters():
        data = p.data.detach()
        hist_data = torch.abs(data) if data.is_complex() else data
        if not hist_data.isnan().all():
            writer.add_histogram(f"weights/{name}", hist_data.cpu(), epoch)

        if p.grad is not None:
            g = p.grad.detach()
            hist_grad = torch.abs(g) if g.is_complex() else g
            writer.add_histogram(f"grads/{name}", hist_grad.cpu(), epoch)


def tb_project_embedding(writer, model, dummy_x, epoch):
    """
    Push penultimate feature vectors to TensorBoard projector.
    If complex, we concatenate (real, imag) → purely real table.
    """
    try:
        with torch.no_grad():
            z = dummy_x
            for layer in model.conv_layers:
                z = layer(z)
            z = model.gap(z).flatten(1, 2)       # (B, C×pool)
            if torch.is_complex(z):
                z = torch.cat((z.real, z.imag), dim=1)
        writer.add_embedding(z.cpu(), global_step=epoch,
                         tag="embedding")
    except Exception as err:
        print(f"[TensorBoard] embedding export failed – skipping ({err})")


def tb_example_spectra(writer, epoch, train_ex, val_ex, real_ex):
    """
    3-panel log-log spectra: train | val | raw-real for quick eyeballing.
    """
    with torch.no_grad():
        fig, ax = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
        for i, (dat, ttl) in enumerate(zip((train_ex, val_ex, real_ex),
                                           ("train", "val", "real"))):
            ax[i].loglog(np.abs(dat.cpu().numpy()))
            ax[i].set_title(ttl)
            ax[i].set_xlabel("freq-bin")
            ax[i].grid(True, which="both")
        writer.add_figure("spectra", fig, epoch, close=True)


def tb_scatter_preds(writer, epoch,
                     y_true, y_pred,
                     where=pred_index_for_gt,
                     field_names=field_names,
                     gridsize=60):
    """
    2-D **density** (hexbin) of target vs prediction for every output dim.
    Colour = log₁₀(N samples per hex); diagonal shows y=x.
    """
    for k, name in enumerate(field_names[:len(where)]):
        yt = y_true[:, k]
        yp = y_pred[:, where[k]]
        fig, ax = plt.subplots(figsize=(3, 3))
        hb = ax.hexbin(yt, yp,
                       gridsize=gridsize, bins='log', mincnt=1)
        fig.colorbar(hb, ax=ax, label='log₁₀(count)')

        # diagonal reference
        lims = np.percentile(np.concatenate([yt, yp]),
                             [1, 99])
        ax.plot(lims, lims, 'k--', lw=.8)

        ax.set_xlabel("target")
        ax.set_ylabel("pred")
        ax.set_title(field_names[k])
        ax.set_aspect("equal", "box")

        writer.add_figure(f"pred_vs_target/{name}",
                          fig, epoch, close=True)
        plt.close(fig)


def tb_multiline_chart(writer: SummaryWriter,
                       chart_name: str,
                       tag_pairs: list[tuple[str, float]],
                       step: int) -> None:
    """
    • Writes each value with add_scalar(tag, …)
    • Registers a Custom Scalars > Multiline chart so TensorBoard plots them
      together.  The layout is written only once per writer.
    """
    # 1. log the individual scalars
    for tag, value in tag_pairs:
        writer.add_scalar(tag, float(value), step)

    # 2. register (once) a multiline chart that overlays those tags
    layout_flag = f"_layout_done_{chart_name.replace('/', '_')}"
    if getattr(writer, layout_flag, False):
        return

    layout = {
        chart_name.split('/')[0]:  # group in TB sidebar
        {chart_name.split('/')[-1]:
         ['Multiline', [t for t, _ in tag_pairs]]}
    }
    writer.add_custom_scalars(layout)
    setattr(writer, layout_flag, True)


def _tb_epoch_metrics(writer, epoch, pred, tgt, model=None, tag="epoch_metrics",
                      where=pred_index_for_gt,
                      field_names=field_names):
    """
    Log bias, variance, MAE, R² *and* last-layer gradient norms.
    """
    err  = pred[:, where] - tgt
    mse  = (err**2).mean(0)
    mae  = err.abs().mean(0)
    r2   = 1.0 - ((err**2).sum(0) / ((tgt - tgt.mean(0))**2).sum(0))
    names  = field_names[:len(where)]


    writer.add_scalars("metrics/MSE", dict(zip(names, mse)), epoch)
    writer.add_scalars("metrics/MAE", dict(zip(names, mae)), epoch)
    writer.add_scalars("metrics/R2",  dict(zip(names, r2)), epoch)

def tb_log_sigmas(writer: SummaryWriter,
                  model: nn.Module,
                  epoch: int,
                  names: tuple[str, ...] = ("A", "t", "f0", "gbw")) -> None:
    """
    Write the four learned log-σ values (indices 1,3,5,7 of fc2.bias)
    to TensorBoard under log_sigma/<name>.
    """
    with torch.no_grad():
        log_sigma = model.head.bias[1::2]            # (4,)
        for k, name in enumerate(names):
            writer.add_scalar(f"log_sigma/{name}", log_sigma[k], epoch)


def tb_grad_flow(writer, epoch):
    """Log a layer×epoch heat-map of gradient L2 norms.
    If a square is black, the gradient is zero.
    If a square is white, the gradient is NaN.
    If a square is grey, the gradient is non-zero."""

    if len(_grad_history) < 2:      # need ≥2 epochs for a decent image
        return
    g = torch.stack(_grad_history).T   # [layers, epochs]
    g_log  = torch.log10(g + 1e-10)
    g_norm = (g_log - g_log.min()) / (g_log.max() - g_log.min() + 1e-9)
    writer.add_image("grad_flow", g_norm.unsqueeze(0), epoch, dataformats="CHW")

def tb_logit_histograms(writer,
                        epoch: int,
                        pred: torch.Tensor,
                        names: tuple[str, ...] = field_names14) -> None:
    """
    Histogram of every output logit in its native order:
        μ_modA, logσ_modA, μ_Δt, logσ_Δt, …, sinφ, cosφ
    """
    p = pred.detach().cpu()          # (N, 14)
    for k, name in enumerate(names[:p.shape[1]]):
        writer.add_histogram(f"logits/{name}", p[:, k], epoch)

def tb_activation_histograms(writer, epoch, acts):
    """acts = list[(name, tensor)] captured by forward hooks."""
    for n, t in acts:
        writer.add_histogram(f"acts/{n}", t.detach().cpu(), epoch)

def tb_residual_stats(writer, epoch, pred, tgt, where=pred_index_for_gt):
    """
    Log residual distributions as histograms (1 tag per target).
    """
    for k, j in enumerate(where):
        residual = (pred[:, j] - tgt[:, k]).detach().cpu()
        writer.add_histogram(f"residuals/{field_names[k]}", residual, epoch)

def tb_polar_phi(writer: SummaryWriter,
                 epoch:  int,
                 preds:  np.ndarray,
                 tgts:   np.ndarray,
                 tag:    str = "phi/polar",) -> None:
    """
    Polar scatter of (cosφ, sinφ) : true vs. predicted.
    True samples at r=1, predicted at r=1.05 so they don’t hide each other.
    """
    phi_true = np.arctan2(tgts[:, 1], tgts[:, 2])
    phi_pred = np.arctan2(preds[:, pred_index_for_gt[1]],   # sin φ
                      preds[:, pred_index_for_gt[2]])   # cos φ

    fig = plt.figure(figsize=(4, 4))
    ax  = fig.add_subplot(111, projection='polar')
    ax.scatter(phi_true, np.ones_like(phi_true),
               s=2, alpha=.6, label="true")
    ax.scatter(phi_pred, np.ones_like(phi_pred) * 1.05,
               s=2, alpha=.6, label="pred")
    ax.set_rticks([])                    # hide radius ticks
    ax.set_title("cos φ / sin φ – polar", fontsize=10)
    ax.legend(loc="upper right", markerscale=4, frameon=False)

    writer.add_figure(tag, fig, epoch, close=True)


def tb_pred_vs_target_dist(writer: SummaryWriter,
                           epoch:  int,
                           preds:  np.ndarray,
                           tgts:   np.ndarray,
                           names:  tuple[str, ...] = field_names,
                           bins:   int = 60,
                           where:  list[int] = pred_index_for_gt) -> None:
    """
    Histogram overlay:  p(target)  vs  p(prediction)  (density-normalised).
    """
    for k, name in enumerate(field_names[:len(where)]):
        yt = tgts[:, k]
        yp = preds[:, where[k]]     # <— use mapping
        fig, ax = plt.subplots(figsize=(3, 3))
        ax.hist(yt, bins=bins, density=True,
                alpha=.6, label="target")
        ax.hist(yp, bins=bins, density=True,
                alpha=.6, label="pred")
        ax.set_title(name)
        ax.legend()
        writer.add_figure(f"dist/{name}", fig, epoch, close=True)

# ─────────────────── forward-hook: sample outputs & NaN guard ───────────────


def _register_output_hook(model, *, print_every: int = 200):
    """
    Attach a hook to model.output_layer that:
      • prints the first few predictions every <print_every> steps *during train*
      • scans parameter grads for NaNs
    Returns a RemovableHandle (call .remove() when done).
    """
    step = {'i': 0}    # mutable counter in closure

    def _hook(_, __, out):
        if model.training and step['i'] % print_every == 0:
            with torch.no_grad():
                print(f"[hook] step {step['i']} | "
                      f"out[:4] = {out[:4].detach().cpu().numpy()}")
        # NaN grad checker
        for name, p in model.named_parameters():
            if p.grad is not None and torch.isnan(p.grad).any():
                print(f"[hook] NaN in {name} grad\n{p.grad}")
        step['i'] += 1

    return model.head.register_forward_hook(_hook)

def _register_activation_hooks(model):
    """
    Returns: list[handle] – call .remove() on each before exit.
    Captures activations after every _CVBlock (or Conv1d) in the backbone.
    """
    handles = []
    for i, m in enumerate(model.modules()):
        if isinstance(m, _CVBlock):          # your own block class
            def _make_hook(name):
                def _hook(_, __, out):
                    if _capture_acts:
                        _latest_acts.append((name, out.detach()))
                return _hook
            handles.append(m.register_forward_hook(_make_hook(f"block{i}")))
    return handles

# ════════════════════ train / evaluate each architecture ════════════════════
if train_complex:
    print("\n─── Training COMPLEX-valued CNN ───────────────────────────")
    _ = train_generic(
        # model_fn=lambda: GlitchNetRes(
        #     conv_type="wfm",      # "conv" if you want the plain version
        #     use_attn=True,
        #     dropout=0.10,         # already applied inside each block
        #     ).to(device),
        # model_fn= lambda: GlitchNetCV(
        #     stem_width=32,
        #     width_mult=2
        # ).to(device),
        model_fn = lambda: GlitchRegressor().to(device),
        tag="complex_" + time.ctime().replace(" ", "_").replace(":", "_"),
        out_dim=14,
        train_loader=train_loader,
        test_loader=test_loader
    )
    memsum()

if test_complex:
    # Quick sanity check: forward pass on a single mini-batch
    print("\n─── Evaluating COMPLEX model ──────────────────────────────")
    # model = ComplexValuedNN(dropout=0.10,
    #                         conv_filters=conv_filters,
    #                         n_conv_layers=n_conv_layers)
    # model = GlitchNetRes(
    # conv_type="wfm",      # "conv" if you want the plain version
    # use_attn=True,
    # dropout=0.10,         # already applied inside each block
    # ).to(device)
    # model = GlitchNetCV(
    #     stem_width=32,
    #     width_mult=2
    # ).to(device)
    model = GlitchRegressor().to(device)

    ep = load_latest_ckpt(model, "complex_model")
    model.eval().to(device)

    xb, yb = next(iter(test_loader))
    xb = torch.stack(xb).to(device)
    with torch.no_grad():
        out = model(xb)[:10].cpu().numpy()
    tgt = torch.stack(yb)[:10].numpy()

    for i in range(10):
        print(f"pred {out[i]}  |  tgt {tgt[i]}")

# ────────────────────────────────────────────────────────────────────────────
if train_real:
    print("\n─── Training REAL (freq-domain) CNN ──────────────────────")
    _ = train_generic(
        model_fn=lambda: RealValuedNN(dropout=0.15),
        tag="real_fd",
        out_dim=2,             # this head predicts only 2 dims
        train_loader=train_loader,
        test_loader=test_loader
    )

if train_real_td:
    print("\n─── Training REAL (time-domain) CNN/LSTM ────────────────")
    # Build *time-domain* datasets fresh so augmentations differ.
    td_train = GlitchDataset(DATADIR, ifos, keys, glitches, dists,
                             NTRAIN, NTEST, device,
                             noise=True, aug_phase=True, aug_time=True,
                             split="train", outtype="real")
    td_test = GlitchDataset(DATADIR, ifos, keys, glitches, dists,
                            NTRAIN, NTEST, device,
                            noise=True, aug_phase=True, aug_time=True,
                            split="test", outtype="real",
                            train_data=td_train)

    td_train_loader = DataLoader(td_train, batch_size=BATCH, shuffle=True,
                                 **dl_kwargs)
    td_test_loader = DataLoader(td_test,  batch_size=BATCH, shuffle=True,
                                **dl_kwargs)

    _ = train_generic(
        model_fn=RealValuedNNtd,
        tag="real_td",
        out_dim=2,
        train_loader=td_train_loader,
        test_loader=td_test_loader
    )
