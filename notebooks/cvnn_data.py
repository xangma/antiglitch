"""
Fast, safer data utilities for antiglitch-CVNN
=============================================

This file rewrites the original antiglitch data helpers with a focus on:
  1. *Faster catalogue construction*  (no PSD loading inside the first pass)
  2. *Lower RAM usage*                (mmap PSDs, cache lazily)
  3. *Deterministic augmentations*    (one RNG per dataset instance)
  4. *Cleaner targets*                (always keep raw values on disk, apply
                                       augmentation + scaling only at load time)

Nothing in the public `GlitchDataset` API changes, so callers do not have to
update their training code.
"""

# ────────────────────────────── future proofing ─────────────────────────────
# allow `-> list[int]` etc. on Py 3.8+
from __future__ import annotations
import sys
sys.path.append("/Users/xangm/OneDrive/repos/antiglitch")
from antiglitch.utils import downsample_invasd, extract_glitch, to_fd
import antiglitch

# ────────────────────── standard / third-party imports ──────────────────────
import os
import re
import json
import sys
import math
from collections import defaultdict
from functools import partial
from typing import Dict, List, Sequence, Tuple

import numpy as np                       # heavy-lifting numeric work
from numpy import median                 # convenience binding
from scipy.stats import median_abs_deviation as mad

from torch.utils.tensorboard import SummaryWriter

import torch                              # final consumer is PyTorch

from pathlib import Path

# ─────────────────── precise FFT aliases with "ortho" norm ──────────────────
# All forward/backward FFTs *must* use the same normalisation to stay
# energy-preserving.  We bind the functions up-front with norm="ortho".
from numpy.fft import rfft as _rfft, irfft as _irfft
rfft = partial(_rfft,  norm="ortho")
irfft = partial(_irfft, norm="ortho")

# honour global log level from the trainer; fallback to INFO
import logging, os, collections

DEFAULT_LVL = os.getenv("CVNN_LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=DEFAULT_LVL,
    format="%(asctime)s | %(levelname)-7s | %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    force=False           # do not override trainer’s config if already set
)
logger = logging.getLogger("cvnn.data")

# tiny helper → first call logs INFO, repeats drop to DEBUG
_call_counter = collections.Counter()
def log_once(tag: str, msg: str) -> None:
    """
    Log *msg* first time at INFO, subsequent times at DEBUG with a counter.
    Lets us detect accidental re-runs without flooding stdout.
    """
    _call_counter[tag] += 1
    n = _call_counter[tag]
    if n == 1:
        logger.info(msg)
    else:
        logger.debug("%s (repeat #%d)", msg, n)
# ────────────────── one process-wide RNG for reproducibility ────────────────
_global_rng = np.random.default_rng(0)     # used only for quick helpers

# ─────────────────────── import antiglitch (repo local) ─────────────────────
# The project lives next to this script; we append its root to `sys.path`
# to import the helper functions we still rely on.
sys.path.append("/Users/xangm/OneDrive/repos/antiglitch")

label_map = {                        # human → short slug used in filenames
    "Blip_Low_Frequency": "lowblip",
    "Blip":               "blip",
    "Koi_Fish":           "koi",
    "Tomte":              "tomte",
}
_IDX2KEY = sorted(label_map.values())
flds = ["f0", "f0_sd", "gbw", "gbw_sd",
        "amp_r", "amp_r_sd", "amp_i", "amp_i_sd",
        "time", "time_sd", "snr", "residual", "num"]


# ───── frequency bin centres (exactly the same 513 bins as downsample_invasd) ────
#
# The antiglitch down-sampler reduces the 0‒4096 Hz positive-frequency half of a
# 8192 Hz-sampled FFT (n = 4096) to 513 log-spaced bins.  We replicate that
# mapping once here so time-shift phase ramps are exact.
#
# NOTE: If you ever change the down-sampler, *update this table as well*.
try:
    _FREQS_DS = np.loadtxt(Path(__file__).with_name("freq_bins_513.txt"))
except FileNotFoundError:
    # fall back to an approximate linear grid (≤ 0.3 % phase error at 100 ms)
    _FREQS_DS = np.linspace(0.0, 4096.0, 513, dtype=np.float64)

# ─────────────────────────── helper for stability ───────────────────────────


def _cart2polar_amp(y: np.ndarray) -> np.ndarray:
    """Convert in-place (*, 2) cartesian (Re, Im) → (|A|, ∠A)."""
    log_once("cart2polar", "Transform: (amp_r, amp_i) → (|A|, ∠A)")

    amp = y[..., 0] + 1j * y[..., 1]
    mag = np.abs(amp)
    mag[np.isnan(mag)] = 1e-6
    phase = np.angle(amp)
    phase[np.isnan(phase)] = 0.0

    y[..., 0] = mag
    y[..., 1] = phase
    return y

# ────────────────────────── lightweight snippet shell ───────────────────────


class DummySnip:
    """
    Stand-in for `antiglitch.SnippetNormed`.

    The original constructor touches disk to read PSDs; we can’t afford that
    while scanning thousands of files.  Instead we record the `.inf` dict and
    patch it later, only if/when the caller actually dereferences the snippet.
    """

    def __init__(self) -> None:
        self.inf: dict = {}                 # will be filled from JSON PE file



# ──────────────────────── catalogue construction utils ──────────────────────


def _quick_invasd(psd: np.ndarray) -> np.ndarray:
    """
    Down-sampled  1/√PSD  for whitening.

    • Clamps the PSD below 1e-12 to prevent divide-by-zero.
    • First 10 bins are nulled (don’t trust the DC region).
    """
    psd = np.clip(psd, 1e-12, None)
    inv_full = (4096.0 * psd) ** -0.5       # inverse ASD at native rate
    inv_full[:10] = 0.0
    return downsample_invasd(inv_full)  # → shape (513,)

# ─────────────────── main file system walk / metadata build ─────────────────


def get_data(datadir: str):
    """
    Parse every  <IFO>-<label>-NNNN.npz  in *datadir* and return:

        distributions : nested dict with per-IFO / per-class parameter lists
        glitches      : nested dict with cached metadata for each snippet
        ifos          : sorted list of IFO ids (e.g. ['H1','L1'])
        keys          : sorted list of glitch keywords (['blip','koi',...])

    Heavy PSD ND-arrays are **not** loaded here – only filenames are stored.
    """
    # 1) Gather available (ifo, key, num) triples ---------------------------
    name_re = re.compile(r"([A-Z0-9]+)-([a-z]+)-(\d+)\.npz")
    ifos, keys = set(), set()
    for fn in os.listdir(datadir):
        m = name_re.match(fn)
        if m:
            ifos.add(m.group(1))
            keys.add(m.group(2))
    ifos, keys = sorted(ifos), sorted(keys)

    # group numbers under each IFO/class for quick access
    index: Dict[str, Dict[str, List[int]]] = {
        i: {k: [] for k in keys} for i in ifos}
    for fn in os.listdir(datadir):
        m = name_re.match(fn)
        if m:
            ifo, key, num = m.groups()
            index[ifo][key].append(int(num))

    # 2) Attach parameter-estimation (PE) results ---------------------------
    with open(os.path.join(datadir, "all_PE_v3.json")) as fh:
        js = json.load(fh)



    # 3) Build main glitch dictionary with lazy PSD refs --------------------
    glitches: Dict[str, Dict[str, Dict[int, dict]]
                   ] = defaultdict(lambda: defaultdict(dict))
    for ifo in ifos:
        for key in keys:
            for num in index[ifo][key]:
                path = os.path.join(datadir, f"{ifo}-{key}-{num:04d}.npz")
                # pre-compute down-sampled inverse ASD for later whitening
                with np.load(path, mmap_mode="r") as npz:
                    invasd_ds = _quick_invasd(npz["psd"])
                glitches[ifo][key][num] = {
                    "npz_path": path,              # defer big arrays
                    "invasd": invasd_ds.astype(np.float32),
                    "snip": DummySnip(),       # PE fields patched next
                }

    # 4) Fill `.inf` with PE parameters (ignore files missing in JSON) ------
    for idx in map(str, range(len(js["num"]))):
        try:
            ifo = js["ifo"][idx]
            key = label_map[js["ml_label"][idx]]
            num = js["num"][idx]
            info = glitches[ifo][key][num]["snip"].inf
            for f in flds:
                if f in js:
                    info[f] = js[f][idx]
        except KeyError:
            # entry may be absent if a file was deleted / renamed – benign
            pass

    # 5) Build per-class empirical distributions for augmentation statistics
    dist: Dict[str, Dict[str, Dict[str, List[float]]]] = {
        i: {k: defaultdict(list) for k in keys} for i in ifos}


    for ifo in glitches:
        for key in glitches[ifo]:
            for num, rec in glitches[ifo][key].items():
                for f in ("f0", "gbw", "amp_r", "amp_i", "time", "snr", "residual"):
                    # raw value ─ always present
                    dist[ifo][key][f].append(rec["snip"].inf[f])
                    # sd value ─ may be missing → use 0.0 placeholder
                    dist[ifo][key][f"{f}_sd"].append(
                        rec["snip"].inf.get(f"{f}_sd", 0.0))


    dist_amp = {k: [] for k in keys}
    for ifo in glitches:
        for key in glitches[ifo]:
            for num, rec in glitches[ifo][key].items():
                a = rec["snip"].inf
                dist_amp[key].append(np.hypot(a["amp_r"], a["amp_i"]))
    for key in keys:                                 # convert to ndarray for speed
        dist_amp[key] = np.asarray(dist_amp[key], dtype=np.float32)
    return dist, glitches, ifos, keys, dist_amp

# ─────────────────────────── augmentation primitives ────────────────────────


def amp_rescale(X: np.ndarray,
                Y: np.ndarray,
                rng: np.random.Generator,
                key,
                amp_pools: Dict[str, np.ndarray],
                clip: Tuple[float, float] = (1e-3, 5e2)):
    """
    Rescale a glitch to a *new* |A| drawn from the empirical pool of its class.

    `key` can be
        • scalar str  ('blip', 'koi', …)
        • scalar int  (0…3)
        • 1-D ndarray of str  or int  (vectorised batch case)
    """
    # ── 1.  normalise key(s) to class strings ─────────────────────────────
    if isinstance(key, np.ndarray):
        if np.issubdtype(key.dtype, np.integer):
            key_strs = np.take(_IDX2KEY, key.astype(int)
                               )    # vectorised int→str
        else:
            key_strs = key                                   # already strings
        # one random amplitude per sample  → 1-D array (batch,)
        Anew = np.asarray([rng.choice(amp_pools[ks]) for ks in key_strs],
                          dtype=np.float32)
    else:
        # scalar
        if isinstance(key, (int, np.integer)):
            key_str = _IDX2KEY[int(key)]
        else:
            key_str = key
        Anew = np.float32(rng.choice(amp_pools[key_str]))

    # ensure shapes are aligned with Y[...,0]
    Aold = np.abs(Y[..., 0] + 1j * Y[..., 1]) + 1e-12       # avoid /0
    scale = np.clip(Anew / Aold, *clip)                      # safe range

    # broadcast over frequency bins & (amp_r, amp_i)
    X = X * scale[..., None]
    Y[..., :2] *= scale[..., None]
    return X, Y


def noise(x: np.ndarray, rng: np.random.Generator,
                    p_sig_ref: float | None=None,
                    snr_range=(10.0, 25.0)):
    """
    Add circular-complex AWGN such that the *power* SNR is uniform in
    `snr_range` [dB].
    """
    snr_db = rng.uniform(*snr_range)
    # use pre-rescale power if supplied; else fall back to current x
    p_sig = float(p_sig_ref) if p_sig_ref is not None else np.mean(np.abs(x) ** 2)
    sigma = np.sqrt(p_sig / (2.0 * 10 ** (snr_db / 10.0)))
    n = (rng.normal(scale=sigma, size=x.shape) +
         1j * rng.normal(scale=sigma, size=x.shape))
    return x + n


def time_shift(x: np.ndarray, rng: np.random.Generator,
               dt_max: float = 0.01) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a random global time-shift  Δt ∈ [−dt_max, +dt_max] s via the exact
    phase ramp  x·e^(-j2πfΔt).

    Returns:  (shifted x, Δt)  so that labels can be updated downstream.
    """
    dt = rng.uniform(-dt_max, dt_max, size=x.shape[:-1])
    rot = np.exp(-1j * 2.0 * np.pi * _FREQS_DS * dt[..., None])
    return x * rot, dt.astype(np.float64)


def _draw_phase(rng: np.random.Generator,
                shape,
                ifo: str | np.ndarray | None = None) -> np.ndarray:
    """
    Draw φ from an empirical mixture:
        • H1/L1 → {0, π}         0.5 / 0.5
        • V1    → {π/2, −π/2}    0.5 / 0.5
        • else  → {0, π/2, π, −π/2}  0.4 / 0.1 / 0.4 / 0.1

    Works for a single *ifo* **or** a vector of IFO codes.
    The returned array has shape = *shape*.
    """
    κ = 5.0                                         # concentration parameter

    # ── scalar case ──────────────────────────────────────────────────────
    if np.isscalar(ifo) or ifo is None:
        if ifo in ("H1", "L1"):
            centres, probs = (np.array([0.0, np.pi]),
                              np.array([0.5, 0.5]))
        elif ifo == "V1":
            centres, probs = (np.array([0.5*np.pi, -0.5*np.pi]),
                              np.array([0.5, 0.5]))
        else:
            centres, probs = (np.array([0.0, 0.5*np.pi, np.pi, -0.5*np.pi]),
                              np.array([0.4, 0.1, 0.4, 0.1]))

        idx = rng.choice(len(centres), size=shape, p=probs)
        return rng.vonmises(centres[idx], κ)

    # ── vector case ──────────────────────────────────────────────────────
    ifo_arr = np.asarray(ifo)
    if ifo_arr.shape != shape:
        # broadcast the IFO labels to the requested output shape
        ifo_arr = np.broadcast_to(ifo_arr, shape)

    out = np.empty(shape, dtype=np.float64)

    # H1 / L1
    mask = np.isin(ifo_arr, ("H1", "L1"))
    n = mask.sum()
    if n:
        centres = np.array([0.0, np.pi])
        probs = np.array([0.5, 0.5])
        idx = rng.choice(len(centres), size=n, p=probs)
        out[mask] = rng.vonmises(centres[idx], κ)

    # V1
    mask = ifo_arr == "V1"
    n = mask.sum()
    if n:
        centres = np.array([0.5*np.pi, -0.5*np.pi])
        probs = np.array([0.5, 0.5])
        idx = rng.choice(len(centres), size=n, p=probs)
        out[mask] = rng.vonmises(centres[idx], κ)

    # everything else
    mask = ~(np.isin(ifo_arr, ("H1", "L1", "V1")))
    n = mask.sum()
    if n:
        centres = np.array([0.0, 0.5*np.pi, np.pi, -0.5*np.pi])
        probs = np.array([0.4, 0.1, 0.4, 0.1])
        idx = rng.choice(len(centres), size=n, p=probs)
        out[mask] = rng.vonmises(centres[idx], κ)

    return out

def augsample(X, Y, rng, k, dist_amp, ifo=None,
              noise_ok=True, phase_ok=True, time_ok=True,
              tilt_ok=False, rescale_amp=True):
    """
    Jointly augment (*X*, *Y*).  All options now on by default.
    """
    X = X.copy()
    Y = Y.copy()


    if rescale_amp:
        p_sig_orig = np.mean(np.abs(X) ** 2)        # scalar per sample
        X, Y = amp_rescale(X, Y, rng, k, dist_amp)


    if noise_ok:
        X = noise(X, rng, p_sig_ref=p_sig_orig if rescale_amp else None)

    # ─── global phase rotation ─────────────────────────────────────────────
    if phase_ok:
        phi = _draw_phase(rng, X.shape[:-1], ifo=ifo)
        rot = np.exp(1j * phi)[..., None]
        X   *= rot
        amp = (Y[..., 0] + 1j * Y[..., 1]) * rot.squeeze(-1)
        Y[..., 0], Y[..., 1] = amp.real, amp.imag

    # ─── global occurrence-time shift ──────────────────────────────────────
    if time_ok:
        X, dt = time_shift(X, rng)                       # uses true freqs
        Y[..., 2] += dt

    if tilt_ok:
        X = psd_tilt(X, rng)

    return X, Y


def phase_rotate(x, rng):
    phi = rng.uniform(0.0, 2*np.pi, size=x.shape[:-1])
    rot = np.exp(1j*phi)[..., None]
    return x*rot, phi.astype(np.float32)


def psd_tilt(x, rng, alpha_std=0.05):
    freqs = np.linspace(0, 4096, x.shape[-1], dtype=np.float32)
    alpha = rng.normal(scale=alpha_std, size=x.shape[:-1])
    tilt  = (freqs / 200.)**alpha[..., None]
    return x * tilt

# ────────────────── internal helpers for cache / FD access ──────────────────


def _draw_ids(rng, ifos, glitches, batch):
    """
    Uniformly sample *batch* snippet identifiers across **all classes**.

    This deliberately ignores class imbalance – the NN will ingest balanced
    mini-batches downstream via its own sampler.
    """
    pool = [(ifo, key, num)
            for ifo in ifos
            for key, sub in glitches[ifo].items()
            for num in sub]
    idx = rng.choice(len(pool), size=batch, replace=True)
    return [pool[i] for i in idx]


def _ensure_fd(rec: dict) -> np.ndarray:
    """
    Return the cached frequency-domain snippet for *rec*.

    Loads once with memory-mapping and stores under `"fd"` to avoid I/O next time.
    """
    if "fd" in rec:
        return rec["fd"]
    with np.load(rec["npz_path"], mmap_mode="r") as npz:
        inv, whts, _ = extract_glitch(npz)          # heavy work lives here
    fd = to_fd(whts).astype(np.complex64)           # (513,) complex64
    rec.update({"fd": fd, "invasd": inv})
    return fd

def _polar_to_sincos(Y_polar: np.ndarray) -> np.ndarray:


    if Y_polar.shape[-1] < 5:
        raise ValueError("need at least (|A|,φ,t,f0,gbw) columns")

    log_once("polar→sincos", "Transform: φ → (sin φ, cos φ)")

    amp   = Y_polar[..., 0]
    phi   = Y_polar[..., 1]
    sinφ  = np.sin(phi)
    cosφ  = np.cos(phi)
    rest  = Y_polar[..., 2:]
    return np.concatenate([
        amp[..., None],
        sinφ[..., None],
        cosφ[..., None],
        rest
    ], axis=-1)

# ───────────────────────────── Dataset wrapper ──────────────────────────────


class GlitchDataset(torch.utils.data.Dataset):
    """
    PyTorch-ready dataset with on-the-fly augmentations *inside* `__getitem__`.

    Parameters
    ----------
    datadir      : directory that contains <IFO>-<label>-NNNN.npz and JSON PE
    ifos, glitches, distributions : objects returned by `get_data()`
    tr_size, te_size : how many *unique* snippets to cache for each split
    device       : torch.device where tensors will be moved
    noise        : toggle AWGN augmentation
    aug_phase    : toggle phase-flip augmentation
    aug_time     : toggle time-shift augmentation
    split        : "train" or "test"
    outtype      : "complex" (default) → FD complex64
                   "time"             → TD float32 via irfft
    seed         : per-dataset RNG seed (independent of global)
    """

    def __init__(self,
                 datadir: str,
                 ifos: Sequence[str],
                 glitches,
                 distributions,
                 dist_amp,
                 tr_size: int,
                 te_size: int,
                 device,
                 noise=True, aug_phase=True, aug_time=True,
                 split="train",
                 outtype="complex",
                 cache_prefix="antiglitch_cvnn_dataset_clean",
                 seed=0):
        # ---------------- basic bookkeeping --------------------------------
        assert split in ("train", "test")
        self.rng = np.random.default_rng(seed)
        self.device = device
        self.noise, self.ap, self.at = noise, aug_phase, aug_time
        self.out = outtype
        self.dist_amp = dist_amp
        tag = f"{outtype}_seed{seed}"
        self.tc = os.path.join(datadir,
                               f"{cache_prefix}_{tag}_train{tr_size}.npz")
        self.sc = os.path.join(datadir,
                               f"{cache_prefix}_{tag}_test{te_size}.npz")

        self.ifos = ifos
        self.glitches = glitches

        # ------------- build or load train/test NPZ caches -----------------
        self._ensure_cache(self.tc, tr_size, label_map)   # create if missing
        self._load_train_arrays()              # we need them for stats

        # `X_raw` / `Y_raw` are the arrays used by __getitem__ & batching
        self.X_raw = self.X_tr
        self.Y_raw = self.Y_tr
        self.size = len(self.X_raw)

        # compute or reuse scalers (depends on augment flags)
        self._ensure_scalers()

        if split == "test":
            self._ensure_cache(self.sc, te_size, label_map)

        fn = self.tc if split == "train" else self.sc
        tmp = np.load(fn)
        self.X_raw = tmp["x_arr"]
        self.Y_raw = tmp["y_arr"]
        self.k_raw = tmp["k_arr"]
        self.i_raw = tmp["i_arr"]
        self.size = len(self.X_raw)

        # ---- pre-compute scaled targets to avoid work in __getitem__ ------
        self.Y_scaled = self._scale_Y(self.Y_raw.copy())
        Y_demo = self._scale_Y(self.Y_tr.copy())
        self._tb_dump_target_hists(Y_demo)

    # ─────────────────────────── static helpers ────────────────────────────
    @staticmethod
    def _log_transform(Y: np.ndarray) -> np.ndarray:
        if not hasattr(GlitchDataset._log_transform, "_logged"):
            logger.info("Transform: log1p(|A|), log1p(f0), log1p(gbw)")
            GlitchDataset._log_transform._logged = True

        Y = Y.copy()
        Y[..., 0] = np.log1p(Y[..., 0])
        Y[..., 4] = np.log1p(Y[..., 4])
        Y[..., 5] = np.log1p(Y[..., 5])
        if Y.shape[-1] > 6:               # snr present
            Y[..., 6] = np.log1p(Y[..., 6])
            Y[..., 7] = np.log1p(Y[..., 7])
        return Y

    @staticmethod
    def _ilog_transform(Y: np.ndarray) -> np.ndarray:
        """Inverse of `_log_transform` (for inference pipeline)."""
        Y = Y.copy()
        Y[..., 0] = np.expm1(Y[..., 0])
        Y[..., 4] = np.expm1(Y[..., 4])
        Y[..., 5] = np.expm1(Y[..., 5])
        if Y.shape[-1] > 6:               # snr present
            Y[..., 6] = np.expm1(Y[..., 6])
            Y[..., 7] = np.expm1(Y[..., 7])
        return Y


    # ──────────────────────── TensorBoard helpers ─────────────────────────

    def _tb_dump_target_hists(self, Y_scaled):
        """One-off TensorBoard histograms of each target component."""
        wr = SummaryWriter(str(Path(self.tc).with_suffix('')))   # runs next to cache
        names = ("|A|", "sinφ", "cosφ", "t",
                         "f0", "gbw", "snr", "residual")
        for k, nm in enumerate(names):
            wr.add_histogram(f"dataset/{nm}", Y_scaled[..., k], global_step=0)
        wr.close()


    # ──────────────────────── cache constructors ───────────────────────────
    def _ensure_cache(self, fname: str, N: int, keys):
        """
        Build an NPZ cache with N randomly-drawn snippets if `fname` is absent.

        The cache stores:
            x_arr : (N, 513) complex64 raw FD spectra
            y_arr : (N,   5) float32 raw targets (cartesian amp)
        """
        if os.path.exists(fname):
            return

        ids = _draw_ids(self.rng, self.ifos, self.glitches, N)
        if len(ids) < N:
            raise RuntimeError(
                f"Requested {N} samples, but only {len(ids)} exist.")

        X = np.empty((N, 513), np.complex64)
        Y = np.empty((N,   7), np.float32)
        key_idx_map = {k: i for i, k in enumerate(sorted(list(label_map.values())))}

        K = np.empty(N, np.int16)              # class index
        I = np.empty(N, dtype='U2')            # IFO as 2-char string

        for i, (ifo, key, num) in enumerate(ids):
            rec = self.glitches[ifo][key][num]
            fd = _ensure_fd(rec)
            X[i] = fd
            inf = rec["snip"].inf
            Y[i] = (inf["amp_r"], inf["amp_i"], inf["time"],
                            inf["f0"], inf["gbw"],
                            inf["snr"], inf["residual"])
            K[i] = key_idx_map[key]
            I[i] = ifo





        # ── π-ambiguity removal ───────────────────────────────
        # (amp_r, amp_i) live in columns 0 and 1
        # amp = Y[:, 0] + 1j * Y[:, 1]

        # pick any equivalent criterion:
        #   • left half-plane  ↔  Re A < 0
        #   • or  |φ| > π/2
        # flip = amp.real < 0               # boolean mask
        # rotate by π:  (Re, Im) → (−Re, −Im)
        # Y[flip, :2] *= -1

        np.savez(fname, x_arr=X, y_arr=Y, k_arr=K, i_arr=I)
        logger.info("Built new cache %s  (N=%d)", os.path.basename(fname), N)

    def _load_train_arrays(self):
        """Load the *train* cache – used for scaler statistics generation."""
        t = np.load(self.tc)
        self.X_tr, self.Y_tr, self.K_tr, self.I_tr = t["x_arr"], t["y_arr"], t["k_arr"], t["i_arr"]


    # ----------------------- augmentation statistics ------------------------
    def _augmented_target_stats(self) -> List[Tuple[float, float]]:
        """
        Empirically estimate (μ, σ) of *augmented* target distribution.

        We simulate 100 k augmented pairs, then compute mean/std on each
        column.  Std is always plain *unbiased* std; no MAD or robust variant
        because outliers are intentionally part of the training targets.
        """
        samp = 100_000
        idx = self.rng.choice(len(self.X_tr), size=samp, replace=True)
        Xs = self.X_tr[idx]
        Ys = self.Y_tr[idx]
        Ks = self.K_tr[idx]
        Is = self.I_tr[idx]
        _, Y_aug = augsample(Xs, Ys, self.rng, Ks, self.dist_amp, ifo=Is,
                             noise_ok=self.noise,
                             phase_ok=self.ap,
                             time_ok=self.at,
                             tilt_ok=False)        # stats should match loader

        Y_aug = self._log_transform(
                    _polar_to_sincos(
                        _cart2polar_amp(Y_aug)))


        mu = Y_aug.mean(axis=0)
        sig = Y_aug.std(axis=0)
        sig[sig == 0] = 1.0
        return [(float(m), float(s)) for m, s in zip(mu, sig)]

    # ---------------------------- scalers -----------------------------------
    def _ensure_scalers(self):
        """
        Compute (or reuse) feature/target scalers and stash them in
        `self.scalers`.  The scalers are also persisted in the *train* cache
        so future runs skip this expensive pass.
        """
        def _needs_rebuild(s):
            tol = 1e-2
            amp_ok = abs(s["Y"][0][0]) < tol and abs(s["Y"][1][0]) < tol and abs(s["Y"][2][0]) < tol
            spread_ok = s["Y"][4][1] > 0.05 and s["Y"][5][1] > 0.05
            return not (amp_ok and spread_ok)

        with np.load(self.tc, allow_pickle=True) as t:
            if "scalers" in t.files and not _needs_rebuild(t["scalers"].item()):
                self.scalers = t["scalers"].item()
                logger.debug("Loaded scalers from existing cache – rebuild skipped")
                return

            X_full = t["x_arr"]
            Y_raw_full = t["y_arr"].copy()
            K_full = t["k_arr"]
            I_full = t["i_arr"]

        if self.out == "complex":
            log_once("scale_Xc",
                      "Input X: per-bin μ/σ for better morphology preservation")
            μr = X_full.real.mean(axis=0)                 # → (513,)
            σr = X_full.real.std(axis=0);  σr[σr == 0] = 1.0
            μi = X_full.imag.mean(axis=0)
            σi = X_full.imag.std(axis=0);  σi[σi == 0] = 1.0
            scalers = {"X_real": (μr.astype(np.float32),
                                   σr.astype(np.float32)),
                        "X_imag": (μi.astype(np.float32),
                                   σi.astype(np.float32)),
                        "Y": []}
        else:
            log_once("scale_Xtd", "Input X: using time-domain format (via irfft)")
            Xtd = irfft(X_full)
            μ, σ = Xtd.mean(), Xtd.std() or 1.0
            scalers = {"X_td": (μ, σ), "Y": []}

        # target stats (already log-scaled, sin/cos φ)
        Y_full = GlitchDataset._log_transform(
                _polar_to_sincos(
                _cart2polar_amp(Y_raw_full.copy())))
        for k in range(8):
            col = Y_full[:, k]
            scalers["Y"].append((float(col.mean()), float(col.std() or 1.0)))

        # replace with augmented stats if any augmentation is active --------
        if self.noise or self.ap or self.at:
            scalers["Y"] = self._augmented_target_stats()

        # persist
        np.savez(self.tc, x_arr=X_full,
                 y_arr=Y_raw_full,
                 k_arr=K_full,
                 i_arr=I_full,
                 scalers=scalers)
        self.scalers = scalers

    # -------------------------- target scaling -----------------------------
    def _scale_Y(self, Y):
        log_once("Y_norm", "Transform: final μ/σ normalisation (8D target)")

        Y = _cart2polar_amp(Y.copy())
        Y = _polar_to_sincos(Y)
        Y = self._log_transform(Y)
        for k in range(Y.shape[-1]):
            if k == 1 or k == 2 or k == 3:          # sinφ, cosφ, Δt
                continue
            μ, σ = self.scalers["Y"][k]
            Y[..., k] = (Y[..., k] - μ) / σ
        return Y

    # ─────────────────────——— PyTorch dataset API ————————————————
    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        """
        Return one (*X*, *Y*) pair with **fresh augmentations** every call.

        • Feature scaling (μ/σ) is applied *after* per-sample whitening.
        • Output dtype: complex64 (FD) or float32 (TD) depending on `self.out`.
        """
        X, Y = augsample(self.X_tr[idx], self.Y_tr[idx], self.rng, self.K_tr[idx], self.dist_amp,
                         ifo=self.I_tr[idx], noise_ok=self.noise, phase_ok=self.ap, time_ok=self.at)

        # ------------- feature scaling -----------------------------------
        if self.out == "complex":
            μr, σr = self.scalers["X_real"]
            μi, σi = self.scalers["X_imag"]
            X = (X.real - μr) / σr + 1j * (X.imag - μi) / σi
            X_t = torch.as_tensor(X, dtype=torch.complex64, device=self.device)
        else:
            Xtd = irfft(X)
            μ, σ = self.scalers["X_td"]
            Xtd = (Xtd - μ) / σ
            X_t = torch.as_tensor(Xtd, dtype=torch.float32, device=self.device)

        Y_t = torch.as_tensor(self._scale_Y(Y), dtype=torch.float32,
                              device=self.device)
        return X_t, Y_t

    # ----------------------------------------------------------------------
    def __getitems__(self, idxs):
        """
        Vectorised *batch* version of `__getitem__`.

        Accepts a 1-D array/sequence of indices and returns two tensors where
        augmentations and scaling are applied to the whole batch at once.
        This cuts Python overhead by ≈ 20× for large mini-batches.
        """
        idxs = np.asarray(idxs)
        Xs = self.X_raw[idxs]
        Ys = self.Y_raw[idxs]
        Ks = self.k_raw[idxs]
        Is = self.i_raw[idxs]

        Xs, Ys = augsample(Xs, Ys, self.rng, Ks,
                                  self.dist_amp, ifo=Is, noise_ok=self.noise, phase_ok=self.ap, time_ok=self.at)

        # -------- feature scaling (vectorised) ----------------------------
        if self.out == "complex":
            μr, σr = self.scalers["X_real"]
            μi, σi = self.scalers["X_imag"]
            log_once("feature_scale", "Transform: scale X (real & imag) with μ/σ from train set")
            Xs = ((Xs.real - μr) / σr) + 1j * ((Xs.imag - μi) / σi)
            X_batch = torch.as_tensor(Xs, dtype=torch.complex64,
                                      device=self.device)
        else:
            log_once("irfft_scale", "Transform: convert X to time-domain via irfft and scale")
            Xtd = irfft(Xs)
            μtd, σtd = self.scalers["X_td"]
            Xtd = (Xtd - μtd) / σtd
            X_batch = torch.as_tensor(Xtd, dtype=torch.float32,
                                      device=self.device)

        Y_batch = torch.as_tensor(self._scale_Y(Ys), dtype=torch.float32,
                                  device=self.device)
        X_batch = X_batch.to(self.device, non_blocking=True)
        Y_batch = Y_batch.to(self.device, non_blocking=True)
        return X_batch, Y_batch

