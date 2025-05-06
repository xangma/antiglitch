import torch
import complextorch.nn as cvnn     # thin wrapper around torch.nn for complex numbers
from torch import nn
from torch.utils.checkpoint import checkpoint, checkpoint_sequential
from torch.nn import functional as F
import math

class SimpleComplexBN1d(nn.Module):
    """
    A convenience wrapper that lets you slap BatchNorm on a
    complex tensor shaped (B,C) or (B,C,L).

    Idea: treat real & imag channels as if they were two separate
    real feature maps, then stitch them back together.
    """
    def __init__(self, num_features: int, **bn_kw):
        super().__init__()
        # double the channels ⇒ [real, imag]
        self.bn = nn.BatchNorm1d(num_features * 2, affine=True)

    def forward(self, z: torch.Tensor) -> torch.Tensor:   # z is complex
        orig_shape = z.shape                              # keep for squeeze logic
        if z.dim() == 2:                                  # (B,C) → (B,C,1)
            z = z.unsqueeze(-1)

        B, C, L = z.shape                                 # (B,C,L)
        ri = torch.view_as_real(z)                        # → (B,C,L,2)
        ri = ri.permute(0, 3, 1, 2).reshape(B, 2 * C, L)  # → (B,2C,L)
        ri = self.bn(ri)                                  # BN on concatenated channels
        ri = ri.view(B, 2, C, L).permute(0, 2, 3, 1)      # back to (B,C,L,2)
        z  = torch.view_as_complex(ri.contiguous())       # …then to complex
        return z.squeeze(-1) if len(orig_shape) == 2 else z


class ComplexValuedNN(nn.Module):
    """
    Minimal example “CVNN”:

    2 × (Conv1d + modReLU + BN)        ┐
          → adaptive avg-pool        ├─ complex trunk
    2 × (Linear + modReLU + Dropout)   ┘
          → real regression head (6 outputs)

    All convolutions are in the *frequency* domain (length-513 spectra).
    """

    def __init__(
        self,
        n_conv_layers: int     = 3,
        conv_filters: list[int] = [16, 32, 64],
        conv_kernel_size: list[int] = [1, 4, 8],
        n_fc_layers: int       = 2,
        n_fc_units: int        = 128,
        dropout: float | None  = None,
        pool_size: int         = 1,           # 1 ⇒ true global avg-pool
    ):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(6))
        # ───────────────────────── convolutional trunk ──────────────────────────
        self.conv_layers = nn.ModuleList()
        in_ch = 1
        for i in range(n_conv_layers):
            self.conv_layers.append(
                nn.Sequential(
                    cvnn.Conv1d(in_ch, conv_filters[i], conv_kernel_size[i], padding="same"),
                    cvnn.modReLU(bias=-0.1),
                    cvnn.BatchNorm1d(conv_filters[i]),
                    *( [cvnn.Dropout(dropout)] if dropout else [] )
                )
            )
            in_ch = conv_filters[i]

        # ───────────────────────── global / adaptive pooling ─────────────────────
        self.gap = cvnn.AdaptiveAvgPool1d(pool_size)     # keeps C×pool_size features

        # ───────────────────────── fully-connected head ──────────────────────────
        self.fc_layers = nn.ModuleList()
        fc_in = conv_filters[-1] * pool_size
        for i in range(n_fc_layers):
            self.fc_layers += [
                cvnn.Linear(fc_in if i == 0 else n_fc_units, n_fc_units),
                # cvnn.modReLU(bias=-0.1),  # complex ReLU
            ]
            if dropout:
                self.fc_layers.append(cvnn.Dropout(dropout))

        # ───────────────────────── real-valued regression head ───────────────────
        self.output_layer = nn.Linear(n_fc_units * 2, 6)

    # ---------------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input: (B,513) real-imag interleaved along complex dtype
        if x.dim() == 2:                        # (B,513)
            x = x.unsqueeze(1)                  # (B,1,513)

        for block in self.conv_layers:
            x = block(x)  # Each block is Conv + modReLU + BN (+ optional Dropout)

        x = self.gap(x)                         # (B,C,1)
        x = x.squeeze(-1)                       # → (B,C)

        for layer in self.fc_layers:
            x = layer(x)

        # split complex last-dim into two real vectors
        x = torch.cat((x.real, x.imag), dim=1)  # (B,2·n_fc_units)
        return self.output_layer(x)             # (B,6)


class CResBlock(nn.Module):
    """
    Complex-valued 1-D residual block
      (Conv/wFM → BN → modReLU) × 2  +  identity / 1×1 projection
    """
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 k: int            = 5,
                 stride: int       = 1,
                 conv_type: str    = "conv",
                 dropout: float | None = None):
        super().__init__()

        Conv = cvnn.wFMConv1d if conv_type == "wfm" else cvnn.Conv1d
        self.proj = (
            Conv(c_in, c_out, 1, stride=stride)        # 1×1 complex conv
            if (c_in != c_out or stride != 1) else nn.Identity()
        )

        layers = [
            Conv(c_in, c_out, k, stride=stride, padding=k // 2),
            cvnn.BatchNorm1d(c_out),
            cvnn.modReLU(bias=-0.1),
        ]
        if dropout:
            layers.append(cvnn.Dropout(dropout))
        layers += [
            Conv(c_out, c_out, k, padding=k // 2),
            cvnn.BatchNorm1d(c_out),
        ]
        if dropout:
            layers.append(cvnn.Dropout(dropout))

        self.net = nn.Sequential(*layers)
        self.act = cvnn.modReLU(bias=-0.1)

    def forward(self, z):
        return self.act(self.net(z) + self.proj(z))


# ─────────────────────────────── full network ────────────────────────────────
class GlitchNetRes(nn.Module):
    """
    Residual complex CNN for 513-bin FD glitches.

    Parameters
    ----------
    channels   : list[int]   output channels of each residual stage
    strides    : list[int]   stride of the *first* block in each stage
    conv_type  : "conv" | "wfm"
    use_attn   : insert one complex Multi-Head Attention before GAP
    dropout    : optional complex dropout inside blocks
    pool_size  : AdaptiveAvgPool1d output length (1 ⇒ global average)
    """
    def __init__(
        self,
        channels:  list[int] = [32, 64, 128],
        strides:   list[int] = [1,   2,   2],
        kernel_sz: list[int] = [7,   5,   3],
        conv_type: str       = "wfm",
        use_attn:  bool      = False,
        n_heads:   int       = 4,
        dropout:   float | None = 0.1,
        pool_size: int       = 1,
        checkpoint_trunk: bool = True
    ):
        super().__init__()
        self.log_vars = nn.Parameter(torch.zeros(6))
        self.checkpoint_trunk = checkpoint_trunk
        assert len(channels) == len(strides) == len(kernel_sz)

        # ─── residual “trunk” ──────────────────────────────────────────────
        blocks = []
        c_in = 1
        for c_out, st, k in zip(channels, strides, kernel_sz):
            blocks.append(CResBlock(c_in, c_out, k, stride=st,
                                    conv_type=conv_type, dropout=dropout))
            blocks.append(CResBlock(c_out, c_out, k,
                                    conv_type=conv_type, dropout=dropout))
            c_in = c_out
        self.trunk = nn.Sequential(*blocks)

        # ─── optional complex self-attention ───────────────────────────────
        if use_attn:
            self.attn = cvnn.MultiheadAttention(
                n_heads=n_heads,
                d_model=channels[-1],
                d_k=channels[-1] // n_heads,
                d_v=channels[-1] // n_heads,
                # bias=True,
            )
        else:
            self.attn = None

        # ─── global pooling & real head ────────────────────────────────────
        self.gap = cvnn.AdaptiveAvgPool1d(pool_size)
        self.head = nn.Sequential(
            nn.Flatten(),                                    # (B, C·P) complex
            nn.Linear(channels[-1] * pool_size * 2, 128),    # ×2 ⇒ split Re/Im
            nn.ReLU(),
            nn.Linear(128, 6),
        )

    # -----------------------------------------------------------------------
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        if z.dim() == 2:                       # (B,513) → (B,1,513)
            z = z.unsqueeze(1)

        if self.checkpoint_trunk:
            z = checkpoint_sequential(self.trunk, len(self.trunk), z)
        else:
            z = self.trunk(z)
        if self.attn is not None:
            # reshape (B,C,L) → (L,B,C) for attention module, then back
            z = z.transpose(1, 2)              # (B,L,C) → (B,L,C)
            z = self.attn(z, z, z)          # complex MHA
            z = z.transpose(1, 2)              # (B,L,C) → (B,C,L)

        z = self.gap(z).squeeze(-1)            # (B,C)
        z = torch.cat((z.real, z.imag), dim=1) # (B,2C)
        return self.head(z.float())            # 6-D real output



class _CVBlock(nn.Module):
    """
    Tiny helper: Conv → Complex BN → Polar-tanh.
    Keeps feature length unchanged (uses padding).
    """
    def __init__(self, c_in: int, c_out: int, k: int = 5,
                 stride: int = 1, groups: int = 1):
        super().__init__()
        self.conv = cvnn.Conv1d(c_in, c_out, k, stride,
                                padding=k//2, bias=False, groups=groups)
        self.bn   = cvnn.BatchNorm1d(c_out, eps=1e-3, momentum=0.1)
        # self.act  = cvnn.CVPolarTanh()
        # self.act = cvnn.modReLU(bias=-0.05)  # complex ReLU
        self.act = cvnn.CVCardiod()  # complex ReLU

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(z)))


class GlitchNetCV(nn.Module):
    """
    Predicts (|A|, sinφ, cosφ, Δt, f0, gbw) **and** log-σ for the 4
    heteroskedastic targets, so the output dimension is 10:
    [|A|, logσ_A, Δt, logσ_t, f0, logσ_f, gbw, logσ_g, sinφ, cosφ]
    """
    def __init__(self, stem_width: int = 32, width_mult: int = 2):
        super().__init__()
        ws = stem_width

        ### 1. Complex convolutional backbone
        self.stem = _CVBlock(1, ws, k=7)
        self.block1 = _CVBlock(ws,  ws*width_mult, stride=2)   # 513→257
        self.block2 = _CVBlock(ws*width_mult, ws*width_mult, k=3)
        self.block3 = _CVBlock(ws*width_mult, ws*width_mult*2, stride=2)  # 257→129
        self.block4 = _CVBlock(ws*width_mult*2, ws*width_mult*2, k=3)
        # keep 16 frequency “tokens” so Δt’s phase-ramp survives the pooling
        self.pool   = cvnn.AdaptiveAvgPool1d(16)
        ### 2. “Tokenise” the global phase (optional but improves φ)
        self.register_buffer("_two_pi", torch.tensor(6.283185307179586, dtype=torch.float32))

        ### 3. Two-step real MLP head
        #  C channels × 16 freq-bins × 2 (Re/Im)  + 1 phase token
        # channels after last block
        C = ws * width_mult * 2
        # 2 × C × 16  (Re/Im)  +   C (dφ)  + 1 (phase token)
        feat_dim = 2 * C * 16 + C + 1
        self.fc1 = nn.Linear(feat_dim, 256)
        self.fc2 = nn.Linear(256, 10)               # final logits

        # Initialise complex kernels with Kaiming-uniform separately
        for m in self.modules():
            if isinstance(m, cvnn.Conv1d):
                nn.init.kaiming_uniform_(m.conv.weight.real, nonlinearity='relu')
                nn.init.kaiming_uniform_(m.conv.weight.imag, nonlinearity='relu')
            if isinstance(m, cvnn.BatchNorm1d):
                m.reset_running_stats()

    # ------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: complex tensor (B,1,513)
        returns: real tensor (B, 10)
        """
        # ─── complex backbone ─────────────────────────────────────────────
        z = self.pool(self.block4(self.block3(self.block2(
                self.block1(self.stem(x))))))          # (B, C, 16) complex

        # ─── phase-slope token  (breaks Δt aliasing) ─────────────────────
        phi  = torch.angle(z)                          # (B,C,16)
        dphi = (phi[..., 1:] - phi[..., :-1]).mean(-1) # (B,C)
        dphi = torch.remainder(dphi + math.pi, 2*math.pi) - math.pi   # unwrap

        # ─── global phase token  (improves φ) ────────────────────────────
        phase_token = torch.atan2(z.imag.mean(dim=(1, 2)),
                                z.real.mean(dim=(1, 2))).unsqueeze(1)
        phase_token = torch.sin(phase_token)           # (B,1)

        # ─── flatten & concatenate features ──────────────────────────────
        h = torch.cat([
            z.real.flatten(1),                         # (B, C*16)
            z.imag.flatten(1),                         # (B, C*16)
            dphi,                                      # (B, C)
            phase_token                                # (B, 1)
        ], dim=1)                                      # (B, 2C*16 + C + 1)

        h   = torch.relu(self.fc1(h))
        out = self.fc2(h)                              # (B, 10)

        # no activation on sinφ / cosφ logits; loss handles them
        return out


class GlitchRegressor(nn.Module):
    def __init__(self):
        super().__init__()
        C1, C2, C3, C4 = 16, 32, 64, 128     # shorthand
        NEG_BIAS = -0.1
        AFFINE = True


        def Cconv(cin, cout, groups=4):
            return nn.Sequential(
                cvnn.Conv1d(cin, cout, kernel_size=7, padding=3),
                cvnn.GroupNorm(num_groups=groups, num_channels=cout, affine=True),
                cvnn.modReLU(NEG_BIAS),
                # cvnn.CVPolarTanh(),
            )

        self.feat1 = nn.Sequential(
            Cconv(1,  C1),
            Cconv(C1, C2),
            Cconv(C2, C3),
        )
        self.pool1 = nn.Sequential(
            cvnn.AdaptiveAvgPool1d(128),
        )

        self.feat2 = nn.Sequential(
            Cconv(C3, C4),
            Cconv(C4, C4),
        )
        self.pool2 = nn.Sequential(
            cvnn.AdaptiveAvgPool1d(1),        # (B,128,1)
        )

        self.trunk = nn.Sequential(
            cvnn.Linear(128, 64),
            cvnn.GroupNorm(1, 64, affine=AFFINE),
            cvnn.modReLU(NEG_BIAS),
            # cvnn.CVPolarTanh(),
            cvnn.Dropout(0.2),
        )
        self.head = nn.Linear(128, 14, bias=True)

    def forward(self, x):
        dev = x.device
        z = self.feat1(x)
        z = self.pool1(z)
        z = self.feat2(z)
        z = self.pool2(z)

        z = z.squeeze(-1)          # (B,128) complex
        z = self.trunk(z)
        return self.head(torch.view_as_real(z).flatten(1))

class RealValuedNN(nn.Module):
    """
    Same topology as above but with *real* weights.

    We treat the complex input as 2-channel real data
    [real, imag] → Conv1d(in=2, …).
    """
    def __init__(
        self,
        n_conv_layers: int = 2,
        conv_filters: list[int] = [8, 16],
        conv_kernel_size: list[int] = [1, 3],
        n_fc_layers: int = 2,
        n_fc_units: int = 128,
        dropout: float | None = None
    ):
        super().__init__()

        self.conv_layers = nn.ModuleList()
        in_ch = 2                                          # real + imag
        for i in range(n_conv_layers):
            self.conv_layers += [
                nn.Conv1d(in_ch, conv_filters[i], conv_kernel_size[i], padding="same"),
                nn.ReLU(),
                nn.BatchNorm1d(conv_filters[i])
            ]
            if dropout:
                self.conv_layers.append(nn.Dropout(dropout))
            in_ch = conv_filters[i]

        self.flatten_size = conv_filters[-1] * 513         # Conv output is (B,C,513)

        # FC trunk
        self.fc_layers = nn.ModuleList()
        for i in range(n_fc_layers):
            self.fc_layers += [
                nn.Linear(self.flatten_size if i == 0 else n_fc_units, n_fc_units),
                nn.ReLU()
            ]
            if i != n_fc_layers - 1:
                self.fc_layers.append(nn.BatchNorm1d(n_fc_units))
            if dropout:
                self.fc_layers.append(nn.Dropout(dropout))

        self.output_layer = nn.Linear(n_fc_units, 2)       # 2 real targets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Make channels=[real, imag]
        x = torch.stack((x.real, x.imag), dim=1)           # (B,2,513)

        for layer in self.conv_layers:
            x = layer(x)                                   # (B,C,513)

        x = x.flatten(1)                                   # (B,flatten_size)

        for layer in self.fc_layers:
            x = layer(x)

        return self.output_layer(x)


class RealValuedNNtd(nn.Module):
    """
    Time-domain variant:
      – Conv → ReLU → BN → MaxPool
      – second Conv block
      – 3-layer LSTM
      – fully-connected regression head
    """
    def __init__(self):
        super().__init__()
        # Conv block 1
        self.conv1   = nn.Conv1d(1,  64, kernel_size=3, padding="same")
        self.max1    = nn.MaxPool1d(2)
        self.act1    = nn.ReLU()
        self.bn1     = nn.BatchNorm1d(64)

        # Conv block 2
        self.conv2   = nn.Conv1d(64, 128, kernel_size=6, padding="same")
        self.max2    = nn.MaxPool1d(2)
        self.act2    = nn.ReLU()
        self.bn2     = nn.BatchNorm1d(128)

        # LSTM over (time,feat) = (256,128) → (B,256,32)
        self.lstm = nn.LSTM(input_size=128, hidden_size=32, num_layers=3, batch_first=True)

        self.output_layer = nn.Linear(32, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.unsqueeze(1)          # (B,1,512?)

        x = self.bn1(self.act1(self.max1(self.conv1(x))))
        x = self.bn2(self.act2(self.max2(self.conv2(x))))

        x = x.permute(0, 2, 1)      # LSTM expects (B,T,F)
        x, _ = self.lstm(x)         # keep full seq, drop hidden state
        return self.output_layer(x[:, -1, :])  # last time-step
