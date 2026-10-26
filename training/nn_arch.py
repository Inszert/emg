"""
================================================================================
 NEURAL NETWORK ARCHITECTURES  --  the only place a layer is defined
================================================================================

 Shared, deliberately, by the trainer (`train_nn.py`) and the live inference
 worker (`nn_serve.py`). One definition, because a checkpoint is just a
 `state_dict`: if the two files described the shapes differently, loading
 would either raise or -- worse -- succeed on a net wired up differently from
 the one that was fitted.

 It imports **torch and nothing else**. Not `emg_dataset_tool`, not sklearn.
 That is what lets `nn_serve.py` reach its first prediction quickly instead
 of pulling in the data layer and scikit-learn to answer one window.

 --------------------------------------------------------------------------
 WHICH ARCHITECTURE, AND WHY  (the question this file exists to answer)
 --------------------------------------------------------------------------

 The input is tiny and the dataset is tiny, and those two facts decide almost
 everything:

   * a window is 18-30 time steps of 8 channels (0.30-0.50 s at MODEL_FS =
     60 Hz), already rectified and envelope-filtered on the armband;
   * after window-level dedup (CLAUDE.md section 8g) the whole dataset is
     roughly a thousand independent windows from two wearers.

 A thousand windows will not train a large recurrent stack. What it will
 train is a few thousand parameters that are forced to look at how the
 envelope CHANGES across the window -- which is the part a per-window
 statistics vector throws away, and the reason to have a net here at all.

 So the DEFAULT is `tcn`:

   * depthwise-separable dilated convolutions, kernel 3, dilations 1/2/4.
     Each block reaches 2*dilation further, so three of them span
     1 + 2*(1+2+4) = 15 ... 29 samples depending on where you count from:
     the whole window either way. Reaching across the window therefore costs
     parameters logarithmically in its length instead of linearly in hidden
     units.
   * depthwise separation matters at this size: a dense Conv1d(8->32, k=3) is
     768 weights, the separable pair is 24 + 256. Same receptive field, a
     third of the parameters, on a dataset that cannot afford them.
   * it is fully parallel over time, so one window is one batched forward
     pass -- the difference between fitting inside the live prediction
     interval and not (CLAUDE.md section 12 records the decision panel
     already overrunning its 150 ms budget).

 `gru` and `cnn_gru` are the recurrent candidates, and they are candidates
 rather than the default: at 30 steps a GRU's sequential advantage is small,
 while its parameter count per unit of memory is large. The sweep measures
 all of them; nothing here is asserted.

 ATTENTION POOLING, not the last time step. The retired experiment
 (`train_torch.py`, CLAUDE.md section 8b) fed `out[:, -1, :]` to the head,
 which asks the net to compress a whole gesture into the state it happens to
 hold at the final sample of an arbitrary window boundary. On a rectified
 envelope that sample is not special, and in a window cut mid-hold it is not
 even the most informative one. `AttentionPool` learns a weight per time step
 and reports the weighted mean AND the plain mean, so the head sees "which
 part mattered" and "how much there was overall" -- the second being the
 activation level this project has twice measured as load-bearing
 (CLAUDE.md section 13).

 `lstm` and `cnn` keep the OLD last-step / global-average heads on purpose.
 They are the historical baseline from the retired sweep, and modernising
 them would make any comparison with its numbers meaningless.
"""
from __future__ import annotations

#  The first five are the original set; the last four were added 2026-10-27
#  after tools/new_models_test.py measured them under the project protocol
#  (docs/REPORT.md, "newer models"). Order matters: the trainer's default
#  (and `--quick`) is the head of this list.
KINDS = ["tcn", "gru", "cnn_gru", "lstm", "cnn",
         "circ_gru", "chan_transformer", "inception", "tsmixer"]

#  What each kind is, in one line, for the trainer's log and the GUI.
DESCRIPTIONS = {
    "tcn": "dilated depthwise-separable temporal convnet (1/2/4) + attention "
           "pooling -- the default: smallest, spans the whole window, fully "
           "parallel over time",
    "gru": "single-layer GRU + attention pooling over time",
    "cnn_gru": "separable conv front end, then a GRU, + attention pooling",
    "lstm": "two stacked LSTMs, last time step -- the retired experiment's "
            "baseline, kept for comparison",
    "cnn": "plain 1D convnet with global average pooling -- the retired "
           "experiment's baseline, kept for comparison",
    "circ_gru": "conv over (time x pod) with CIRCULAR padding on the pod axis "
                "(the band is a ring) + GRU + attention pooling; pod identity "
                "is kept",
    "chan_transformer": "each electrode's whole window is one token "
                        "(iTransformer style), 2 pre-norm encoder layers",
    "inception": "small InceptionTime: 3 inception modules (kernels 3/7/15) "
                 "with a residual, mean ++ max pooling",
    "tsmixer": "all-MLP TSMixer: time-mixing and feature-mixing blocks, "
               "mean ++ max pooling",
}

#  Defaults. The trainer may override either per candidate; a checkpoint
#  stores what was actually used and `nn_serve.py` rebuilds from the
#  checkpoint, never from these.
HIDDEN = 32
DROPOUT = 0.25


def _build_module(n_channels, n_classes, win, hidden, dropout, kind):
    """Build the module for `kind`. Separate so `build()` stays readable."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class AttentionPool(nn.Module):
        """
        (B, C, L) -> (B, 2C):  attention-weighted mean  ++  plain mean.

        Two halves on purpose. The attention half answers "which part of this
        window carried the gesture", which is the thing that varies in time;
        the plain mean keeps the overall activation level. Keeping only the
        attention half would let the softmax normalise that level away, and
        removing the level is the mistake CLAUDE.md section 13 records
        measuring twice.
        """

        def __init__(self, channels):
            super().__init__()
            self.score = nn.Conv1d(channels, 1, kernel_size=1)

        def forward(self, h):                            # h: (B, C, L)
            weights = torch.softmax(self.score(h), dim=-1)        # (B, 1, L)
            attended = (h * weights).sum(dim=-1)                  # (B, C)
            return torch.cat([attended, h.mean(dim=-1)], dim=1)

    class SeparableBlock(nn.Module):
        """
        Depthwise dilated conv -> pointwise mix -> norm -> GELU -> dropout,
        with a residual path.

        `padding = dilation` at kernel 3 keeps the length unchanged, so the
        stack can be as deep as the dilation schedule without the window
        shrinking away. At 18 samples there is nothing to spare.
        """

        def __init__(self, c_in, c_out, dilation):
            super().__init__()
            self.depthwise = nn.Conv1d(c_in, c_in, kernel_size=3,
                                       padding=dilation, dilation=dilation,
                                       groups=c_in)
            self.pointwise = nn.Conv1d(c_in, c_out, kernel_size=1)
            self.norm = nn.BatchNorm1d(c_out)
            self.act = nn.GELU()
            self.drop = nn.Dropout(dropout)
            self.skip = (nn.Identity() if c_in == c_out
                         else nn.Conv1d(c_in, c_out, kernel_size=1))

        def forward(self, x):
            y = self.drop(self.act(self.norm(
                self.pointwise(self.depthwise(x)))))
            return y + self.skip(x)

    class TCN(nn.Module):
        def __init__(self):
            super().__init__()
            widths = [n_channels, max(16, hidden * 3 // 4), hidden, hidden]
            self.blocks = nn.Sequential(*[
                SeparableBlock(widths[i], widths[i + 1], dilation)
                for i, dilation in enumerate((1, 2, 4))])
            self.pool = AttentionPool(hidden)
            self.head = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(hidden * 2, n_classes))

        def forward(self, x):                            # x: (B, win, C)
            return self.head(self.pool(self.blocks(x.transpose(1, 2))))

    class GRUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.rnn = nn.GRU(n_channels, hidden, num_layers=1,
                              batch_first=True)
            self.pool = AttentionPool(hidden)
            self.head = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(hidden * 2, n_classes))

        def forward(self, x):
            out, _ = self.rnn(x)                         # (B, L, H)
            return self.head(self.pool(out.transpose(1, 2)))

    class CNNGRU(nn.Module):
        def __init__(self):
            super().__init__()
            width = max(16, hidden * 3 // 4)
            self.front = SeparableBlock(n_channels, width, 1)
            self.rnn = nn.GRU(width, hidden, num_layers=1, batch_first=True)
            self.pool = AttentionPool(hidden)
            self.head = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(hidden * 2, n_classes))

        def forward(self, x):
            z = self.front(x.transpose(1, 2)).transpose(1, 2)
            out, _ = self.rnn(z)
            return self.head(self.pool(out.transpose(1, 2)))

    class LSTMNet(nn.Module):
        """The retired experiment's LSTM, unchanged: last time step."""

        def __init__(self):
            super().__init__()
            self.rnn = nn.LSTM(n_channels, hidden, num_layers=2,
                               batch_first=True, dropout=dropout)
            self.head = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(hidden, n_classes))

        def forward(self, x):
            out, _ = self.rnn(x)
            return self.head(out[:, -1, :])

    class CNN(nn.Module):
        """The retired experiment's convnet, unchanged: global average pool."""

        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(n_channels, 32, kernel_size=5, padding=2),
                nn.BatchNorm1d(32), nn.ReLU(),
                nn.Conv1d(32, 64, kernel_size=5, padding=2),
                nn.BatchNorm1d(64), nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(64, hidden, kernel_size=3, padding=1),
                nn.BatchNorm1d(hidden), nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout),
                                      nn.Linear(hidden, n_classes))

        def forward(self, x):
            return self.head(self.features(x.transpose(1, 2)))

    class CircConvGRU(nn.Module):
        """
        Conv over (time x pod) with CIRCULAR padding on the pod axis: pod 8 is
        next to pod 1, as on the band. The pod axis is NOT pooled away -- which
        pod is active is what separates WAVE_LEFT from WAVE_RIGHT -- so the net
        keeps pod identity and only learns neighbourhood patterns.
        """

        def __init__(self, filters=8):
            super().__init__()
            self.conv = nn.Conv2d(1, filters, kernel_size=(3, 3),
                                  padding=(1, 0))
            self.bn = nn.BatchNorm2d(filters)
            self.rnn = nn.GRU(filters * n_channels, hidden, batch_first=True)
            self.pool = AttentionPool(hidden)
            self.head = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(hidden * 2, n_classes))

        def forward(self, x):                                # (B, T, C)
            z = F.pad(x.unsqueeze(1), (1, 1, 0, 0), mode="circular")
            z = F.gelu(self.bn(self.conv(z)))                # (B, f, T, C)
            b, f, t, c = z.shape
            z = z.permute(0, 2, 1, 3).reshape(b, t, f * c)
            out, _ = self.rnn(z)
            return self.head(self.pool(out.transpose(1, 2)))

    class ChanTransformer(nn.Module):
        """Each electrode's whole window is one token (iTransformer style)."""

        def __init__(self, heads=4, layers=2):
            super().__init__()
            d = hidden
            self.embed = nn.Linear(win, d)
            self.pos = nn.Parameter(torch.randn(1, n_channels, d) * 0.02)
            layer = nn.TransformerEncoderLayer(
                d, heads, dim_feedforward=2 * d, dropout=dropout,
                activation="gelu", batch_first=True, norm_first=True)
            self.enc = nn.TransformerEncoder(layer, layers,
                                             enable_nested_tensor=False)
            self.norm = nn.LayerNorm(d)
            #  Flattened over pods, so pod identity reaches the head.
            self.head = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(d * n_channels, n_classes))

        def forward(self, x):                                # (B, T, C)
            z = self.embed(x.transpose(1, 2)) + self.pos     # (B, C, d)
            z = self.norm(self.enc(z))
            return self.head(z.flatten(1))

    class InceptionModule(nn.Module):
        def __init__(self, c_in, nf=8, kernels=(3, 7, 15)):
            super().__init__()
            self.bottle = nn.Conv1d(c_in, nf, 1, bias=False)
            self.convs = nn.ModuleList(
                [nn.Conv1d(nf, nf, k, padding=k // 2, bias=False)
                 for k in kernels])
            self.pool = nn.MaxPool1d(3, stride=1, padding=1)
            self.pconv = nn.Conv1d(c_in, nf, 1, bias=False)
            self.bn = nn.BatchNorm1d(nf * (len(kernels) + 1))

        def forward(self, x):
            b = self.bottle(x)
            out = torch.cat([c(b) for c in self.convs]
                            + [self.pconv(self.pool(x))], dim=1)
            return F.gelu(self.bn(out))

    class InceptionTimeS(nn.Module):
        def __init__(self, nf=8):
            super().__init__()
            width = nf * 4
            self.m1 = InceptionModule(n_channels, nf)
            self.m2 = InceptionModule(width, nf)
            self.m3 = InceptionModule(width, nf)
            self.res = nn.Sequential(nn.Conv1d(n_channels, width, 1,
                                               bias=False),
                                     nn.BatchNorm1d(width))
            #  Mean ++ max: the mean keeps the activation level.
            self.head = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(width * 2, n_classes))

        def forward(self, x):
            x = x.transpose(1, 2)
            h = self.m3(self.m2(self.m1(x)))
            h = F.gelu(h + self.res(x))
            return self.head(torch.cat([h.mean(-1), h.amax(-1)], dim=1))

    class MixerBlock(nn.Module):
        def __init__(self, d):
            super().__init__()
            self.n1 = nn.LayerNorm(d)
            self.time = nn.Sequential(nn.Linear(win, win), nn.GELU(),
                                      nn.Dropout(dropout))
            self.n2 = nn.LayerNorm(d)
            self.feat = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(),
                                      nn.Dropout(dropout), nn.Linear(2 * d, d),
                                      nn.Dropout(dropout))

        def forward(self, x):                                # (B, T, d)
            x = x + self.time(self.n1(x).transpose(1, 2)).transpose(1, 2)
            return x + self.feat(self.n2(x))

    class TSMixer(nn.Module):
        def __init__(self, blocks=2):
            super().__init__()
            d = max(16, hidden * 3 // 4)
            self.inp = nn.Linear(n_channels, d)
            self.blocks = nn.Sequential(*[MixerBlock(d)
                                          for _ in range(blocks)])
            self.head = nn.Sequential(nn.Dropout(dropout),
                                      nn.Linear(d * 2, n_classes))

        def forward(self, x):
            z = self.blocks(self.inp(x))
            return self.head(torch.cat([z.mean(1), z.amax(1)], dim=1))

    table = {"tcn": TCN, "gru": GRUNet, "cnn_gru": CNNGRU,
             "lstm": LSTMNet, "cnn": CNN, "circ_gru": CircConvGRU,
             "chan_transformer": ChanTransformer, "inception": InceptionTimeS,
             "tsmixer": TSMixer}
    if kind not in table:
        raise ValueError(f"unknown architecture {kind!r}; known: "
                         f"{', '.join(KINDS)}")
    return table[kind]()


def build(kind: str, n_channels: int, n_classes: int, win: int,
          hidden: int = HIDDEN, dropout: float = DROPOUT, seed: int = 0):
    """
    One net, ready to fit.

    THE SEED IS TAKEN HERE, not by the caller afterwards. Weight
    initialisation happens during construction, so seeding after the fact
    leaves the starting point wherever the global RNG had got to and two
    "same-seed" fits diverge. CLAUDE.md section 8b records this being found
    the hard way; `tests/test_nn.py` pins it by comparing probabilities
    rather than the argmax, because on a separable problem even two unrelated
    fits agree on the argmax.
    """
    import torch
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    return _build_module(int(n_channels), int(n_classes), int(win),
                         int(hidden), float(dropout), kind)


def count_parameters(model) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
