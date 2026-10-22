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

KINDS = ["tcn", "gru", "cnn_gru", "lstm", "cnn"]

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

    table = {"tcn": TCN, "gru": GRUNet, "cnn_gru": CNNGRU,
             "lstm": LSTMNet, "cnn": CNN}
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
