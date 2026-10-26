"""
Training-time augmentation for the neural nets. torch only; never imported by
the recorder.

Five augmentations, each switchable on its own (the ablation study measures
them one at a time, then only the ones that helped together):

    gain      one gain per window (+-15 %) and a small gain per channel
              (+-8 %): electrode pressure and skin drift
    noise     multiplicative jitter on every sample (8 %): sensor noise
    timewarp  each window is replayed 15 % faster or slower and shifted by up
              to 2 samples: how long and how fast a person holds a gesture
    rotate    the band slid round the arm by one pod either way (circular roll
              of the channel axis): the one thing calibration cannot fix
    chandrop  one electrode reads zero (30 % of windows): a pod that lost
              contact

They act on the RAW calibrated window (counts), BEFORE the net's own
log1p + standardise scaler, because that is where the physical effect lives
(a gain is a multiplication in counts, not in log space). `Augmenter.__call__`
therefore takes raw `(B, T, C)` and returns raw `(B, T, C)`; the trainer then
scales it.

Nothing here is applied at inference, and with `kinds == []` nothing is
applied at all and the trainer's original code path is used unchanged.
"""
from __future__ import annotations

AUG_TYPES = ["gain", "noise", "timewarp", "rotate", "chandrop"]

AUG_DESCRIPTIONS = {
    "gain": "random overall and per-channel gain (+-15 % / +-8 %)",
    "noise": "multiplicative sample jitter (8 %)",
    "timewarp": "speed +-15 % and shift +-2 samples",
    "rotate": "band rotated by one pod (circular channel roll)",
    "chandrop": "one electrode reads zero (30 % of windows)",
}


def parse_augment(spec):
    """
    "none" / "" / None -> [];  "all" -> every type;  "gain,rotate" -> those.
    Raises ValueError on an unknown name (a typo must not train un-augmented).
    """
    if spec is None:
        return []
    if isinstance(spec, (list, tuple)):
        names = [str(x).strip().lower() for x in spec]
    else:
        text = str(spec).strip().lower()
        if text in ("", "none", "off", "0", "false"):
            return []
        if text in ("all", "on"):
            return list(AUG_TYPES)
        names = [p.strip() for p in text.split(",") if p.strip()]
    bad = [n for n in names if n not in AUG_TYPES]
    if bad:
        raise ValueError(f"unknown augmentation {bad}; known: "
                         f"{', '.join(AUG_TYPES)}")
    return [n for n in AUG_TYPES if n in names]          # canonical order


class Augmenter:
    """Callable `(B, T, C) raw -> (B, T, C) raw`. Own RNG, so it is seeded."""

    def __init__(self, kinds, seed=0, device=None):
        import torch
        self.kinds = parse_augment(kinds)
        self.device = device
        self.gen = torch.Generator(device=device or "cpu")
        self.gen.manual_seed(int(seed))

    # -- random helpers (all on the generator's device) ------------------------
    def _rand(self, *shape):
        import torch
        return torch.rand(*shape, generator=self.gen, device=self.gen.device)

    def _randn(self, *shape):
        import torch
        return torch.randn(*shape, generator=self.gen, device=self.gen.device)

    def __call__(self, x):
        import torch
        if not self.kinds:
            return x
        b, t, c = x.shape
        if "gain" in self.kinds:
            g = torch.exp(0.15 * self._randn(b, 1, 1)
                          + 0.08 * self._randn(b, 1, c))
            x = x * g
        if "noise" in self.kinds:
            x = x * torch.exp(0.08 * self._randn(b, t, c))
        if "timewarp" in self.kinds:
            speed = 1.0 + 0.15 * (2 * self._rand(b, 1) - 1)          # (b,1)
            shift = 2.0 * (2 * self._rand(b, 1) - 1)
            base = torch.arange(t, device=x.device, dtype=x.dtype)[None, :]
            centre = (t - 1) / 2.0
            pos = ((base - centre) * speed + centre + shift).clamp(0, t - 1)
            lo = pos.floor().long()
            hi = (lo + 1).clamp(max=t - 1)
            w = (pos - lo.to(pos.dtype))[:, :, None]
            lo_i = lo[:, :, None].expand(b, t, c)
            hi_i = hi[:, :, None].expand(b, t, c)
            x = x.gather(1, lo_i) * (1 - w) + x.gather(1, hi_i) * w
        if "rotate" in self.kinds:
            shift = torch.randint(-1, 2, (b,), generator=self.gen,
                                  device=self.gen.device)
            shift = torch.where(self._rand(b) < 0.5, shift,
                                torch.zeros_like(shift))
            idx = (torch.arange(c, device=x.device)[None, :]
                   - shift[:, None]) % c
            x = x.gather(2, idx[:, None, :].expand(b, t, c))
        if "chandrop" in self.kinds:
            hit = self._rand(b) < 0.3
            which = torch.randint(0, c, (b,), generator=self.gen,
                                  device=self.gen.device)
            mask = torch.zeros(b, 1, c, dtype=torch.bool, device=x.device)
            mask[torch.arange(b, device=x.device), 0, which] = True
            mask &= hit[:, None, None]
            x = torch.where(mask, torch.zeros_like(x), x)
        return x


def torch_scale(raw, mean, std):
    """`(log1p(|x|) - mean) / std`, the net's scaler, on a tensor."""
    import torch
    return (torch.log1p(raw.abs()) - mean) / std
