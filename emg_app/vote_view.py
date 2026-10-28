"""
How a soft vote voted (Qt-free, torch-free).

A vote of several models is the mean of their probabilities, which is exactly
what hides disagreement: the panel shows one answer. `member_rows()` keeps the
members' own answers for the SAME window, and `vote_html()` writes them as the
few lines the LIVE CLASSIFIER and NEURAL NETWORK panels show under their bars.

The members' probabilities go through the same class mask as the vote (a
gesture switched off in the settings is off for every member too), so a member
can never name a gesture the vote was told to skip. It is display only: the
answer is still the mean, computed where it always was.
"""
from __future__ import annotations

import html
import math
from typing import List, Sequence

import numpy as np

from emg_app.class_mask import apply_class_mask


def member_rows(names: Sequence[str], member_probs: Sequence, labels: Sequence[str],
                final_label: str) -> List[dict]:
    """One dict per member: its name, its own top answer, and whether it agrees
    with the vote. Members with a probability vector of the wrong width are
    skipped rather than guessed at."""
    rows = []
    for name, probs in zip(names, member_probs):
        p = np.asarray(probs, dtype=float).ravel()
        if p.shape[0] != len(labels):
            continue
        masked, answered = apply_class_mask(p, labels)
        if not answered:
            rows.append({"name": str(name), "label": "?", "confidence": math.nan,
                         "agrees": False})
            continue
        top = int(np.argmax(masked))
        rows.append({"name": str(name), "label": str(labels[top]),
                     "confidence": float(masked[top]),
                     "agrees": str(labels[top]) == str(final_label)})
    return rows


def vote_html(rows: Sequence[dict], final_label: str, good: str, warn: str,
              dim: str) -> str:
    """The panel text: a header, one line per member, and the verdict."""
    if not rows:
        return ""
    agree = sum(1 for r in rows if r["agrees"])
    head = (f"<span style='color:{dim}'>VOTE of {len(rows)}: "
            f"{agree} of {len(rows)} say {html.escape(str(final_label))}</span>")
    lines = [head]
    for r in rows:
        conf = ("--" if math.isnan(r["confidence"])
                else f"{r['confidence'] * 100:.0f} %")
        colour = good if r["agrees"] else warn
        mark = "=" if r["agrees"] else "x"
        lines.append(f"<span style='color:{colour}'>{mark} "
                     f"{html.escape(r['name'])}: {html.escape(r['label'])} "
                     f"{conf}</span>")
    return "<br>".join(lines)
