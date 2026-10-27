"""
The model zoo: every trained classifier as a selectable entry (Qt-free,
torch-free).

LAYOUT
------
    models/zoo/
        ensembles.json            best soft-voting subset per family, measured
        ml-et-500/
            meta.json             what it is, how it was trained, how well it scores
            model.joblib          a payload exactly like gesture_model.joblib
        nn-gru/
            meta.json
            model.pt              a checkpoint exactly like gesture_nn.pt
            model.json            the plain-JSON sidecar NN STATS reads

`meta.json` is the only thing the GUI needs to list, rank, star and size a
model; the application never opens a `model.pt` (no torch). It is written by
`training/train_zoo.py` at the moment a model is trained, so the scores on
display always belong to the file next to them.

STARS (computed here, never stored)
-----------------------------------
Within each family (ML, NN) the models are ranked by leave-one-person-out
accuracy per window, ties broken by macro-F1 -- never by anything measured on
the training data. The top `STAR_TOP_N` (3) of each family get a star. The best
soft-voting subset (from `ensembles.json`) gets a star only if it scores higher
than every single model of its family. Because the stars are a function of the
metadata, they are recomputed whenever the zoo is listed -- in particular after
every training, which rewrites the metadata.

SELECTION
---------
`cfg.MODEL_SELECTION_ENABLED` False: nothing here is used and the application
loads MODEL_PATH / NN_MODEL_PATH as it always did. True: `cfg.SELECTED_ML` /
`cfg.SELECTED_NN` are lists of zoo ids; one id = that model, several = their
soft vote. A selection that cannot be voted (different windows, labels or
calibration) is refused with the reason by `check_vote()`; it is never averaged
anyway.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from emg_app import config as cfg

STAR_TOP_N = 3
#  A model trained WITHOUT someone earns a blue star when it scores higher than
#  its yellow star on the same held-out people (`variant_stars`). The margin is
#  zero: any higher score, however small; the tooltip and the docs say how small.
BLUE_STAR_MIN_GAIN = 0.0
FAMILIES = ("ml", "nn")
ENSEMBLES_FILE = "ensembles.json"
META_FILE = "meta.json"


# ==============================================================================
# LOCATION
# ==============================================================================


def zoo_root(models_dir: str = None) -> str:
    """`<models dir>/zoo`, the models dir being the one MODEL_PATH lives in."""
    if models_dir is None:
        path = cfg.MODEL_PATH
        path = path if os.path.isabs(path) else os.path.join(cfg.PROJECT_DIR,
                                                             path)
        models_dir = os.path.dirname(path)
    return os.path.join(models_dir, "zoo")


def human_size(n) -> str:
    """Bytes as the unit a person would say."""
    if n is None:
        return "?"
    n = float(n)
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} GB"                                   # pragma: no cover


def slug(text: str) -> str:
    out = "".join(c.lower() if c.isalnum() else "-" for c in text)
    while "--" in out:
        out = out.replace("--", "-")
    return out.strip("-")


# ==============================================================================
# ENTRIES
# ==============================================================================


def variant_id(base_id: str, excluded) -> str:
    """
    The id of `base_id` trained WITHOUT some people: `ml-rf-500-without-barbora`.
    No people excluded = the base id itself.
    """
    people = sorted(set(excluded or ()))
    if not people:
        return base_id
    return f"{base_id}-without-" + "-".join(slug(p) for p in people)


@dataclass
class ZooEntry:
    id: str
    family: str                       # "ml" | "nn"
    dir: str
    meta: dict = field(default_factory=dict)

    @property
    def excluded(self) -> List[str]:
        """People whose data this model was trained WITHOUT (usually none)."""
        return [str(p) for p in (self.meta.get("excluded_persons") or [])]

    @property
    def name(self) -> str:
        return str(self.meta.get("name") or self.id)

    @property
    def kind(self) -> str:
        return str(self.meta.get("kind") or "")

    @property
    def file(self) -> str:
        return os.path.join(self.dir, self.meta.get(
            "file", "model.joblib" if self.family == "ml" else "model.pt"))

    @property
    def exists(self) -> bool:
        return os.path.exists(self.file)

    @property
    def scores(self) -> dict:
        return self.meta.get("scores") or {}

    @property
    def window_acc(self) -> Optional[float]:
        v = self.scores.get("window_acc")
        return None if v is None else float(v)

    @property
    def macro_f1(self) -> Optional[float]:
        v = self.scores.get("macro_f1")
        return None if v is None else float(v)

    @property
    def size_bytes(self) -> Optional[int]:
        """The size on disk, measured from the file when it is there."""
        try:
            return os.path.getsize(self.file)
        except OSError:
            v = self.meta.get("size_bytes")
            return None if v is None else int(v)

    @property
    def params(self) -> Optional[int]:
        v = self.meta.get("params")
        return None if v is None else int(v)

    @property
    def trained_with(self) -> dict:
        return self.meta.get("trained_with") or {}


def list_entries(root: str = None, family: str = None) -> List[ZooEntry]:
    """Every entry with a readable meta.json, sorted by family then id."""
    root = root or zoo_root()
    out = []
    if not os.path.isdir(root):
        return out
    for name in sorted(os.listdir(root)):
        folder = os.path.join(root, name)
        meta_path = os.path.join(folder, META_FILE)
        if not os.path.isfile(meta_path):
            continue
        try:
            with open(meta_path, encoding="utf-8") as fh:
                meta = json.load(fh)
        except (OSError, ValueError):
            continue
        if not isinstance(meta, dict) or meta.get("family") not in FAMILIES:
            continue
        entry = ZooEntry(id=str(meta.get("id") or name), family=meta["family"],
                         dir=folder, meta=meta)
        if family in (None, entry.family):
            out.append(entry)
    return sorted(out, key=lambda e: (e.family, e.id))


def get_entry(entry_id: str, root: str = None) -> Optional[ZooEntry]:
    for entry in list_entries(root):
        if entry.id == entry_id:
            return entry
    return None


def write_meta(folder: str, meta: dict) -> str:
    """Atomic write of one entry's meta.json."""
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, META_FILE)
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1, default=str)
    os.replace(tmp, path)
    return path


# ==============================================================================
# ENSEMBLES AND STARS
# ==============================================================================


def read_ensembles(root: str = None) -> dict:
    """`{"ml": {...}, "nn": {...}}` -- the best measured subset per family."""
    path = os.path.join(root or zoo_root(), ENSEMBLES_FILE)
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def write_ensembles(data: dict, root: str = None) -> str:
    root = root or zoo_root()
    os.makedirs(root, exist_ok=True)
    path = os.path.join(root, ENSEMBLES_FILE)
    tmp = path + ".partial"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    os.replace(tmp, path)
    return path


def rank(entries: List[ZooEntry]) -> List[ZooEntry]:
    """Best first: window accuracy, then macro-F1, then id (so it is stable).
    Entries with no score are last -- an unmeasured model is never starred."""
    def key(e):
        a, f = e.window_acc, e.macro_f1
        return (a is None, -(a or 0.0), -(f or 0.0), e.id)
    return sorted(entries, key=key)


def compute_stars(entries: List[ZooEntry], ensembles: dict = None) -> dict:
    """
    `{"models": {id: reason}, "ensembles": {family: reason}}` -- only starred
    things appear. Pure function of the entries' scores.
    """
    ensembles = ensembles if ensembles is not None else {}
    models, combos = {}, {}
    for fam in FAMILIES:
        #  A model trained without someone was scored on fewer people, so its
        #  number is not comparable with the rest and it is never ranked.
        fam_entries = [e for e in entries
                       if e.family == fam and e.exists and not e.excluded]
        ranked = [e for e in rank(fam_entries) if e.window_acc is not None]
        for place, e in enumerate(ranked[:STAR_TOP_N], 1):
            models[e.id] = (f"#{place} of {len(ranked)} {fam.upper()} models: "
                            f"{e.window_acc * 100:.1f} % per window"
                            + (f", macro-F1 {e.macro_f1 * 100:.1f} %"
                               if e.macro_f1 is not None else "")
                            + f", {human_size(e.size_bytes)}")
        combo = ensembles.get(fam)
        if combo and ranked:
            ids = list(combo.get("members") or [])
            present = {e.id for e in fam_entries}
            #  The star is decided on the NESTED estimate when there is one:
            #  the best subset was picked on the same folds it is scored on,
            #  which is optimistic; the nested number picks it without the
            #  held-out person.
            score = combo.get("nested_acc", combo.get("window_acc"))
            #  Compared with the best single model ON THE SAME BASIS: the
            #  search ran on one set of out-of-fold probabilities, so the
            #  single it must beat is that set's best single, not a score
            #  averaged over seeds (`best_single_acc`, written beside it).
            best_single = combo.get("best_single_acc", ranked[0].window_acc)
            if (score is not None and len(ids) >= 2
                    and all(i in present for i in ids)
                    and float(score) > best_single):
                combos[fam] = (
                    f"soft vote of {', '.join(ids)}: "
                    f"{float(score) * 100:.1f} % per window "
                    f"({'nested' if 'nested_acc' in combo else 'same folds'}),"
                    f" better than the best single model "
                    f"({best_single * 100:.1f} %)")
    return {"models": models, "ensembles": combos,
            "variants": variant_stars(entries, models)}


def variant_gain(entry: ZooEntry) -> Optional[float]:
    """
    How much better a trained-without variant is than the ordinary model it
    comes from, on the SAME held-out people: mean over the people it was
    scored on of (its score - the ordinary model's score for that person).
    `None` when it is not a variant or the comparison is not stored.
    """
    if not entry.excluded:
        return None
    mine = entry.scores.get("per_person") or {}
    base = entry.scores.get("base_per_person") or {}
    common = [p for p in mine if p in base]
    if not common:
        return None
    return sum(float(mine[p]) - float(base[p]) for p in common) / len(common)


def variant_reference(entry: ZooEntry, entries: List[ZooEntry],
                      yellow) -> Optional[ZooEntry]:
    """
    The yellow-star model a trained-without variant is judged against: the
    ordinary model it was made from when that one is starred, otherwise the
    best yellow star of its family. `None` when the family has no star.
    """
    by_id = {e.id: e for e in entries}
    base = by_id.get(entry.meta.get("variant_of"))
    if base is not None and base.exists and base.id in yellow:
        return base
    starred = rank([e for e in entries if e.family == entry.family
                    and e.exists and e.id in yellow])
    return starred[0] if starred else None


def variant_stars(entries: List[ZooEntry], yellow=None) -> dict:
    """
    `{id: reason}` of the BLUE stars, computed live from the scores: every
    trained-without variant that scores HIGHER than its yellow star on the
    people it was scored on (`variant_reference`). Same people on both sides,
    never the variant's headline score (that one is over fewer, easier
    people). Nothing is stored, so the stars follow the numbers whenever a
    model is retrained or a variant is added.
    """
    if yellow is None:
        yellow = compute_stars(entries, {})["models"]
    out = {}
    for e in entries:
        if not e.excluded or not e.exists:
            continue
        ref = variant_reference(e, entries, yellow)
        if ref is None:
            continue
        mine = e.scores.get("per_person") or {}
        theirs = ref.scores.get("per_person") or {}
        common = [p for p in mine if p in theirs]
        if not common:
            continue
        a = sum(float(mine[p]) for p in common) / len(common)
        b = sum(float(theirs[p]) for p in common) / len(common)
        if a - b > BLUE_STAR_MIN_GAIN:
            out[e.id] = (
                f"higher than its yellow star {ref.name}: {a * 100:.1f} % "
                f"against {b * 100:.1f} % ({(a - b) * 100:+.1f} points) on the "
                f"same {len(common)} people, trained without "
                f"{', '.join(e.excluded)} (docs/EXCLUDING_PEOPLE.md)")
    return out


def star_mark(entry: ZooEntry, stars: dict) -> str:
    return "*" if entry.id in stars.get("models", {}) else ""


# ==============================================================================
# VOTING COMPATIBILITY
# ==============================================================================


#  What two models must agree on before their probabilities can be averaged.
COMPAT_KEYS = {
    "ml": ("labels", "fs", "window_s", "window_samples", "representation",
           "feature_set", "freq_features", "calibration_mode",
           "adaptation_mode"),
    "nn": ("labels", "window_s", "win_samples", "n_channels",
           "calibration_mode"),
}


def check_vote(entries: List[ZooEntry]) -> Tuple[bool, str]:
    """
    Can these models be voted together? `(True, "")` or `(False, reason)`.
    One model is always fine; several must be one family and agree on
    everything in COMPAT_KEYS.
    """
    if not entries:
        return False, "nothing selected"
    families = {e.family for e in entries}
    if len(families) > 1:
        return False, ("ML models and neural nets are not voted together -- "
                       "select one family")
    fam = families.pop()
    missing = [e.id for e in entries if not e.exists]
    if missing:
        return False, f"no model file for {', '.join(missing)}"
    first = entries[0].meta.get("compat") or {}
    for other in entries[1:]:
        c = other.meta.get("compat") or {}
        for key in COMPAT_KEYS[fam]:
            if first.get(key) != c.get(key):
                return False, (f"{entries[0].name} and {other.name} disagree "
                               f"on {key} ({first.get(key)!r} against "
                               f"{c.get(key)!r}), so their probabilities "
                               f"cannot be averaged")
    return True, ""


def selection_summary(entries: List[ZooEntry]) -> dict:
    """What the GUI shows for a selection: count, total size and parameters."""
    sizes = [e.size_bytes for e in entries]
    params = [e.params for e in entries]
    return {"n": len(entries),
            "size_bytes": (sum(s for s in sizes if s is not None)
                           if entries else 0),
            "size_known": all(s is not None for s in sizes),
            "params": (sum(p for p in params if p is not None)
                       if any(p is not None for p in params) else None),
            "names": [e.name for e in entries]}


# ==============================================================================
# THE LIVE SELECTION
# ==============================================================================


def selected_entries(family: str, root: str = None) -> List[ZooEntry]:
    """The entries of the current selection for `family`, in selection order.
    Empty when selection is off or nothing valid is selected."""
    if not getattr(cfg, "MODEL_SELECTION_ENABLED", False):
        return []
    ids = list(cfg.SELECTED_ML if family == "ml" else cfg.SELECTED_NN)
    by_id = {e.id: e for e in list_entries(root, family)}
    return [by_id[i] for i in ids if i in by_id and by_id[i].exists]


def selected_paths(family: str, root: str = None) -> List[str]:
    """Absolute paths of the selected models; `[]` means "use the default"."""
    return [e.file for e in selected_entries(family, root)]


def describe_selection(family: str, root: str = None) -> str:
    """One line for the panels: what is actually being used."""
    entries = selected_entries(family, root)
    if not entries:
        return ""
    if len(entries) == 1:
        return entries[0].name
    return "soft vote: " + " + ".join(e.name for e in entries)
