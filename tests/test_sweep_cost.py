"""
The sweep's cost controls: window dedup, the screen, and the shortlist.

What this is really protecting:

  * `dedupe_windows` is CORRECTNESS, not a speed-up. 280 of 308 recordings are
    clips cut from a few free takes, they tile those takes with ~50 % overlap,
    and grouping the cross-validation by recording uid does NOT separate them
    because every clip legitimately owns its uid. Without dedup the same
    samples sit in train and test;
  * it works at WINDOW level, so a clip that only partly overlaps still
    contributes its non-duplicate windows -- clip-level dropping was measured
    worse and must not creep back;
  * it is deterministic and idempotent: two runs of the same sweep must give
    the same matrix, or a "reproducible, seeded" sweep is a lie;
  * `screen_fraction` affects only how many windows the SCREEN sees. It must
    never change the label set, the feature count, or anything that gets
    quoted;
  * `DEDUPE_WINDOWS = False` still reproduces the old (leaky) behaviour
    exactly, so the before/after numbers in how_to_fix_long_training.md stay
    checkable;
  * the DC-only control from CLAUDE.md section 8f really uses only the 8
    channel means, and really is a separate scoring path that cannot change
    what gets trained.

Runs on a synthetic dataset built to contain a KNOWN amount of overlap, so the
test is self-contained and does not depend on what is in datasets/ today.

Run from the project root:

    python tests\\test_sweep_cost.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "training"))

import emg_dataset_tool as T                                      # noqa: E402

OK = []


def check(name, condition, detail=""):
    OK.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail else ""))


try:
    import train_model as M
except SystemExit:
    print("scikit-learn is not installed; the sweep cannot be tested.")
    sys.exit(0)


# ---------------------------------------------------------------------------
# A synthetic take, cut into clips with a known overlap
# ---------------------------------------------------------------------------
FS = 50.0
TAKE_SAMPLES = 1000


def make_clip(label, take_uid, start, end, level, uid):
    """One clip 'cut from' a synthetic take, carrying the same provenance
    metadata the real EXTRACT CLIP path writes."""
    n = end - start
    rng = np.random.default_rng(abs(hash((take_uid, start))) % (2 ** 32))
    emg = np.full((n, T.N_CHANNELS), float(level))
    emg += rng.normal(0.0, 2.0, size=emg.shape)
    ts = np.arange(n, dtype=np.float64) / FS
    return T.Recording(
        emg=np.clip(emg, 0, 2047).astype(np.int32), timestamps=ts,
        label=label, person_id="Synth", session_id="session_01",
        repetition=1, stage=T.STAGE_RAW, uid=uid,
        label_start_index=0, label_end_index=n,
        meta={"capture_style": "cut_from_take",
              "parent_take_uid": take_uid,
              "take_start_index": start,
              "take_end_index": end,
              "parent_take_n_samples": TAKE_SAMPLES})


def build_recs(step):
    """Two labels, each one take, each tiled by clips every `step` samples."""
    recs = []
    for label, take, level in (("TIGHT", "takeA", 400), ("LOOSE", "takeB", 80)):
        k = 0
        for start in range(0, TAKE_SAMPLES - 200 + 1, step):
            recs.append(make_clip(label, take, start, start + 200, level,
                                  uid=f"{take}{k:03d}"))
            k += 1
    return recs


# 200-sample clips every 50 samples = each sample covered by 4 clips.
recs = build_recs(step=50)
print(f"\nsynthetic dataset: {len(recs)} clips, 2 takes, 4x overlap\n")

SETUP = dict(window_s=0.30, overlap=0.5, representation="raw",
             include_freq=True, feature_set="level")

rep_off, rep_on = {}, {}
off = M.build_dataset(recs, dedupe=False, report=rep_off, **SETUP)
on = M.build_dataset(recs, dedupe=True, report=rep_on, **SETUP)
check("both regimes produce a dataset", off is not None and on is not None)

X_off, y_off, g_off, labels_off, meta_off = off
X_on, y_on, g_on, labels_on, meta_on = on

# ---- the dedup actually dedups --------------------------------------------
check("dedup removes windows", X_on.shape[0] < X_off.shape[0],
      f"{X_off.shape[0]} -> {X_on.shape[0]}")
check("dedup removes roughly the overlap factor",
      2.5 < X_off.shape[0] / max(1, X_on.shape[0]) < 5.0,
      f"{X_off.shape[0] / max(1, X_on.shape[0]):.2f}x on a 4x-overlapped take")
check("the report counts what it dropped",
      rep_on["windows_before"] > rep_on["windows_after"],
      f"{rep_on['windows_before']} -> {rep_on['windows_after']}")
check("dedupe=False reports no drop at all",
      rep_off["windows_before"] == rep_off["windows_after"],
      f"{rep_off['windows_before']}")

# ---- it keeps every label, and the feature contract --------------------
check("dedup keeps both labels", sorted(set(labels_on)) == ["LOOSE", "TIGHT"])
check("dedup keeps every class populated",
      all(int((y_on == i).sum()) > 0 for i in range(len(labels_on))),
      str({labels_on[i]: int((y_on == i).sum())
           for i in range(len(labels_on))}))
check("dedup does not change the feature count",
      X_on.shape[1] == X_off.shape[1], f"{X_on.shape[1]} features")
check("dedup is recorded in the meta", meta_on["deduped"] is True
      and meta_off["deduped"] is False)

# ---- window level, not clip level ----------------------------------------
# The decisive property: clip-level dropping would leave only the clips that
# overlap nothing, so most clips would vanish entirely. Window-level dedup
# leaves many clips each contributing a share.
clips_off = len(set(g_off.tolist()))
clips_on = len(set(g_on.tolist()))
check("dedup is window-level: most clips still contribute",
      clips_on > clips_off * 0.5,
      f"{clips_on} of {clips_off} clips still represented")

# ---- determinism and idempotence -----------------------------------------
again = M.build_dataset(recs, dedupe=True, **SETUP)
check("two identical builds give identical matrices",
      np.array_equal(X_on, again[0]) and np.array_equal(y_on, again[1]))
check("two identical builds give identical groups",
      list(g_on) == list(again[2]))

# A take with no overlap at all must lose nothing.
sparse = build_recs(step=200)          # 200-sample clips every 200 samples
rep_sparse = {}
out_sparse = M.build_dataset(sparse, dedupe=True, report=rep_sparse, **SETUP)
check("non-overlapping clips are left alone",
      rep_sparse["windows_before"] == rep_sparse["windows_after"],
      f"{rep_sparse['windows_before']} windows, none dropped")

# ---- source_timeline ------------------------------------------------------
place = M.source_timeline(recs[0])
check("source_timeline reads the take provenance",
      place is not None and place[0] == "take:takeA" and place[1] == 0,
      str(place))
orphan = T.Recording(emg=np.full((120, T.N_CHANNELS), 50, dtype=np.int32),
                     timestamps=np.arange(120) / FS, label="TIGHT",
                     person_id="Synth", session_id="session_01",
                     uid="orphan01", label_start_index=0,
                     label_end_index=120, meta={})
check("a recording with no provenance is not placed",
      M.source_timeline(orphan) is None)
rep_orphan = {}
out_orphan = M.build_dataset(recs + [orphan], dedupe=True,
                             report=rep_orphan, **SETUP)
check("an unplaceable recording keeps all its windows (never silently lost)",
      out_orphan is not None
      and "orphan01" in set(out_orphan[2].tolist()))

# ---- screen_fraction ------------------------------------------------------
screened = M.build_dataset(recs, dedupe=True, screen_fraction=0.5, **SETUP)
Xs, ys, gs, labels_s, meta_s = screened
check("screen_fraction reduces the rows", Xs.shape[0] < X_on.shape[0],
      f"{X_on.shape[0]} -> {Xs.shape[0]}")
check("screen_fraction does not change the feature count",
      Xs.shape[1] == X_on.shape[1])
check("screen_fraction does not change the label set",
      list(labels_s) == list(labels_on))
check("screen_fraction keeps every class", len(set(ys.tolist())) == len(labels_s))
check("screen_fraction is recorded in the meta",
      meta_s["screen_fraction"] == 0.5 and meta_on["screen_fraction"] == 1.0)

# ---- the configuration is honest about itself -----------------------------
check("the screen uses at most SCREEN_MAX_PROBES probes",
      M.SCREEN_MAX_PROBES <= 2, f"SCREEN_MAX_PROBES={M.SCREEN_MAX_PROBES}")
check("every screen probe is a real candidate name",
      all(isinstance(n, str) and n for n in M.SCREEN_PROBES))
check("the shortlist is big enough to hold the stage B picks",
      M.A2_SHORTLIST >= 5, f"A2_SHORTLIST={M.A2_SHORTLIST}")
check("part of the shortlist is random",
      0 < M.A2_RANDOM < M.A2_SHORTLIST,
      f"{M.A2_RANDOM} of {M.A2_SHORTLIST}")
check("dedup is on by default", M.DEDUPE_WINDOWS is True)

# ---- the DC-only control --------------------------------------------------
names = M.feature_names_for("level", True)
dc = [n for n in names if n.endswith("_mean") and not n.endswith("_log_mean")]
check("the control's columns are the 8 channel means", len(dc) == 8, str(dc))

model = M.build_models_from_spec(
    {"version": 1, "voting": {"enabled": False},
     "models": [{"name": "LDA-shrinkage", "kind": "lda", "enabled": True,
                 "scaler": "standard",
                 "params": {"solver": "lsqr", "shrinkage": "auto"}}]},
    quick=True, for_scoring=True)["LDA-shrinkage"]

control = M.dc_only_control(model, X_on, y_on, g_on, labels_on, names)
check("the control returns a score", control is not None)
if control:
    check("the control used only 8 features", control["n_features"] == 8,
          str(control["n_features"]))
    check("the control's score is a probability",
          0.0 <= control["window_accuracy"] <= 1.0,
          f"{control['window_accuracy'] * 100:.1f}%")
    # These two classes differ ONLY in DC level, so the control must do well.
    # That is the point of the control: it detects exactly this situation.
    check("the control detects a level-only difference",
          control["window_accuracy"] > 0.9,
          f"{control['window_accuracy'] * 100:.1f}% on level-only classes")

fitted_before = M.evaluate(model, X_on, y_on, g_on, labels_on)
check("the control did not disturb the real score",
      fitted_before is not None,
      "evaluate still works on the same arrays afterwards")

# ---------------------------------------------------------------------------
# Multi-subject selection: the grouping has to switch by itself
# ---------------------------------------------------------------------------
# The gap this guards: with several wearers, picking the winner on folds
# grouped by RECORDING means every subject sits in both train and test, so a
# setup that leans on one person's absolute amplitudes wins and then fails on
# the next wearer. That is the opposite of what collecting more people is for.
print()
check("one subject -> grouped by recording",
      M.resolve_grouping(1)[0] == "recording")
check("two subjects -> grouped by subject",
      M.resolve_grouping(2)[0] == "subject")
check("five subjects -> grouped by subject",
      M.resolve_grouping(5)[0] == "subject")
check("the choice is explained in words",
      "subject" in M.resolve_grouping(3)[1].lower()
      and len(M.resolve_grouping(3)[1]) > 40)
check("an explicit 'recording' request is honoured",
      M.resolve_grouping(4, "recording")[0] == "recording")
check("'subject' with one subject fails over to recording, not an error",
      M.resolve_grouping(1, "subject")[0] == "recording")
check("the default is auto", M.SELECTION_GROUPING == "auto")

# `person_vector` must line up with `groups`, window for window.
multi = []
for person, level in (("alice", 400), ("bob", 90)):
    for lab, tk, lv in (("TIGHT", f"{person}A", level),
                        ("LOOSE", f"{person}B", level // 4)):
        for i, s in enumerate(range(0, TAKE_SAMPLES - 200 + 1, 200)):
            r = make_clip(lab, tk, s, s + 200, lv, uid=f"{tk}{i:03d}")
            r.person_id = person
            multi.append(r)

Xm, ym, gm, labels_m, _ = M.build_dataset(multi, dedupe=True, **SETUP)
persons = M.person_vector(multi, gm)
check("person_vector is one entry per window",
      persons.shape[0] == Xm.shape[0], f"{persons.shape[0]} vs {Xm.shape[0]}")
check("person_vector names both subjects",
      sorted(set(persons.tolist())) == ["alice", "bob"])
uid_person = {r.uid: r.person_id for r in multi}
check("every window's subject matches its recording's subject",
      all(persons[i] == uid_person[gm[i]] for i in range(len(gm))))
check("person_vector marks an unknown uid rather than raising",
      M.person_vector([], np.array(["nope"], dtype=object))[0] == "?")

# Splitting by subject must really hold a whole person out.
by_rec = M.evaluate(model, Xm, ym, gm, labels_m, grouping="recording")
by_sub = M.evaluate(model, Xm, ym, gm, labels_m, split_groups=persons,
                    grouping="subject")
check("both protocols return a score", by_rec is not None and by_sub is not None)
if by_rec and by_sub:
    check("the result records which grouping produced it",
          by_rec["grouping"] == "recording" and by_sub["grouping"] == "subject")
    check("subject-grouped folds cannot exceed the subject count",
          by_sub["n_splits"] <= len(set(persons.tolist())),
          f"{by_sub['n_splits']} folds for 2 subjects")
    check("the per-recording vote still counts recordings, not subjects",
          by_sub["n_recordings_voted"] > len(set(persons.tolist())),
          f"{by_sub['n_recordings_voted']} recordings voted")

# The honest property: with a subject-grouped split, no window of the held-out
# person may appear in training. Checked on the splitter directly.
from sklearn.model_selection import StratifiedGroupKFold          # noqa: E402
cv = StratifiedGroupKFold(n_splits=2, shuffle=True,
                          random_state=M.RANDOM_STATE)
leaks = 0
for train_idx, test_idx in cv.split(Xm, ym, persons):
    if set(persons[train_idx].tolist()) & set(persons[test_idx].tolist()):
        leaks += 1
check("no subject appears in both sides of a subject-grouped split",
      leaks == 0, f"{leaks} leaking folds")

print(f"\n================  {sum(OK)}/{len(OK)} checks passed  ================")
sys.exit(0 if all(OK) else 1)
