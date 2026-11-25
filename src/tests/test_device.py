"""
The device profile, the label set, and the recording plan.

Three things added 2026-10-19, and the first is the one with teeth:
`N_CHANNELS` used to be a bare 8 read at 44 sites, so the whole dataset
depended on a literal. It is now derived from `DEVICE_PROFILES[EMG_DEVICE]`,
and the rule that makes that safe is checked here rather than described:

    ANYTHING HOLDING REAL DATA FOLLOWS THE DATA, NOT THE CONFIG.

If that ever stops being true, changing one config line silently makes every
existing recording unreadable -- which is why the round trips below are run
at 4, 8, 12 and 16 channels *while the config still says 8*.

Run from the project root:

    python tests\\test_device.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import emg_dataset_tool as T                                       # noqa: E402

OK = []
TEMP = tempfile.mkdtemp(prefix="emgdev_")


def check(name, condition, detail=""):
    OK.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail else ""))


def section(title):
    print(f"\n--- {title} ---")


def make(width, n=60, seed=0, label="FIST", person="p", session="s1"):
    rng = np.random.default_rng(seed)
    emg = np.abs(rng.normal(200, 30, (n, width))).astype(int)
    t = np.arange(n) / 50.0 + 1_000_000_000.0
    return T.Recording(emg=emg, timestamps=t, label=label, person_id=person,
                       session_id=session, repetition=1)


# ==============================================================================
section("1. the profile block")
# ==============================================================================

check("there is a profile for the Myo", "myo" in T.DEVICE_PROFILES)
check("the active profile is the Myo by default", T.EMG_DEVICE == "myo",
      T.EMG_DEVICE)
check("N_CHANNELS is DERIVED from it, not a literal",
      T.N_CHANNELS == T.DEVICE_PROFILES["myo"]["channels"] == 8,
      str(T.N_CHANNELS))
check("CHANNEL_NAMES is sized to it",
      len(T.CHANNEL_NAMES) == T.N_CHANNELS
      and T.CHANNEL_NAMES[0] == "CH1" and T.CHANNEL_NAMES[-1] == "CH8")
check("every profile declares a channel count",
      all(int(p["channels"]) > 0 for p in T.DEVICE_PROFILES.values()))

try:
    T.set_device("generic_16")
    check("set_device switches the active profile", T.EMG_DEVICE == "generic_16")
    check("...and recomputes N_CHANNELS", T.N_CHANNELS == 16, str(T.N_CHANNELS))
    check("...and the channel names", len(T.CHANNEL_NAMES) == 16
          and T.CHANNEL_NAMES[-1] == "CH16")
    check("...and the feature-name length follows",
          len(T.model_feature_names("level", True)) == 98,
          str(len(T.model_feature_names("level", True))))
    probe = np.abs(np.random.default_rng(1).normal(200, 30, (30, 16)))
    vector = T.model_feature_vector(probe, "raw", "level", True, 60.0)
    check("...and the vector matches its own names at the new width",
          len(vector) == len(T.model_feature_names("level", True)),
          f"{len(vector)}")
    ring = T.SampleRing(64)
    check("a ring created after the switch is the new width",
          ring.snapshot(4)[0].shape[1] == 16,
          str(ring.snapshot(4)[0].shape))
finally:
    T.set_device("myo")
check("switching back restores 8", T.N_CHANNELS == 8 and T.EMG_DEVICE == "myo")

try:
    T.set_device("no_such_band")
    check("an unknown device is refused", False, "it was accepted")
except ValueError as exc:
    check("an unknown device is refused with the known names listed",
          "no_such_band" in str(exc) and "myo" in str(exc))
check("...and the active profile is unchanged after the refusal",
      T.EMG_DEVICE == "myo" and T.N_CHANNELS == 8)

check("channel_names_for() sizes captions to real data",
      len(T.channel_names_for(16)) == 16
      and T.channel_names_for(16)[-1] == "CH16")
check("...and matches CHANNEL_NAMES at the active width",
      T.channel_names_for(8) == list(T.CHANNEL_NAMES))


# ==============================================================================
section("2. a Recording follows its OWN width, not the config")
# ==============================================================================

for width in (4, 8, 12, 16):
    rec = make(width)
    check(f"a {width}-channel array keeps {width} channels",
          rec.n_channels == width and rec.emg.shape[1] == width,
          str(rec.emg.shape))
    check(f"...and its captions are sized to {width}",
          len(rec.channel_names) == width)
check("the config is still 8 throughout -- the data won, not the config",
      T.N_CHANNELS == 8)

flat = T.Recording(emg=np.arange(24, dtype=int), timestamps=np.arange(3) / 50.0,
                   label="FIST", person_id="p", session_id="s1")
check("a FLAT array folds to the config width -- nothing else can say",
      flat.n_channels == 8 and flat.emg.shape == (3, 8), str(flat.emg.shape))
flat6 = T.Recording(emg=np.arange(18, dtype=int), timestamps=np.arange(3) / 50.0,
                    label="FIST", person_id="p", session_id="s1",
                    n_channels=6)
check("...unless the caller says otherwise", flat6.emg.shape == (3, 6),
      str(flat6.emg.shape))


# ==============================================================================
section("3. round trip on disk, at four widths, config still 8")
# ==============================================================================

for width in (4, 8, 12, 16):
    root = os.path.join(TEMP, f"w{width}")
    writer = T.DatasetWriter(root, "p", "s1")
    rec = make(width, seed=width)
    written = writer.save(rec)

    back_npz = T.load_recording(written["npz"])
    check(f"{width:>2}ch npz: width survives", back_npz.n_channels == width,
          str(back_npz.n_channels))
    check(f"{width:>2}ch npz: samples bit-identical",
          np.array_equal(back_npz.emg, rec.emg))

    back_csv = T.load_recording(written["csv"])
    check(f"{width:>2}ch csv: width survives", back_csv.n_channels == width,
          str(back_csv.n_channels))
    check(f"{width:>2}ch csv: samples bit-identical",
          np.array_equal(back_csv.emg, rec.emg))

    with open(written["meta"], encoding="utf-8") as fh:
        meta = json.load(fh)
    check(f"{width:>2}ch metadata reports the real count",
          meta["n_channels"] == width, str(meta["n_channels"]))
    check(f"{width:>2}ch metadata names all of them",
          len(meta["channel_names"]) == width)

    X, _y, _g, _m = T.make_windows([back_npz])
    check(f"{width:>2}ch make_windows third dim follows the data",
          X.shape[2] == width, str(X.shape))

check("csv_header() is parameterised by width",
      len(T.csv_header(16)) - len(T.csv_header(8)) == 8,
      f"{len(T.csv_header(8))} vs {len(T.csv_header(16))}")

#  The ordering trap: a plain lexical sort puts ch10 between ch1 and ch2, so
#  a 12-channel file would load with its columns permuted -- silently, and
#  only above nine channels.
header = T.csv_header(12)
channels = [c for c in header if c.startswith("ch")]
check("csv_header orders channels numerically, not lexically",
      channels == [f"ch{i + 1}" for i in range(12)],
      ", ".join(channels[:4] + ["..."] + channels[-2:]))

root = os.path.join(TEMP, "perm")
writer = T.DatasetWriter(root, "p", "s1")
rec12 = make(12, seed=99)
written12 = writer.save(rec12)
reloaded = T.load_recording(written12["csv"])
check("a 12-channel CSV reloads with its columns in the right order",
      np.array_equal(reloaded.emg, rec12.emg))
check("...and column 10 is really column 10, not column 2",
      np.array_equal(reloaded.emg[:, 9], rec12.emg[:, 9]))


# ==============================================================================
section("4. features match their names at every width")
# ==============================================================================

for width in (4, 8, 12, 16):
    window = np.abs(np.random.default_rng(width).normal(200, 30, (30, width)))
    for feature_set in ("level", "both", "classic"):
        vector = T.model_feature_vector(window, "raw", feature_set, True, 60.0)
        names = T.model_feature_names(feature_set, True, width)
        check(f"{width:>2}ch {feature_set:<8} vector and names agree",
              len(vector) == len(names), f"{len(vector)} vs {len(names)}")
check("the name generators default to the active profile",
      len(T.model_feature_names("level", True)) == 50,
      str(len(T.model_feature_names("level", True))))


# ==============================================================================
section("5. calibration is sized by the RECORD, and refuses a mismatch")
# ==============================================================================

for width in (4, 8, 16):
    record = T.calibration_record([20] * width, [600] * width, person_id="p",
                                  n_rest=200, n_effort=200, fs=50.0)
    ok, problems = T.validate_calibration(record)
    check(f"a {width}-channel calibration validates", ok, str(problems))
    resolved = T.resolve_calibration(record, "affine")
    check(f"...and resolves to a {width}-wide map",
          resolved is not None and np.size(resolved["offset"]) == width,
          str(np.size(resolved["offset"]) if resolved else None))
    block = np.full((10, width), 300.0)
    out = T.apply_calibration(block, resolved)
    check(f"...and applying it changes a {width}-wide window",
          not np.allclose(out, block))

eight = T.resolve_calibration(
    T.calibration_record([20] * 8, [600] * 8, person_id="p", n_rest=200,
                         n_effort=200, fs=50.0), "affine")
wide = np.full((10, 16), 300.0)
out = T.apply_calibration(wide, eight)
check("an 8-wide map on a 16-wide window is REFUSED, not broadcast",
      np.allclose(out, wide), "the block came back untouched")
narrow = np.full((10, 4), 300.0)
check("...and the same the other way round",
      np.allclose(T.apply_calibration(narrow, eight), narrow))
check("a mismatched record resolves/degrades rather than raising",
      T.resolve_calibration({"rest": [1, 2], "reference": [3]},
                            "affine") is None)
check("unequal level vectors are rejected outright",
      T.calibration_levels({"rest": [20] * 8, "reference": [600] * 4}) is None)

#  The quiet-channel message has to name real channels, not CH1..CH8 blindly.
bad = T.calibration_record([100] * 12, [600] * 11 + [50], person_id="p",
                           n_rest=200, n_effort=200, fs=50.0)
ok, problems = T.validate_calibration(bad)
check("a 12-channel record names the offending channel by its own index",
      not ok and any("CH12" in p for p in problems),
      "; ".join(problems)[:70])


# ==============================================================================
section("6. the label set")
# ==============================================================================

check("the optional gestures are NOT in the default set",
      not set(T.OPTIONAL_GESTURES) & set(T.GESTURES),
      ", ".join(T.GESTURES))
for name in ("PINCH", "THUMB_OVER_INDEX", "POINT"):
    check(f"{name} is available as an option", name in T.OPTIONAL_GESTURES)
    check(f"...and carries an instruction, not just a name",
          len(T.gesture_cue(name)) > 10, T.gesture_cue(name))
check("known_gestures() is the union, in-use first",
      T.known_gestures()[:len(T.GESTURES)] == list(T.GESTURES)
      and set(T.known_gestures()) == set(T.GESTURES) | set(T.OPTIONAL_GESTURES))
check("every built-in gesture has a cue too",
      all(T.gesture_cue(g) for g in T.GESTURES),
      str([g for g in T.GESTURES if not T.gesture_cue(g)]))
check("an unknown gesture's cue is empty rather than an error",
      T.gesture_cue("NOT_A_GESTURE") == "")


# ==============================================================================
section("7. the recording plan")
# ==============================================================================

plan = T.build_plan(["REST", "FIST", "OPEN"], repetitions=5, order="blocked")
labels = [g for g, _r in plan]
check("blocked order groups every repetition of one gesture together",
      labels == ["REST"] * 5 + ["FIST"] * 5 + ["OPEN"] * 5,
      ", ".join(labels[:7]) + " ...")
check("blocked order numbers the repetitions 1..n within each gesture",
      [r for _g, r in plan][:5] == [1, 2, 3, 4, 5])

inter = [g for g, _r in T.build_plan(["REST", "FIST", "OPEN"], repetitions=5,
                                     order="interleaved")]
check("interleaved order does NOT group them",
      inter != labels and len(inter) == 15)
check("...and covers every gesture in the first round",
      set(inter[:3]) == {"REST", "FIST", "OPEN"}, ", ".join(inter[:3]))

check("a plan with an optional gesture switched on includes it",
      "PINCH" in [g for g, _r in T.build_plan(
          list(T.GESTURES) + ["PINCH"], repetitions=2, order="blocked")])

check("the hold time is the 2.5 s asked for, not the old 4.0",
      abs(T.RECORD_TIME - 2.5) < 1e-9, str(T.RECORD_TIME))
check("a repetition is still longer than the model's window",
      T.RECORD_TIME > T.WINDOW_MS / 1000.0,
      f"{T.RECORD_TIME} s vs {T.WINDOW_MS} ms")
check("extra signal is still captured around the labelled window",
      T.PRE_GESTURE_MS > 0 and T.POST_GESTURE_MS > 0)


shutil.rmtree(TEMP, ignore_errors=True)
print(f"\n================  {sum(OK)}/{len(OK)} checks passed  "
      f"================")
sys.exit(0 if all(OK) else 1)
