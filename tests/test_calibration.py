# -*- coding: utf-8 -*-
"""
Checks for per-wearer calibration (CLAUDE.md section 8k).

No hardware, no armband, no trained model needed. Five subjects:

  1. the TRANSFORM -- does each mode do what it claims? The important one is
     that "gain" leaves the spatial profile exactly proportional, because
     that is the whole reason the conservative mode exists.

  2. REFUSAL -- a calibration whose "maximum effort" capture is really rest
     would divide a wearer's signal by noise. Unlike a model spec, which gets
     coerced, a bad calibration has to be refused, and it must be refused
     before it reaches the disk.

  3. STORAGE -- per wearer, append-only history, and clearing keeps the
     record in the log. Also: adding a calibration must not move the dataset
     fingerprint, or every model on disk would instantly read "stale".

  4. the FEATURE PATH -- one function applies it (model_feature_vector), the
     trainer and the live predictor both go through that function, and a
     calibrated model REFUSES to predict for a wearer who has none.

  5. the GUI -- the button, the status line, and the derive-from-recordings
     path that needs no armband.

Run:  python tests\\test_calibration.py
"""
import json
import os
import shutil
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT = os.path.dirname(HERE)
if PROJECT not in sys.path:
    sys.path.insert(0, PROJECT)

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import emg_dataset_tool as T                                    # noqa: E402

PASS = 0
FAIL = 0
FS = 50.0
TMP = []


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"  ok    {name}")
    else:
        FAIL += 1
        print(f"  FAIL  {name}" + (f"   {detail}" if detail else ""))


def section(title):
    print(f"\n{title}")
    print("-" * len(title))


def tmpdir(prefix="emgcal_"):
    path = tempfile.mkdtemp(prefix=prefix)
    TMP.append(path)
    return path


def good_record(person="Ann", rest=20.0, ref=None):
    """A record that validates: flat rest, a clear per-channel reference."""
    ref = ref if ref is not None else [200, 300, 400, 500, 600, 700, 800, 900]
    rest_block = np.full((60, T.N_CHANNELS), rest)
    effort_block = np.tile(np.asarray(ref, dtype=float), (60, 1))
    return T.calibration_from_captures(rest_block, effort_block,
                                       person_id=person, fs=FS)


# ==============================================================================
section("1. the transform -- what each mode actually does")
# ==============================================================================
rec = good_record()
ok, problems = T.validate_calibration(rec)
check("a clean two-capture record validates", ok, str(problems))
check("rest is the median of the rest capture",
      np.allclose(rec["rest"], 20.0), str(rec["rest"]))
check("reference is the high percentile of the effort capture",
      np.allclose(rec["reference"],
                  [200, 300, 400, 500, 600, 700, 800, 900]),
      str(rec["reference"]))
check("contrast is reported", abs(rec["contrast"] - 550.0 / 20.0) < 1e-6,
      f"{rec['contrast']:.3f}")

check("mode 'off' resolves to None -- the uncalibrated pipeline",
      T.resolve_calibration(rec, "off") is None)
check("an unknown mode resolves to None rather than guessing",
      T.resolve_calibration(rec, "sideways") is None)
check("a record of the wrong shape resolves to None",
      T.resolve_calibration({"rest": [1, 2], "reference": [3, 4]},
                            "per_channel") is None)
check("apply_calibration with None is the identity",
      T.apply_calibration(np.arange(16.0).reshape(2, 8), None).tolist()
      == np.arange(16.0).reshape(2, 8).tolist())

window = np.tile(np.asarray([40, 80, 120, 160, 200, 240, 280, 320],
                            dtype=float), (5, 1))

#  --- "gain": one divisor, so every ratio between channels survives --------
cal = T.resolve_calibration(rec, "gain")
out = T.apply_calibration(window, cal)
check("gain mode uses ONE divisor for all eight channels",
      len(set(np.round(cal["scale"], 9).tolist())) == 1,
      str(cal["scale"]))
check("gain mode applies no offset", np.allclose(cal["offset"], 0.0))
ratio_in = window[0] / window[0].sum()
ratio_out = out[0] / out[0].sum()
check("gain mode leaves the spatial profile BIT-FOR-BIT proportional -- the "
      "point of the conservative mode",
      np.allclose(ratio_in, ratio_out),
      f"{np.round(ratio_in, 6)} vs {np.round(ratio_out, 6)}")
check("gain mode scales by target / mean(reference)",
      np.allclose(out[0], window[0] * T.CALIBRATION_TARGET_LEVEL / 550.0),
      str(np.round(out[0], 2)))

#  --- "per_channel": each channel divided by its own reference -------------
cal = T.resolve_calibration(rec, "per_channel")
out = T.apply_calibration(window, cal)
check("per_channel uses a different divisor per channel",
      len(set(np.round(cal["scale"], 9).tolist())) == T.N_CHANNELS,
      str(cal["scale"]))
check("per_channel maps a reference-level window onto the target exactly",
      np.allclose(T.apply_calibration(
          np.tile(np.asarray(rec["reference"], dtype=float), (3, 1)), cal),
          T.CALIBRATION_TARGET_LEVEL))
check("per_channel does NOT preserve the spatial profile (it equalises it)",
      not np.allclose(out[0] / out[0].sum(), ratio_in))

#  --- "affine": per channel, with the wearer's rest removed ----------------
cal = T.resolve_calibration(rec, "affine")
check("affine subtracts the wearer's rest level",
      np.allclose(cal["offset"], 20.0), str(cal["offset"]))
check("affine maps rest to 0",
      np.allclose(T.apply_calibration(
          np.full((3, T.N_CHANNELS), 20.0), cal), 0.0))
check("affine maps the reference to the target",
      np.allclose(T.apply_calibration(
          np.tile(np.asarray(rec["reference"], dtype=float), (3, 1)), cal),
          T.CALIBRATION_TARGET_LEVEL))
below = T.apply_calibration(np.full((3, T.N_CHANNELS), 5.0), cal)
check("affine floors below-rest signal at 0 -- a negative channel would make "
      "`share` meaningless, not merely noisy",
      float(below.min()) == 0.0, str(below.min()))

check("the target level is what the config says",
      abs(T.resolve_calibration(rec, "gain")["target"]
          - T.CALIBRATION_TARGET_LEVEL) < 1e-9)
check("a hand-edited record cannot ask for an unbounded gain (scale floors "
      "at one device count)",
      float(np.min(T.resolve_calibration(
          {"rest": [0] * 8, "reference": [0.0001] * 8,
           "n_rest_samples": 999, "n_effort_samples": 999},
          "per_channel")["scale"]
          if T.resolve_calibration(
              {"rest": [0] * 8, "reference": [0.0001] * 8},
              "per_channel") else [1.0])) >= 1.0
      or T.resolve_calibration({"rest": [0] * 8, "reference": [0.0001] * 8},
                               "per_channel") is None)


# ==============================================================================
section("2. refusal -- a bad calibration never reaches the disk")
# ==============================================================================
flat = T.calibration_from_captures(np.full((60, 8), 20.0),
                                   np.full((60, 8), 22.0),
                                   person_id="Flat", fs=FS)
ok, problems = T.validate_calibration(flat)
check("an effort capture that is really rest is REFUSED", not ok)
check("the reason names the reference level",
      any("CALIBRATION_MIN_REFERENCE" in p for p in problems), str(problems))
check("a refused record resolves to None in every mode",
      all(T.resolve_calibration(flat, m) is None
          for m in T.CALIBRATION_MODES))

low_contrast = T.calibration_from_captures(
    np.full((60, 8), 300.0), np.full((60, 8), 450.0),
    person_id="Low", fs=FS)
ok, problems = T.validate_calibration(low_contrast)
check("a reference only 1.5x rest is refused on contrast", not ok)
check("the reason names the contrast floor",
      any("CALIBRATION_MIN_CONTRAST" in p for p in problems), str(problems))

thin = T.calibration_from_captures(np.full((3, 8), 20.0),
                                   np.full((3, 8), 600.0),
                                   person_id="Thin", fs=FS)
ok, problems = T.validate_calibration(thin)
check("too few samples in a capture is refused", not ok)
check("the reason names the sample floor",
      any("CALIBRATION_MIN_SAMPLES" in p for p in problems), str(problems))

one_dead = good_record(ref=[200, 300, 400, 500, 600, 700, 800, 10.0])
ok, problems = T.validate_calibration(one_dead)
check("a channel whose reference is below its rest is refused", not ok)
check("the reason names the channel",
      any("CH8" in p for p in problems), str(problems))

check("garbage is refused rather than raising",
      T.validate_calibration({"rest": "nope"})[0] is False)
check("None is refused rather than raising",
      T.validate_calibration(None)[0] is False)
check("validate never raises on a record with NaN",
      T.validate_calibration(
          {"rest": [float("nan")] * 8, "reference": [600.0] * 8})[0] is False)
check("describe_calibration says 'no calibration' for None",
      "no calibration" in T.describe_calibration(None, "gain"))
check("describe_calibration says REFUSED for a bad record",
      "REFUSED" in T.describe_calibration(flat, "gain"))
check("describe_calibration says NOT APPLIED when the mode is off",
      "NOT APPLIED" in T.describe_calibration(rec, "off"))


# ==============================================================================
section("3. storage -- per wearer, append-only, fingerprint-neutral")
# ==============================================================================
root = tmpdir()
check("no calibration is the normal case, not an error",
      T.load_calibration(root, "Ann") is None)
check("status reports 'none' with the wearer named",
      T.calibration_status(root, "Ann", "gain")["state"] == "none")

path = T.save_calibration(good_record("Ann"), root, "Ann")
check("the record lands in the WEARER's folder, not next to the model",
      os.path.normpath(path)
      == os.path.normpath(os.path.join(root, "Ann", T.CALIBRATION_NAME)),
      path)
check("it reads back", T.load_calibration(root, "Ann") is not None)
check("it reads back as the same numbers",
      T.load_calibration(root, "Ann")["reference"]
      == good_record("Ann")["reference"])
check("person_id is stamped into the stored record",
      T.load_calibration(root, "Ann")["person_id"] == "Ann")
check("status is 'stored' while the mode is off",
      T.calibration_status(root, "Ann", "off")["state"] == "stored")
check("status is 'active' once a mode is chosen",
      T.calibration_status(root, "Ann", "affine")["state"] == "active")

T.save_calibration(good_record("Ann", ref=[300] * 8), root, "Ann")
log = T.load_calibration_log(root, "Ann")
check("the log has one line per calibration, oldest first",
      [e["action"] for e in log] == ["calibrated", "calibrated"],
      str([e["action"] for e in log]))
check("the second line records that it replaced one", log[1]["replaced"] is True)
check("the current record is the newest one",
      T.load_calibration(root, "Ann")["reference"] == [300.0] * 8)
check("the REPLACED record is still in the log -- a calibration is never "
      "edited away",
      log[0]["record"]["reference"] == good_record("Ann")["reference"])

dropped = T.clear_calibration(root, "Ann", "unit test")
check("clearing returns the record that was active", dropped is not None)
check("clearing removes the current record",
      T.load_calibration(root, "Ann") is None)
log = T.load_calibration_log(root, "Ann")
check("clearing is logged with the full record, so nothing is lost",
      log[-1]["action"] == "cleared"
      and log[-1]["record"]["reference"] == [300.0] * 8)
check("the reason is kept", log[-1]["reason"] == "unit test")
check("clearing twice is a no-op, not an error",
      T.clear_calibration(root, "Ann") is None)

try:
    T.save_calibration(flat, root, "Flat")
    refused = False
except ValueError:
    refused = True
check("save_calibration REFUSES a record that would divide by noise", refused)
check("nothing was written for the refused wearer",
      not os.path.exists(T.calibration_path(root, "Flat")))

T.save_calibration(good_record("Ann"), root, "Ann")
T.save_calibration(good_record("Bob", ref=[250] * 8), root, "Bob")
check("calibrations are per wearer, and both are found",
      sorted(T.load_calibrations(root)) == ["Ann", "Bob"],
      str(sorted(T.load_calibrations(root))))
check("one wearer's numbers never leak into another's",
      T.load_calibration(root, "Bob")["reference"] == [250.0] * 8)
check("the table is empty when the mode is off",
      T.calibration_table(root, "off") == {})
check("the table resolves every usable wearer when a mode is chosen",
      sorted(T.calibration_table(root, "gain")) == ["Ann", "Bob"])
check("persons= narrows the table",
      sorted(T.calibration_table(root, "gain", persons=["Bob"])) == ["Bob"])

#  An archived dataset must not be able to contribute a calibration.
os.makedirs(os.path.join(root, "_archive", "Ann"), exist_ok=True)
shutil.copy(T.calibration_path(root, "Ann"),
            os.path.join(root, "_archive", "Ann", T.CALIBRATION_NAME))
check("underscore folders are pruned, exactly as discovery prunes them",
      "_archive" not in T.load_calibrations(root))

#  A corrupt file degrades to uncalibrated, never to wrongly calibrated.
with open(T.calibration_path(root, "Bob"), "w", encoding="utf-8") as fh:
    fh.write("{not json")
check("a corrupt record loads as None, not as a crash",
      T.load_calibration(root, "Bob") is None)
check("a corrupt record keeps that wearer out of the table",
      sorted(T.calibration_table(root, "gain")) == ["Ann"])

#  A relative dataset root must resolve against the PROJECT, not against
#  whatever directory the app happened to be launched from -- the same rule
#  LivePredictor.full_path follows for the model file.
check("a relative root resolves against the project directory, not the cwd",
      os.path.normpath(T.calibration_path("datasets", "Ann"))
      == os.path.normpath(os.path.join(T.PROJECT_DIR, "datasets", "Ann",
                                       T.CALIBRATION_NAME)),
      T.calibration_path("datasets", "Ann"))
check("an absolute root is used as given",
      os.path.normpath(T.calibration_path(root, "Ann")).startswith(
          os.path.normpath(root)))

#  Fingerprint neutrality: adding a calibration must not age every model.
fp_root = tmpdir()
writer = T.DatasetWriter(fp_root, "Cal", "session_01")
for i in range(3):
    emg = np.random.default_rng(i).integers(20, 400, (80, 8)).astype(np.int32)
    t = np.arange(80) / FS
    writer.save(T.Recording(emg=emg, timestamps=t, label="FIST",
                            person_id="Cal", session_id="session_01",
                            repetition=i + 1, stage=T.STAGE_RAW))
before = T.dataset_fingerprint(fp_root)
before_n = len(T.discover_recordings(fp_root))
T.save_calibration(good_record("Cal"), fp_root, "Cal")
check("adding a calibration does not move the dataset fingerprint -- "
      "otherwise every saved model would read 'stale' at once",
      T.dataset_fingerprint(fp_root) == before,
      f"{T.dataset_fingerprint(fp_root)['uid_digest']} vs "
      f"{before['uid_digest']}")
check("a calibration is not discovered as a recording",
      len(T.discover_recordings(fp_root)) == before_n, str(before_n))
check("the calibration file is valid JSON on disk",
      isinstance(json.load(open(T.calibration_path(fp_root, "Cal"),
                                encoding="utf-8")), dict))


# ==============================================================================
section("4. derivation from recordings -- labels are NOT used")
# ==============================================================================
derived_root = tmpdir()
writer = T.DatasetWriter(derived_root, "Derived", "session_01")
rng = np.random.default_rng(7)
for label, level in (("REST", 20), ("FIST", 400), ("OPEN", 300)):
    for i in range(3):
        emg = (rng.normal(level, 5.0, (120, 8))
               .clip(1, 2047).astype(np.int32))
        writer.save(T.Recording(emg=emg, timestamps=np.arange(120) / FS,
                                label=label, person_id="Derived",
                                session_id="session_01", repetition=i + 1,
                                stage=T.STAGE_RAW))
recs = T.load_dataset(derived_root, stage=T.STAGE_RAW, prefer="npz")
check("the fixture loaded", len(recs) == 9, str(len(recs)))

derived = T.calibration_from_recordings(recs, "Derived")
ok, problems = T.validate_calibration(derived)
check("a calibration can be derived from existing recordings", ok,
      str(problems))
check("it is marked as derived, not as a live measurement",
      derived["source"] == "derived_from_recordings", derived["source"])
check("it records that labels were not used",
      derived["derived_from"]["uses_labels"] is False)
check("it records how much signal it saw",
      derived["derived_from"]["n_recordings"] == 9)
#  The fixture is one third REST at level 20 and two thirds active at
#  300-450, so the pooled 10th percentile has to land inside the REST
#  cluster. That is the whole trick: the quiet percentile finds rest without
#  ever being told which recordings are labelled REST.
check("the derived rest level lands inside the quiet cluster, found without "
      "reading a single label",
      10.0 < derived["rest_level"] < 60.0, f"{derived['rest_level']:.1f}")
check("the derived reference lands near the loud end",
      derived["reference_level"] > derived["rest_level"] * 2,
      f"{derived['reference_level']:.1f}")

#  Label independence, stated as a test rather than as a comment: shuffling
#  every label must not move a single number.
shuffled = []
for i, r in enumerate(recs):
    clone = T.Recording(emg=r.emg, timestamps=r.timestamps,
                        label=["REST", "FIST", "OPEN"][i % 3],
                        person_id=r.person_id, session_id=r.session_id,
                        repetition=r.repetition, stage=r.stage)
    shuffled.append(clone)
reshuffled = T.calibration_from_recordings(shuffled, "Derived")
check("relabelling every recording changes NOTHING -- the derivation cannot "
      "be leaking the held-out wearer's labels",
      reshuffled["rest"] == derived["rest"]
      and reshuffled["reference"] == derived["reference"])

check("another wearer's id derives nothing",
      T.calibration_from_recordings(recs, "Nobody") is None)
check("an empty list derives nothing",
      T.calibration_from_recordings([], "Derived") is None)


# ==============================================================================
section("5. the feature path -- one function, and a refusal when it mismatches")
# ==============================================================================
block = np.tile(np.asarray([40, 80, 120, 160, 200, 240, 280, 320],
                           dtype=float), (18, 1))
cal = T.resolve_calibration(rec, "per_channel")

plain = T.model_feature_vector(block, "raw", "level", True, 60.0)
calibrated = T.model_feature_vector(block, "raw", "level", True, 60.0, cal)
check("the feature vector keeps its width when calibrated",
      plain.shape == calibrated.shape, f"{plain.shape} vs {calibrated.shape}")
check("calibration actually changes the features", not np.allclose(plain,
                                                                   calibrated))
check("passing calibration=None reproduces the uncalibrated vector exactly",
      np.allclose(plain, T.model_feature_vector(block, "raw", "level", True,
                                                60.0, None)))
by_hand = T.model_feature_vector(T.apply_calibration(block, cal),
                                 "raw", "level", True, 60.0)
check("calibrating then featurising is the same as asking the one function "
      "to do both -- there is no second code path",
      np.allclose(by_hand, calibrated))

#  `share` is amplitude-invariant, so the GAIN mode must not touch it.
names = T.model_feature_names("level", True)
share_cols = [i for i, n in enumerate(names) if n.endswith("_share")]
gain_cal = T.resolve_calibration(rec, "gain")
gain_feats = T.model_feature_vector(block, "raw", "level", True, 60.0,
                                    gain_cal)
check("gain mode leaves every `share` feature untouched, because share "
      "already has amplitude divided out",
      np.allclose(plain[share_cols], gain_feats[share_cols]),
      str(np.round(plain[share_cols] - gain_feats[share_cols], 9)))
check("gain mode DOES move the absolute level features",
      not np.allclose(plain[[names.index("CH1_mean")]],
                      gain_feats[[names.index("CH1_mean")]]))

#  LivePredictor must refuse a model it cannot reproduce the scale for.
pred_root = tmpdir()
T.DATA_FOLDER = pred_root
T.PERSON_ID = "Wearer"


class _FakeModel:
    classes_ = np.array([0, 1])

    def predict(self, X):
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X):
        return np.tile([0.6, 0.4], (len(X), 1))


def _write_payload(path, mode, subjects=()):
    import joblib
    joblib.dump({
        "model": _FakeModel(), "labels": ["REST", "FIST"], "fs": 60.0,
        "window_s": 0.3, "window_samples": 18, "representation": "raw",
        "feature_set": "level", "freq_features": True,
        "calibration": {"mode": mode, "subjects": list(subjects),
                        "target_level": T.CALIBRATION_TARGET_LEVEL},
        "stats": {},
    }, path)


try:
    import joblib                                              # noqa: F401
    have_joblib = True
except ImportError:
    have_joblib = False

if not have_joblib:
    print("  (joblib missing -- the LivePredictor checks are skipped)")
else:
    model_path = os.path.join(pred_root, "m.joblib")

    #  Backward compatibility: a payload written before calibration existed
    #  has no `calibration` key at all. It must load exactly as it did.
    joblib.dump({
        "model": _FakeModel(), "labels": ["REST", "FIST"], "fs": 60.0,
        "window_s": 0.3, "window_samples": 18, "representation": "raw",
        "feature_set": "level", "freq_features": True, "stats": {},
    }, model_path)
    p = T.LivePredictor(model_path)
    check("a payload with NO calibration block at all still loads",
          p.load() and p.ready, p.status)
    check("and is treated as uncalibrated",
          p.calibration_mode == "off" and p.calibration is None)

    _write_payload(model_path, "off")
    p = T.LivePredictor(model_path)
    check("a model trained with calibration off loads for an uncalibrated "
          "wearer", p.load() and p.ready, p.status)
    check("and it applies no calibration", p.calibration is None)
    check("and it records the mode it was trained with",
          p.calibration_mode == "off")

    _write_payload(model_path, "affine", ["Wearer"])
    p = T.LivePredictor(model_path)
    loaded = p.load()
    check("a CALIBRATED model REFUSES to load for a wearer who has none -- "
          "it would be confidently wrong, not broken",
          not loaded and not p.ready)
    check("the refusal says what to do",
          "CALIBRATE" in p.status and "retrain" in p.status, p.status)

    T.save_calibration(good_record("Wearer"), pred_root, "Wearer")
    p = T.LivePredictor(model_path)
    check("once the wearer is calibrated the same model loads",
          p.load() and p.ready, p.status)
    check("and the resolved calibration is in force",
          p.calibration is not None and p.calibration["mode"] == "affine")
    check("the note names the wearer and the mode",
          "affine" in p.calibration_note and "Wearer" in p.calibration_note,
          p.calibration_note)

    emg = np.tile(np.asarray([40, 80, 120, 160, 200, 240, 280, 320],
                             dtype=np.int32), (24, 1))
    t = np.arange(24) / 50.0
    out = p.predict(emg, t)
    check("it predicts, and from the CALIBRATED features",
          out is not None and p.last_features is not None)
    p_off = T.LivePredictor(model_path)
    p_off.load()
    p_off.calibration = None
    p_off.predict(emg, t)
    check("the uncalibrated features of the same window are different -- "
          "which is why the refusal above matters",
          not np.allclose(p.last_features, p_off.last_features))

    #  THE TRAP: a mode is configured but nobody has a usable calibration.
    #  The trainer must then record what it DID (nothing) and not what was
    #  asked for -- otherwise it stamps the payload "affine", trains on plain
    #  device counts, and the app refuses to load its own fresh model.
    empty_root = tmpdir()
    check("with no calibration on disk the table is empty, whatever the mode",
          T.calibration_table(empty_root, "affine") == {})
    joblib.dump({
        "model": _FakeModel(), "labels": ["REST", "FIST"], "fs": 60.0,
        "window_s": 0.3, "window_samples": 18, "representation": "raw",
        "feature_set": "level", "freq_features": True, "stats": {},
        "calibration": {"mode": "off", "requested": "affine",
                        "subjects": []},
    }, model_path)
    p = T.LivePredictor(model_path)
    check("a payload that RECORDS what was applied ('off') rather than what "
          "was requested ('affine') loads for an uncalibrated wearer",
          p.load() and p.ready, p.status)
    check("and applies nothing", p.calibration is None)
    joblib.dump({
        "model": _FakeModel(), "labels": ["REST", "FIST"], "fs": 60.0,
        "window_s": 0.3, "window_samples": 18, "representation": "raw",
        "feature_set": "level", "freq_features": True, "stats": {},
        "calibration": {"mode": "affine", "subjects": []},
    }, model_path)
    #  Judged as an UNCALIBRATED wearer would see it -- `Wearer` has a
    #  calibration by this point in the file, and for them the affine stamp
    #  is honest and loads. The lock-out is what happens to everybody else.
    _was = T.PERSON_ID
    T.PERSON_ID = "SomebodyElse"
    check("whereas stamping the REQUESTED mode on uncalibrated features "
          "locks the model out of its own app -- the bug this guards",
          not T.LivePredictor(model_path).load())
    T.PERSON_ID = _was
    check("the same payload still loads for a wearer who IS calibrated, so "
          "the lock-out is about the wearer and not about the stamp",
          T.LivePredictor(model_path).load())
    _write_payload(model_path, "affine", ["Wearer"])

    #  A calibration that stops validating is as bad as a missing one.
    broken = dict(good_record("Wearer"))
    broken["reference"] = [21.0] * 8
    with open(T.calibration_path(pred_root, "Wearer"), "w",
              encoding="utf-8") as fh:
        json.dump(broken, fh)
    p = T.LivePredictor(model_path)
    check("a calibration that no longer validates is refused too",
          not p.load())
    check("and the reason is the validation reason, not a generic error",
          "cannot be applied" in p.status, p.status)


# ==============================================================================
section("6. the GUI -- the control, the status line, and derive")
# ==============================================================================
from PySide6 import QtWidgets                                   # noqa: E402

QtWidgets.QMessageBox.information = staticmethod(
    lambda *a, **k: QtWidgets.QMessageBox.Ok)
QtWidgets.QMessageBox.warning = staticmethod(
    lambda *a, **k: QtWidgets.QMessageBox.Ok)
QtWidgets.QMessageBox.question = staticmethod(
    lambda *a, **k: QtWidgets.QMessageBox.Yes)

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

gui_root = tmpdir()
T.DATA_FOLDER = gui_root
T.MYO_PORT = "COM254"
T.PERSON_ID = "Gui"
T.CALIBRATION_MODE = "off"

#  Recordings to derive from, so the no-armband path has something to read.
writer = T.DatasetWriter(gui_root, "Gui", "session_01")
rng = np.random.default_rng(11)
for label, level in (("REST", 20), ("FIST", 450)):
    for i in range(3):
        emg = rng.normal(level, 6.0, (120, 8)).clip(1, 2047).astype(np.int32)
        writer.save(T.Recording(emg=emg, timestamps=np.arange(120) / FS,
                                label=label, person_id="Gui",
                                session_id="session_01", repetition=i + 1,
                                stage=T.STAGE_RAW))

dialog = T.CalibrationDialog(None, None, "Gui", gui_root, "off")
check("the dialog opens with no armband and no calibration",
      dialog.pending_record() is None)
check("SAVE is disabled with nothing measured",
      not dialog.save_button.isEnabled())
check("CLEAR is disabled when there is nothing to clear",
      not dialog.clear_button.isEnabled())
dialog._start("rest")
check("a live capture refuses to start with no acquisition at all",
      dialog.phase == dialog.IDLE
      and "Cannot capture" in dialog.warning_label.text(),
      dialog.warning_label.text())

#  THE BUG THIS REPLACED: the readiness check was
#  `getattr(acq, "running", False)`, and MyoAcquisition has no `running`
#  attribute -- so it was ALWAYS False and a live capture could never start,
#  armband or not. Pin the real contract so it cannot come back.
real_acq = T.MyoAcquisition(port="COM254")
check("MyoAcquisition still has no `running` attribute -- the readiness "
      "check must not go looking for one again",
      not hasattr(real_acq, "running"))
check("it does expose last_sample_age in stats(), which is what readiness "
      "is judged on",
      "last_sample_age" in real_acq.stats())
probe = T.CalibrationDialog.__new__(T.CalibrationDialog)
probe.acq = real_acq
ok_real, why_real = T.CalibrationDialog.stream_state(probe)
check("a stopped acquisition reads as not streaming, and names its state",
      not ok_real and "DISCONNECTED" in why_real, why_real)
probe.acq = None
check("no acquisition reads as not streaming too",
      not T.CalibrationDialog.stream_state(probe)[0])

dialog._derive()
check("DERIVE FROM RECORDED DATA produces a record with no armband",
      dialog.pending_record() is not None)
check("SAVE becomes available once the record validates",
      dialog.save_button.isEnabled())
check("the per-channel table is filled in",
      dialog.table.item(0, 2) is not None
      and dialog.table.item(0, 2).text() != "--",
      dialog.table.item(0, 2).text() if dialog.table.item(0, 2) else "none")
dialog.mode_combo.setCurrentIndex(dialog.mode_combo.findData("gain"))
check("choosing a mode in the dialog is remembered", dialog.mode == "gain")
dialog._save()
check("saving reports success and writes the wearer's record",
      dialog.saved and T.load_calibration(gui_root, "Gui") is not None)
dialog.deleteLater()

# ---- the guided gesture sweep, on a fake stream ---------------------------
#  A stand-in acquisition whose ring returns whatever level the test sets, so
#  the sweep can be walked without hardware and without waiting out any
#  countdown.
class _FakeRing:
    level = np.full(8, 21.0)

    def snapshot(self, n=None):
        n = int(min(n or 4000, 4000))
        emg = np.random.default_rng(0).normal(self.level, 4.0, (n, 8))
        return emg.clip(1, None).astype(np.int32), np.arange(n) / 50.0


class _FakeAcq:
    state = "connected"

    def __init__(self):
        self.ring = _FakeRing()
        self.age = 0.05

    def stats(self):
        return {"rate": 50.0, "last_sample_age": self.age}


PROFILES = {"FIST":       [550, 300, 60, 70, 400, 300, 200, 180],
            "OPEN":       [200, 650, 80, 90, 300, 250, 300, 220],
            "THUMB_UP":   [30, 33, 31, 30, 35, 34, 32, 31],   # a shrug
            "WAVE_LEFT":  [100, 120, 70, 80, 900, 990, 400, 300],
            "WAVE_RIGHT": [120, 140, 160, 240, 300, 260, 390, 430]}

acq = _FakeAcq()
T.GESTURES = ["REST", "FIST", "OPEN", "THUMB_UP", "WAVE_LEFT", "WAVE_RIGHT"]
sweep = T.CalibrationDialog(None, acq, "Sweeper", gui_root, "affine")
check("with samples arriving the dialog reports a live stream",
      sweep.stream_state()[0], str(sweep.stream_state()))
acq.age = 9.0
check("a stream with no recent sample is NOT ready, however connected it is",
      not sweep.stream_state()[0] and "stopped" in sweep.stream_state()[1],
      sweep.stream_state()[1])
acq.age = None
check("and no samples at all is its own message",
      "no samples yet" in sweep.stream_state()[1], sweep.stream_state()[1])
acq.age = 0.05
check("REST is not one of the cued gestures -- it sets the floor separately",
      "REST" not in sweep.sweep_gestures(), str(sweep.sweep_gestures()))

sweep.start_sweep()
check("the sweep starts on rest", sweep._target == "rest")
guard = 0
while (sweep.phase != sweep.IDLE or sweep._queue) and guard < 40:
    guard += 1
    acq.ring.level = (np.full(8, 21.0) if sweep._target == "rest"
                      else np.asarray(PROFILES.get(sweep._target, [21] * 8),
                                      dtype=float))
    sweep._deadline = 0.0
    sweep._on_tick()
    if sweep.phase == sweep.COUNTING:
        sweep._deadline = 0.0
        sweep._on_tick()
check("the sweep ran every gesture without being waited on",
      sorted(sweep.gesture_blocks) == sorted(PROFILES),
      str(sorted(sweep.gesture_blocks)))

swept = sweep.pending_record()
check("the record says it came from a gesture sweep",
      swept["source"] == "live_gesture_sweep", swept["source"])
check("the shrug is REJECTED, with an actionable reason",
      swept["gesture_sweep"]["rejected"] == ["THUMB_UP"],
      str(swept["gesture_sweep"]["rejected"]))
check("the four real gestures are accepted",
      swept["gesture_sweep"]["gestures"]
      == ["FIST", "OPEN", "WAVE_LEFT", "WAVE_RIGHT"],
      str(swept["gesture_sweep"]["gestures"]))
check("a rejected capture contributes nothing to the reference rather than "
      "dragging its channels down",
      swept["reference"] == T.calibration_from_gesture_captures(
          sweep.rest_block,
          {k: v for k, v in sweep.gesture_blocks.items() if k != "THUMB_UP"},
          person_id="Sweeper")["reference"])
check("the swept record validates", T.validate_calibration(swept)[0],
      str(T.validate_calibration(swept)[1]))
check("every channel gets a reference above its rest, which one clench "
      "cannot promise",
      all(r > s for r, s in zip(swept["reference"], swept["rest"])),
      str(np.round(swept["reference"], 0)))

#  THE REASON THE SWEEP EXISTS, as a measurement rather than a claim.
single = T.calibration_from_captures(sweep.rest_block,
                                     sweep.gesture_blocks["FIST"],
                                     person_id="Sweeper")
ratio = (np.asarray(swept["reference"])
         / np.maximum(np.asarray(single["reference"]), 1e-9))
check("a ONE-GESTURE reference under-scales most channels badly -- the "
      "quietest by more than 2x",
      float(np.max(ratio)) > 2.0 and int((ratio > 1.5).sum()) >= 5,
      f"sweep/single = {np.round(ratio, 2)}")

check("the table shows a row per gesture with its verdict",
      sweep.sweep_table.rowCount() == len(PROFILES))
verdicts = {sweep.sweep_table.item(r, 0).text():
            sweep.sweep_table.item(r, 3).text()
            for r in range(sweep.sweep_table.rowCount())}
check("and the shrug's row says what to do about it",
      "hold it harder" in verdicts.get("THUMB_UP", ""),
      verdicts.get("THUMB_UP"))
check("while a good capture reads as accepted",
      verdicts.get("FIST") == "accepted", verdicts.get("FIST"))

#  Losing the stream mid-capture must abandon it, not store a stub.
sweep.gesture_blocks.pop("FIST")
sweep._start("FIST")
acq.age = 9.0
sweep._on_tick()
check("a capture whose stream dies is abandoned and says so",
      sweep.phase == sweep.IDLE and "FIST" not in sweep.gesture_blocks
      and "stopped during the capture" in sweep.warning_label.text(),
      sweep.warning_label.text())
acq.age = 0.05
sweep.deleteLater()

#  A second dialog on a wearer who has none, to check the empty path again.
empty = T.CalibrationDialog(None, None, "Nobody", gui_root, "gain")
empty._derive()
check("deriving for a wearer with no recordings says so rather than "
      "inventing numbers",
      empty.pending_record() is None
      and "no recordings" in empty.warning_label.text(),
      empty.warning_label.text())
empty.deleteLater()

#  The main window's panel.
T.CALIBRATION_MODE = "gain"
window = T.MainWindow()
check("the LIVE CLASSIFIER panel has a calibration control",
      hasattr(window, "calibration_button"))
check("the button offers RECALIBRATE for a wearer who already has one",
      window.calibration_button.text() == "RECALIBRATE...",
      window.calibration_button.text())
check("the status line says the calibration is being applied",
      "CALIBRATED" in window.calibration_label.text()
      and "not applied" not in window.calibration_label.text(),
      window.calibration_label.text())

T.CALIBRATION_MODE = "off"
window._refresh_calibration_panel()
check("with the mode off the status line says stored but not applied",
      "not applied" in window.calibration_label.text(),
      window.calibration_label.text())

window.person_edit.setText("Stranger")
window._on_identity_changed()
check("switching wearer switches which calibration is in force",
      window.calibration_label.text().startswith("NOT CALIBRATED"),
      window.calibration_label.text())
check("and the button offers CALIBRATE, not RECALIBRATE",
      window.calibration_button.text() == "CALIBRATE...",
      window.calibration_button.text())

window.person_edit.setText("Gui")
window._on_identity_changed()
check("switching back finds it again",
      window.calibration_label.text().startswith("CALIBRATED"),
      window.calibration_label.text())

window.free_capture = object()
window.open_calibration()
check("calibrating is refused while a take is recording -- it reads the live "
      "signal",
      window.calibration_button.text() == "RECALIBRATE...")
window.free_capture = None

check("the retrain command passes the mode to the trainer, so the "
       "subprocess cannot calibrate differently from the GUI",
      "--calibration" in open(os.path.join(PROJECT, "emg_dataset_tool.py"),
                              encoding="utf-8").read())
window.close()
window.deleteLater()


# ==============================================================================
for path in TMP:
    shutil.rmtree(path, ignore_errors=True)

print(f"\n{'=' * 60}")
print(f"{PASS}/{PASS + FAIL} checks passed")
print(f"{'=' * 60}")
sys.exit(1 if FAIL else 0)
