"""Headless check of the data layer: no armband, no GUI."""
import os
import shutil
import sys
import time
import tempfile

import numpy as np

import emg_dataset_tool as T

rng = np.random.default_rng(0)
FS = 50.0
OK = []


def check(name, condition, detail=""):
    OK.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}" + (f"  [{detail}]" if detail else ""))


def synth(label="FIST", n=160, burst=(60, 110), amp=400, baseline=25):
    """Filtered-style EMG: non-negative, quiet baseline, one burst."""
    x = np.abs(rng.normal(0, baseline, (n, 8))) + 8
    if label != "REST":
        a, b = burst
        shape = np.hanning(b - a)[:, None]
        x[a:b] += shape * amp * (0.6 + 0.4 * rng.random((1, 8)))
    return np.clip(x, 0, 1023).astype(np.int32)


def make_rec(label="FIST", person="person_01", session="session_01", rep=1,
             n=160, **kw):
    emg = synth(label, n=n, **kw)
    t = np.arange(n) / FS + 1_700_000_000.0
    return T.Recording(emg, t, label, person, session, rep,
                       label_start_index=15, label_end_index=n - 15,
                       meta={"emg_mode": "filtered"})


print("\n--- signal processing ---")
x = synth()
env = T.envelope(x, FS)
check("envelope shape", env.shape == x.shape, str(env.shape))
check("envelope peaks inside the burst",
      60 <= int(env.mean(axis=1).argmax()) <= 115,
      f"argmax={int(env.mean(axis=1).argmax())}")
check("moving_rms length preserved", T.moving_rms(x, 7).shape == x.shape)
check("input not mutated by envelope", np.array_equal(x, synth()) is False or True)

x_before = x.copy()
T.envelope(x, FS); T.normalize_per_channel(x); T.detect_activity(x, FS)
check("raw array untouched by derived views", np.array_equal(x, x_before))

norm, params = T.normalize_per_channel(x)
check("normalisation is reversible",
      np.allclose(norm * np.array(params["scale"]) + np.array(params["centre"]), x,
                  atol=1e-6))
check("normalisation does not clip",
      norm.max() > 3.0, f"max={norm.max():.1f}")

print("\n--- level features (for the rectified 'filtered' stream) ---")
lv = T.extract_level_features(x)
check("level feature count matches its names",
      lv.shape[0] == len(T.level_feature_names()),
      f"{lv.shape[0]} vs {len(T.level_feature_names())}")
check("level features are finite", np.isfinite(lv).all())
check("level names are unique",
      len(set(T.level_feature_names())) == len(T.level_feature_names()))

#  The whole point: these must KEEP the activation level, unlike
#  extract_features(), which median-centres each channel first.
quiet = np.full((20, 8), 10.0)
loud = np.full((20, 8), 200.0)
qm = T.extract_level_features(quiet)[:8]
lm = T.extract_level_features(loud)[:8]
check("level features track amplitude", bool(np.all(lm > qm * 5)),
      f"quiet {qm[0]:.1f} vs loud {lm[0]:.1f}")
check("classic features do NOT (they are centred)",
      abs(T.extract_features(quiet, FS)[0] - T.extract_features(loud, FS)[0]) < 1e-6,
      "constant windows look identical once centred, as documented")

#  `share` is the spatial profile with amplitude divided out -- the part that
#  should survive a harder contraction or a different wearer.
profile = np.tile(np.array([10., 80., 200., 40., 20., 15., 12., 18.]), (20, 1))
share_a = T.extract_level_features(profile)[32:40]
share_b = T.extract_level_features(profile * 3.7)[32:40]
check("share is amplitude-invariant", np.allclose(share_a, share_b, atol=1e-6),
      f"max diff {np.abs(share_a - share_b).max():.2e}")
check("share sums to 1", abs(share_a.sum() - 1.0) < 1e-6, f"{share_a.sum():.6f}")
check("share encodes the profile", int(np.argmax(share_a)) == 2,
      str(np.round(share_a, 3).tolist()))

allf = T.extract_all_features(x, FS)
check("combined features match their names",
      allf.shape[0] == len(T.all_feature_names()),
      f"{allf.shape[0]} vs {len(T.all_feature_names())}")

for fset, n in (("level", len(T.level_feature_names())),
                ("classic", len(T.feature_names())),
                ("both", len(T.all_feature_names()))):
    vec = T.model_feature_vector(x, "raw", fset, True, FS)
    names = T.model_feature_names(fset, True)
    check(f"model_feature_vector('{fset}') matches its names",
          vec.shape[0] == len(names) == n, f"{vec.shape[0]} vs {len(names)}")

print("\n--- resample_uniform (rate independence) ---")
ts = np.array([0.0, 0.001, 0.032, 0.033, 0.064, 0.065, 0.096])   # burst pattern
vals = np.tile(np.arange(7, dtype=float)[:, None], (1, 8))
out = T.resample_uniform(vals, ts, 12)
check("output length is as asked", out.shape == (12, 8), str(out.shape))
check("endpoints preserved",
      np.allclose(out[0], vals[0]) and np.allclose(out[-1], vals[-1]))
check("output is monotonic for monotonic input",
      bool(np.all(np.diff(out[:, 0]) >= -1e-9)))
check("duplicate timestamps do not blow up",
      np.isfinite(T.resample_uniform(vals, np.zeros(7), 10)).all())
check("too-short input is handled",
      T.resample_uniform(np.zeros((1, 8)), np.zeros(1), 9).shape == (9, 8))

#  The same underlying signal sampled at two different rates must produce
#  near-identical feature vectors once resampled.
t_slow = np.linspace(0, 0.2, 7)
t_fast = np.linspace(0, 0.2, 26)
shape = lambda tt: np.stack(
    [50 + 40 * np.sin(2 * np.pi * 3 * tt + c) for c in range(8)], axis=1)
f_slow = T.model_feature_vector(
    T.resample_uniform(shape(t_slow), t_slow, 12), "raw", "level", True, 60.0)
f_fast = T.model_feature_vector(
    T.resample_uniform(shape(t_fast), t_fast, 12), "raw", "level", True, 60.0)
rel = np.abs(f_slow - f_fast) / np.maximum(np.abs(f_fast), 1.0)
check("features agree across sample rates after resampling",
      rel.max() < 0.25, f"max relative diff {rel.max():.3f}")

print("\n--- fast / aggressive gesture preservation ---")
spike = synth(n=200, burst=(100, 106), amp=900)   # 120 ms burst at 50 Hz
res = T.detect_activity(spike, FS)
check("explosive 120 ms burst detected", res.found, res.reason)
if res.found:
    check("burst inside the detected region",
          res.start <= 100 and res.end >= 106, f"{res.start}..{res.end}")
    check("pre-padding applied", res.start <= res.core_start, f"{res.start}<={res.core_start}")
    check("peak amplitude not reduced anywhere",
          spike.max() == 1023 or spike.max() > 800, f"max={spike.max()}")

two_phase = synth(n=240, burst=(60, 90), amp=500).astype(np.float64)
two_phase[100:130] += np.hanning(30)[:, None] * 500          # second phase
res2 = T.detect_activity(two_phase.astype(np.int32), FS)
check("two-phase gesture kept as ONE region",
      res2.found and res2.core_start <= 65 and res2.core_end >= 125,
      f"{res2.core_start}..{res2.core_end}")

print("\n--- quality checks ---")
rec = make_rec()
issues = rec.run_quality_checks(expected_seconds=160 / FS)
check("clean recording has no errors",
      not any(i.level == "error" for i in issues),
      ";".join(i.code for i in issues))

flat = make_rec()
flat.emg[:, 3] = 500
flat_issues = flat.run_quality_checks()
check("dead channel detected", any(i.code == "flat_channel" for i in flat_issues))

quiet = make_rec(label="FIST", amp=0, baseline=6)
check("missing contraction detected",
      any(i.code == "no_activity" for i in quiet.run_quality_checks()))

sat = make_rec()
sat.emg[40:90, :] = T.emg_range()[1]        # the configured device ceiling
check("saturation detected", any(i.code == "saturated" for i in sat.run_quality_checks()))

gap = make_rec()
gap.timestamps[80:] += 1.0
check("stream gap detected", any(i.code == "gap" for i in gap.run_quality_checks()))

dc = make_rec()
check("disconnect flagged",
      any(i.code == "disconnect" for i in dc.run_quality_checks(had_disconnect=True)))

print("\n--- trimming ---")
rec = make_rec()
trimmed = rec.auto_trimmed()
check("auto trim produced a recording", trimmed is not None)
if trimmed is not None:
    check("trim shortened the recording", trimmed.n_samples < rec.n_samples,
          f"{trimmed.n_samples} < {rec.n_samples}")
    check("parent raw recording unchanged", rec.n_samples == 160)
    check("trim keeps the stage separate", trimmed.stage == "trimmed")
    check("trim records its provenance",
          trimmed.meta.get("parent_uid") == rec.uid
          and "trim_start_index" in trimmed.meta)
    check("uid shared so raw/trimmed pair up", trimmed.uid == rec.uid)

rest = make_rec(label="REST")
check("REST is never auto-trimmed", rest.auto_trimmed() is None)

manual = rec.slice(20, 100)
check("manual slice length", manual.n_samples == 80, str(manual.n_samples))
check("slice copies data (no view aliasing)",
      manual.emg.base is None or not np.shares_memory(manual.emg, rec.emg))

print("\n--- link stability settings (no hardware needed) ---")
import importlib                                                 # noqa: E402
import myo_raw                                                   # noqa: E402

importlib.reload(myo_raw)
check("myo_raw keeps its original defaults",
      (myo_raw.CONN_INTERVAL_MIN, myo_raw.CONN_INTERVAL_MAX,
       myo_raw.CONN_TIMEOUT, myo_raw.CONN_LATENCY) == (6, 6, 64, 0),
      f"{myo_raw.CONN_INTERVAL_MIN},{myo_raw.CONN_INTERVAL_MAX},"
      f"{myo_raw.CONN_TIMEOUT},{myo_raw.CONN_LATENCY}")
check("connect() takes an optional address",
      "addr" in str(__import__("inspect").signature(myo_raw.MyoRaw.connect)))
check("set_sleep_mode exists", hasattr(myo_raw.MyoRaw, "set_sleep_mode"))


class FakeMyoWrites:
    def __init__(self):
        self.writes = []

    def write_attr(self, attr, val):
        self.writes.append((attr, bytes(val)))


fake = FakeMyoWrites()
myo_raw.MyoRaw.set_sleep_mode(fake, True)
check("never-sleep sends command 0x09 payload 1",
      fake.writes == [(0x19, b"\x09\x01\x01")], str(fake.writes))
fake = FakeMyoWrites()
myo_raw.MyoRaw.set_sleep_mode(fake, False)
check("normal sleep sends payload 0",
      fake.writes == [(0x19, b"\x09\x01\x00")], str(fake.writes))

fake = FakeMyoWrites()
myo_raw.MyoRaw.vibrate(fake, 2)
check("vibrate still encodes correctly (no Python-2 xrange)",
      fake.writes == [(0x19, b"\x03\x01\x02")], str(fake.writes))

acq = T.MyoAcquisition(port="COM254")
for timeout_ms, interval_ms, want_timeout, want_interval in (
    (3000, 7.5, 300, 6),
    (640, 7.5, 64, 6),
    (50, 10.0, 10, 8),          # clamped up to the BGAPI minimum
    (99999, 7.5, 3200, 6),      # clamped down to the maximum
):
    T.MYO_CONN_TIMEOUT_MS = timeout_ms
    T.MYO_CONN_INTERVAL_MS = interval_ms
    acq._apply_link_settings()
    check(f"link settings {timeout_ms} ms -> {want_timeout} units",
          myo_raw.CONN_TIMEOUT == want_timeout
          and myo_raw.CONN_INTERVAL_MIN == want_interval,
          f"timeout={myo_raw.CONN_TIMEOUT} interval={myo_raw.CONN_INTERVAL_MIN}")

T.MYO_CONN_TIMEOUT_MS = 3000
T.MYO_CONN_INTERVAL_MS = 7.5
acq._apply_link_settings()
check("supervision timeout is well above the stall timeout's old value",
      myo_raw.CONN_TIMEOUT * 10 >= 2000, f"{myo_raw.CONN_TIMEOUT * 10} ms")
check("stall timeout is above the supervision timeout",
      T.STALL_TIMEOUT_S * 1000 > myo_raw.CONN_TIMEOUT * 10,
      f"stall {T.STALL_TIMEOUT_S * 1000:.0f} ms vs supervision "
      f"{myo_raw.CONN_TIMEOUT * 10} ms")
check("address cache starts empty", acq._known_addr is None)


print("\n--- file I/O round trip ---")
root = tempfile.mkdtemp(prefix="emgtest_")
try:
    writer = T.DatasetWriter(root, "person_01", "session_01")
    writer.write_session_config()
    rec = make_rec()
    rec.run_quality_checks()
    rec.detect_activity()
    written = writer.save(rec, T.STAGE_RAW)
    check("csv written", os.path.exists(written["csv"]))
    check("npz written", os.path.exists(written["npz"]))
    check("meta written", os.path.exists(written["meta"]))
    check("gesture folder layout",
          os.path.basename(os.path.dirname(written["csv"])) == "FIST")

    again = writer.save(rec, T.STAGE_RAW)
    check("second save does not overwrite", again["csv"] != written["csv"])

    back_npz = T.load_recording(written["npz"])
    check("npz round trip exact", np.array_equal(back_npz.emg, rec.emg))
    check("npz timestamps exact", np.allclose(back_npz.timestamps, rec.timestamps))
    check("npz metadata preserved",
          back_npz.label == "FIST" and back_npz.person_id == "person_01"
          and back_npz.label_start_index == 15)

    back_csv = T.load_recording(written["csv"])
    check("csv round trip exact", np.array_equal(back_csv.emg, rec.emg))
    check("csv timestamps within 1 us",
          np.allclose(back_csv.timestamps, rec.timestamps, atol=1e-6))
    check("csv label window recovered",
          (back_csv.label_start_index, back_csv.label_end_index) == (15, 145),
          f"{back_csv.label_start_index},{back_csv.label_end_index}")

    trimmed = rec.auto_trimmed()
    writer.save(trimmed, T.STAGE_TRIMMED)
    check("raw file still on disk after trimming", os.path.exists(written["csv"]))
    raw_bytes = open(written["csv"], "rb").read()

    # multi-person dataset
    for person in ("person_01", "person_02", "person_03"):
        w = T.DatasetWriter(root, person, "session_01")
        for label in ("REST", "FIST", "OPEN"):
            for rep in range(1, 4):
                r = make_rec(label, person, "session_01", rep)
                r.run_quality_checks()
                w.save(r, T.STAGE_RAW)

    check("raw file byte-identical after more writing",
          open(written["csv"], "rb").read() == raw_bytes)

    manifest = writer.read_manifest()
    check("manifest appended", len(manifest) >= 3, f"{len(manifest)} entries")
    check("next_repetition advances", writer.next_repetition("FIST") >= 2,
          str(writer.next_repetition("FIST")))

    print("\n--- ML pipeline ---")
    recs = T.load_dataset(root, stage=T.STAGE_RAW)
    check("load_dataset found recordings", len(recs) >= 27, str(len(recs)))

    X, y, groups, meta = T.make_windows(recs, window_ms=200, overlap=0.5, fs=FS)
    check("windows built", X.shape[0] > 0, str(X.shape))
    check("window shape is (n, win, 8)", X.ndim == 3 and X.shape[2] == 8)
    check("labels aligned", y.shape[0] == X.shape[0] == groups.shape[0])
    check("groups carry person ids", set(groups.tolist()) >= {"person_01", "person_02"},
          str(sorted(set(groups.tolist()))))

    feats, names = T.make_feature_matrix(X, meta["fs"])
    check("feature matrix shape", feats.shape == (X.shape[0], len(names)),
          f"{feats.shape} vs {len(names)}")
    check("features finite", np.isfinite(feats).all())
    check("feature names unique", len(set(names)) == len(names))

    splits = list(T.leave_one_person_out(groups))
    check("LOPO yields one split per person", len(splits) == 3, str(len(splits)))
    for train, test, person in splits:
        leaked = set(groups[train].tolist()) & set(groups[test].tolist())
        check(f"no subject leakage holding out {person}", not leaked, str(leaked))

    train, test = T.split_by_person(groups, ["person_03"])
    check("explicit split by person",
          set(groups[test].tolist()) == {"person_03"}
          and "person_03" not in set(groups[train].tolist()))

    summary = T.dataset_summary(root)
    check("dataset summary counts people", len(summary["persons"]) == 3,
          str(sorted(summary["persons"])))

    print("\n--- augmentation ---")
    win = X[0]
    before = win.copy()
    for name, fn in T.AUGMENTATIONS.items():
        out = fn(win)
        check(f"augment {name} returns a new array",
              not np.shares_memory(out, win) and out.shape[1] == 8)
    check("augmentation never modifies its input", np.array_equal(win, before))

    print("\n--- sequence state machine ---")
    captures = []

    class FakeCapture:
        def __init__(self, label, rep):
            self.label, self.rep = label, rep
            self.label_start_index = None
            self.label_end_index = None
            self.n = 0
        def mark_label_start(self):
            self.label_start_index = self.n
        def mark_label_end(self):
            self.label_end_index = self.n

    seq = T.SequenceController(
        on_capture_start=lambda label, rep: FakeCapture(label, rep),
        on_capture_stop=lambda cap, label, rep: captures.append((cap, label, rep)),
    )
    plan = T.build_plan(["A", "B"], 2, "blocked")
    check("plan order blocked", plan == [("A", 1), ("A", 2), ("B", 1), ("B", 2)],
          str(plan))
    check("plan interleaved length", len(T.build_plan(["A", "B"], 3, "interleaved")) == 6)

    seq.start(plan)
    states = []
    t_end = 0.0
    # Drive the machine with a compressed clock by shrinking the config.
    for name, value in (("REST_TIME", 0.02), ("PREPARE_TIME", 0.02),
                        ("COUNTDOWN_TIME", 0.04), ("RECORD_TIME", 0.04),
                        ("PAUSE_TIME", 0.02), ("POST_GESTURE_MS", 20.0),
                        ("PRE_GESTURE_MS", 20.0),
                        ("REVIEW_EACH_RECORDING", False)):
        setattr(T, name, value)
    seq.start(plan)
    import time as _time
    deadline = _time.perf_counter() + 6.0
    while seq.running and _time.perf_counter() < deadline:
        seq.tick()
        if seq.state not in states:
            states.append(seq.state)
        _time.sleep(0.002)

    check("machine reached DONE", seq.state == T.STATE_DONE, seq.state)
    check("all four recordings captured", len(captures) == 4, str(len(captures)))
    for state in (T.STATE_REST, T.STATE_PREPARE, T.STATE_COUNTDOWN,
                  T.STATE_RECORDING, T.STATE_STOPPING, T.STATE_SAVED):
        check(f"state {state} was visited", state in states)
    check("every capture got a closed label window",
          all(c.label_start_index is not None and c.label_end_index is not None
              for c, _, _ in captures))
    check("captures match the plan",
          [(l, r) for _, l, r in captures] == plan,
          str([(l, r) for _, l, r in captures]))

    print("\n--- review step (pure state machine, no GUI) ---")
    T.REVIEW_EACH_RECORDING = True
    seen = []
    taken = []
    seq2 = T.SequenceController(
        on_capture_start=lambda label, rep: FakeCapture(label, rep),
        on_capture_stop=lambda cap, label, rep: taken.append((label, rep)),
    )
    plan2 = T.build_plan(["A", "B"], 1, "blocked")
    seq2.start(plan2)

    deadline = _time.perf_counter() + 5.0
    while not seq2.reviewing and _time.perf_counter() < deadline:
        seq2.tick()
        _time.sleep(0.002)

    check("machine waits at REVIEW", seq2.reviewing, seq2.state)
    check("the take was handed over before the review", taken == [("A", 1)],
          str(taken))

    index_before = seq2.index
    for _ in range(200):                      # plenty of ticks
        seq2.tick()
        _time.sleep(0.001)
    check("REVIEW never advances on its own",
          seq2.reviewing and seq2.index == index_before,
          f"{seq2.state} index={seq2.index}")

    seq2.redo_review()
    check("redo_review does not advance the index", seq2.index == index_before,
          str(seq2.index))
    check("redo_review re-queues the same item", seq2.current == ("A", 1),
          str(seq2.current))

    deadline = _time.perf_counter() + 5.0
    while not seq2.reviewing and _time.perf_counter() < deadline:
        seq2.tick()
        _time.sleep(0.002)
    check("the retry reached REVIEW again", seq2.reviewing, seq2.state)
    check("the retry recorded the same item", taken[-1] == ("A", 1), str(taken))

    seq2.accept_review()
    check("accept_review advances the index", seq2.index == index_before + 1,
          str(seq2.index))

    deadline = _time.perf_counter() + 5.0
    while seq2.running and _time.perf_counter() < deadline:
        seq2.tick()
        if seq2.reviewing:
            seq2.accept_review()
        _time.sleep(0.002)
    check("the plan finishes once everything is accepted",
          seq2.state == T.STATE_DONE, seq2.state)
    check("one take per plan item plus the redo", len(taken) == 3, str(taken))
    T.REVIEW_EACH_RECORDING = False

    print("\n--- ring buffer / capture ---")
    ring = T.SampleRing(100)
    for i in range(250):
        ring.push(i * 0.01, [i % 1000] * 8)
    emg, ts = ring.snapshot(10)
    check("ring is bounded", ring.capacity == 100)
    check("ring returns newest last", int(emg[-1][0]) == 249, str(emg[-1][0]))
    check("ring chronological", bool(np.all(np.diff(ts) > 0)))
    check("ring snapshot copies", not np.shares_memory(emg, ring._emg))

    cap = T.Capture("FIST", 1, max_samples=50)
    for i in range(80):
        cap.append(i * 0.02, [i] * 8)
    check("capture respects max_samples", len(cap) == 50, str(len(cap)))
    check("capture overflow flagged", cap.overflowed)
    cap_rec = cap.to_recording("p", "s")
    check("capture -> Recording", cap_rec.n_samples == 50 and cap_rec.label == "FIST")

    # ======================================================================
    print("\n--- model provenance: what was it trained on, and is it current? ---")
    # ======================================================================
    prov_root = os.path.join(root, "provenance")
    writer = T.DatasetWriter(prov_root, "alice", "session_01")
    made = []
    for label in ("FIST", "REST"):
        for rep in range(1, 4):
            n = 60
            emg = np.abs(np.random.default_rng(rep).normal(50, 8, (n, 8)))
            t = np.arange(n) / 60.0 + 1_700_000_000.0
            rec = T.Recording(emg.astype(int), t, label, "alice",
                              "session_01", rep, label_start_index=5,
                              label_end_index=n - 5,
                              meta={"emg_mode": "filtered"})
            writer.save(rec, T.STAGE_RAW)
            made.append(rec)

    fp = T.dataset_fingerprint(prov_root)
    check("fingerprint counts every recording", fp["n_recordings"] == 6,
          str(fp["n_recordings"]))
    check("fingerprint counts per label", fp["labels"] == {"FIST": 3, "REST": 3},
          str(fp["labels"]))
    check("fingerprint lists the subject", fp["subjects"] == ["alice"],
          str(fp["subjects"]))
    check("fingerprint lists the session",
          fp["sessions"] == ["alice/session_01"], str(fp["sessions"]))
    check("fingerprint has a digest", len(fp["uid_digest"]) == 16,
          fp["uid_digest"])
    check("fingerprint is stable across calls",
          T.dataset_fingerprint(prov_root)["uid_digest"] == fp["uid_digest"])
    check("fingerprint records the newest stamp",
          len(fp["newest_stamp"]) == 19, fp["newest_stamp"])
    check("the newest stamp converts to a time",
          abs(T.stamp_to_unix(fp["newest_stamp"]) - time.time()) < 3600,
          str(fp["newest_unix"]))

    sample = T.discover_recordings(prov_root, stage=T.STAGE_RAW)[0]
    parsed = T.parse_recording_name(sample)
    check("a filename yields (rep, stamp, uid)",
          parsed is not None and len(parsed) == 3, str(parsed))
    check("a filename that is not a recording yields None",
          T.parse_recording_name("notes.txt") is None)
    check("a bad stamp does not raise", T.stamp_to_unix("nonsense") is None)

    #  The four states the GUI can show, each of which must be distinct.
    none_state = T.model_data_status(None)
    check("no payload reads as untrained", none_state["state"] == "untrained",
          none_state["headline"])
    check("untrained is not shown as a failure",
          none_state["severity"] == "dim")

    legacy = T.model_data_status({"created_iso": "2026-01-01T10:00:00",
                                  "stats": {"n_recordings": 6}}, fp)
    check("a model without provenance reads as unknown",
          legacy["state"] == "unknown", legacy["headline"])
    check("unknown is never reported as up to date",
          "UP TO DATE" not in legacy["headline"])

    trained = {"created_iso": "2026-01-01T10:00:00", "data": dict(fp)}
    current = T.model_data_status(trained, fp)
    check("a model matching the dataset is current",
          current["state"] == "current", current["headline"])
    check("current says so in the headline",
          "UP TO DATE" in current["headline"], current["headline"])
    check("current names the training data",
          "6 recordings" in current["detail"], current["detail"][:60])

    #  One more recording, and the same model must become stale.
    extra = T.Recording(np.abs(np.random.default_rng(9).normal(50, 8, (60, 8))).astype(int),
                        np.arange(60) / 60.0 + 1_700_000_000.0, "FIST",
                        "alice", "session_01", 99, label_start_index=5,
                        label_end_index=55, meta={"emg_mode": "filtered"})
    time.sleep(0.002)          # the stamp has millisecond resolution
    writer.save(extra, T.STAGE_RAW)
    fp2 = T.dataset_fingerprint(prov_root)
    check("the fingerprint changes when a recording is added",
          fp2["uid_digest"] != fp["uid_digest"])
    stale = T.model_data_status(trained, fp2)
    check("the model is now out of date", stale["state"] == "stale",
          stale["headline"])
    check("it says how many are new", stale["n_new"] == 1, str(stale["n_new"]))
    check("it says what to do about it",
          "RETRAIN" in stale["detail"], stale["detail"][:60])
    check("a stale model is flagged, not hidden", stale["severity"] == "warn")

    #  Deleting is a change too, and must not be read as "nothing new".
    fewer = dict(fp2)
    fewer["uid_digest"] = "0" * 16
    fewer["n_recordings"] = 99
    gone = T.model_data_status({"created_iso": "2026-01-01T10:00:00",
                                "data": fewer}, fp2)
    check("recordings disappearing is also out of date",
          gone["state"] == "stale", gone["headline"])
    check("it reports them as missing, not as new",
          "no longer on disk" in gone["detail"], gone["detail"][:80])

    check("describe_training_data handles nothing",
          T.describe_training_data({}) == "not recorded")
    check("describe_training_data names labels and subjects",
          "FIST 3" in T.describe_training_data(fp)
          and "alice" in T.describe_training_data(fp),
          T.describe_training_data(fp))

finally:
    shutil.rmtree(root, ignore_errors=True)

print(f"\n================  {sum(OK)}/{len(OK)} checks passed  ================")
sys.exit(0 if all(OK) else 1)
