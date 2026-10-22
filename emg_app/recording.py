"""
Recording, the writers (csv/npz/parquet/meta), DatasetWriter and the single-
file loaders.
"""

import csv
import json
import os
import re
import time
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import List

import numpy as np

from emg_app import config as cfg
from emg_app.utils import (
    emg_range, ensure_dir, iso_time, mode_value, nominal_rate, safe_slug,
    timestamp_slug, unique_path)
from emg_app.signal_processing import (
    detect_activity, envelope, normalize_per_channel)
from emg_app.quality import QualityIssue, check_quality, measured_rate
from emg_app.markers import (
    Marker, append_marker_jsonl, append_recording_note, load_markers)


# ==============================================================================
# SECTION 4 -- DATASET MODEL AND FILE I/O
#
# Data-preservation rules enforced here:
#   * a *_raw.* file is written exactly once and never rewritten;
#   * unique_path() guarantees nothing is ever overwritten;
#   * trimming/processing produce new files with a new stage suffix;
#   * every file gets a *_meta.json sidecar with the full configuration used.
# ==============================================================================

STAGE_RAW = "raw"
STAGE_TRIMMED = "trimmed"
STAGE_PROCESSED = "processed"
STAGE_AUGMENTED = "augmented"
# A long, unlabelled free recording that clips get cut out of. Deliberately
# not a training stage: no stage selector asks for it, and it lives in the
# _takes/ folder, which discovery prunes.
STAGE_TAKE = "take"

MANIFEST_NAME = "manifest.jsonl"
SESSION_CONFIG_NAME = "session_config.json"
GRAPHS_DIRNAME = "_graphs"

# Where anything taken out of the dataset goes. Nothing in this project calls
# os.remove on recorded signal (safety rule 4) -- DELETE LAST, REDO and
# DELETE TAKE all move files here instead. Starts with "_" so discovery
# prunes it, exactly like _takes.
DELETED_DIRNAME = "_deleted"


def recording_config_snapshot() -> dict:
    """
    Every tunable that could influence the stored signal, captured at save
    time. Without this, a dataset collected weeks apart is not comparable.
    """
    return {
        "app_version": cfg.APP_VERSION,
        "emg_mode": cfg.EMG_MODE,
        "nominal_sample_rate": nominal_rate(),
        "emg_range": list(emg_range()),
        "gestures": list(cfg.GESTURES),
        "repetitions": cfg.REPETITIONS,
        "gesture_order": cfg.GESTURE_ORDER,
        "random_seed": cfg.RANDOM_SEED,
        "rest_time": cfg.REST_TIME,
        "prepare_time": cfg.PREPARE_TIME,
        "countdown_time": cfg.COUNTDOWN_TIME,
        "record_time": cfg.RECORD_TIME,
        "pause_time": cfg.PAUSE_TIME,
        "pre_gesture_ms": cfg.PRE_GESTURE_MS,
        "post_gesture_ms": cfg.POST_GESTURE_MS,
        "window_ms": cfg.WINDOW_MS,
        "window_overlap": cfg.WINDOW_OVERLAP,
        "auto_trim": cfg.AUTO_TRIM,
        "default_cut_seconds": cfg.LABEL_DEFAULT_SELECTION_S,
        "model_spec_file": cfg.MODEL_SPEC_NAME,
        # The mode is a snapshot, not the calibration itself: the record
        # lives in the wearer's folder and is a measurement of the person,
        # not of this recording. Noting the mode is what makes it possible
        # to tell later whether a sitting was recorded while calibration was
        # being applied to the live classifier.
        "calibration_mode": cfg.CALIBRATION_MODE if cfg.CALIBRATION_ENABLED else "off",
        "calibration_target_level": cfg.CALIBRATION_TARGET_LEVEL,
        "min_selection_seconds": cfg.LABEL_MIN_SELECTION_S,
        "free_take_max_minutes": cfg.FREE_TAKE_MAX_MINUTES,
        "trim": {
            "envelope_ms": cfg.TRIM_ENVELOPE_MS,
            "baseline_ms": cfg.TRIM_BASELINE_MS,
            "threshold_high": cfg.TRIM_THRESHOLD_HIGH,
            "threshold_low": cfg.TRIM_THRESHOLD_LOW,
            "abs_floor": mode_value(cfg.TRIM_ABS_FLOOR),
            "min_active_ms": cfg.TRIM_MIN_ACTIVE_MS,
            "min_inactive_ms": cfg.TRIM_MIN_INACTIVE_MS,
            "pre_padding_ms": cfg.PRE_PADDING_MS,
            "post_padding_ms": cfg.POST_PADDING_MS,
            "min_active_channels": cfg.TRIM_MIN_ACTIVE_CHANNELS,
        },
        "quality": {
            "enabled": cfg.QC_ENABLED,
            "min_sample_fraction": cfg.QC_MIN_SAMPLE_FRACTION,
            "max_gap_ms": cfg.QC_MAX_GAP_MS,
            "max_saturated_fraction": cfg.QC_MAX_SATURATED_FRACTION,
            "max_baseline_rms": mode_value(cfg.QC_MAX_BASELINE_RMS),
            "min_peak_amplitude": mode_value(cfg.QC_MIN_PEAK_AMPLITUDE),
        },
    }


class Recording:
    """
    One labeled EMG recording held in memory.

    `emg` is the untouched device signal: `(n_samples, n_channels)` integer
    counts exactly as the band reported them. Nothing in this class mutates
    it -- every derived view and every trim returns a new array or a new
    Recording.

    **The channel count comes from the DATA, not from the config.** A 2-D
    array keeps its own width, so a recording made on an 8-channel band still
    loads as eight channels after `EMG_DEVICE` moves to a wider one. Only a
    flat array has to be told how to fold, and then the config is the only
    thing that can say.
    """

    def __init__(self, emg, timestamps, label, person_id, session_id,
                 repetition=0, stage=STAGE_RAW, uid=None, created_unix=None,
                 label_start_index=0, label_end_index=None, meta=None,
                 source=None, n_channels=None):
        incoming = np.asarray(emg, dtype=np.int32)
        if incoming.ndim >= 2 and incoming.shape[-1] > 0:
            width = int(incoming.shape[-1])
        else:
            width = int(n_channels or cfg.N_CHANNELS)
        self.emg = np.ascontiguousarray(
            incoming.reshape(-1, width)
        )
        self.timestamps = np.asarray(timestamps, dtype=np.float64).reshape(-1)
        if self.timestamps.size != self.emg.shape[0]:
            raise ValueError(
                f"timestamp count ({self.timestamps.size}) does not match "
                f"sample count ({self.emg.shape[0]})"
            )

        self.label = str(label)
        self.person_id = str(person_id)
        self.session_id = str(session_id)
        self.repetition = int(repetition)
        self.stage = str(stage)
        self.uid = uid or uuid.uuid4().hex[:8]
        self.created_unix = float(
            created_unix if created_unix is not None else time.time()
        )

        # The cue window inside `emg`, i.e. where the operator asked for the
        # gesture. PRE/POST_GESTURE_MS of context sits outside it.
        self.label_start_index = int(label_start_index)
        self.label_end_index = int(
            self.emg.shape[0] if label_end_index is None else label_end_index
        )

        self.meta = dict(meta or {})
        self.source = source                 # path this was loaded from, if any
        self.files = {}                      # "stage:format" -> written path
        self.issues = []                     # List[QualityIssue]
        self.activity = None                 # Optional[ActivityResult]

    def primary_path(self, root: str = "", stage: str = None):
        """
        Absolute path of this recording's main data file, or None if it has
        not been written yet. CSV first because it is the human-readable one
        and is written by default; NPZ is the fallback when SAVE_CSV is off.

        `self.files` holds paths relative to the dataset root, so `root` has
        to be the same root the DatasetWriter used.
        """
        stage = stage or self.stage
        for key in (f"{stage}:csv", f"{stage}:npz",
                    f"{STAGE_RAW}:csv", f"{STAGE_RAW}:npz"):
            relative = self.files.get(key)
            if relative:
                return os.path.join(root, relative) if root else relative
        return os.path.abspath(self.source) if self.source else None

    # -- basics ---------------------------------------------------------------

    def __len__(self) -> int:
        return int(self.emg.shape[0])

    def __repr__(self) -> str:
        return (f"Recording({self.person_id}/{self.session_id} {self.label} "
                f"rep{self.repetition} {len(self)} samples {self.stage})")

    @property
    def n_samples(self) -> int:
        return int(self.emg.shape[0])

    @property
    def n_channels(self) -> int:
        """
        How wide THIS recording is -- read off the array, never the config.

        Everything that writes or describes a recording uses this, so a file
        recorded on one band keeps reporting its own channel count after the
        config has moved to another.
        """
        return int(self.emg.shape[1])

    @property
    def channel_names(self) -> List[str]:
        """Captions sized to this recording, not to the active profile."""
        return cfg.channel_names_for(self.n_channels)

    @property
    def fs(self) -> float:
        return measured_rate(self.timestamps,
                             nominal_rate(self.meta.get("emg_mode")))

    @property
    def duration(self) -> float:
        if self.timestamps.size < 2:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])

    @property
    def t_rel(self) -> np.ndarray:
        if self.timestamps.size == 0:
            return self.timestamps.copy()
        return self.timestamps - self.timestamps[0]

    @property
    def worst_issue_level(self):
        for level in ("error", "warning", "info"):
            if any(i.level == level for i in self.issues):
                return level
        return None

    # -- derived views (always new arrays) ------------------------------------

    def envelope(self, win_ms: float = None) -> np.ndarray:
        return envelope(self.emg, self.fs, win_ms)

    def normalized(self, mode: str = "mad"):
        return normalize_per_channel(self.emg, mode)

    def detect_activity(self):
        self.activity = detect_activity(
            self.emg, self.fs, self.meta.get("emg_mode", cfg.EMG_MODE)
        )
        return self.activity

    def run_quality_checks(self, expected_seconds: float = None,
                           had_disconnect: bool = False):
        self.issues = check_quality(
            self.emg, self.timestamps, self.label,
            expected_seconds=expected_seconds,
            mode=self.meta.get("emg_mode", cfg.EMG_MODE),
            had_disconnect=had_disconnect,
        )
        return self.issues

    # -- trimming -------------------------------------------------------------

    def slice(self, start: int, end: int, stage: str = STAGE_TRIMMED,
              extra_meta: dict = None):
        """
        Return a NEW Recording covering [start, end). The original is untouched,
        which is the whole point: slice() can never lose raw data.
        """
        n = self.n_samples
        start = int(np.clip(start, 0, max(0, n - 1)))
        end = int(np.clip(end, start + 1, n))

        meta = dict(self.meta)
        meta.update({
            "parent_uid": self.uid,
            "parent_stage": self.stage,
            "parent_n_samples": n,
            "parent_raw_file": (self.files.get(STAGE_RAW + ":csv")
                                or self.files.get(STAGE_RAW + ":npz")
                                or self.source),
            "trim_start_index": start,
            "trim_end_index": end,
            "trim_start_s": float(self.t_rel[start]) if n else 0.0,
            "trim_end_s": float(self.t_rel[end - 1]) if n else 0.0,
            "trim_removed_samples": n - (end - start),
        })
        meta.update(extra_meta or {})

        child = Recording(
            self.emg[start:end].copy(),
            self.timestamps[start:end].copy(),
            label=self.label,
            person_id=self.person_id,
            session_id=self.session_id,
            repetition=self.repetition,
            stage=stage,
            uid=self.uid,                 # keeps raw/trimmed pairs together
            created_unix=self.created_unix,
            label_start_index=max(0, self.label_start_index - start),
            label_end_index=min(end - start, max(1, self.label_end_index - start)),
            meta=meta,
        )
        child.files = dict(self.files)
        child.issues = list(self.issues)
        return child

    def auto_trimmed(self):
        """Apply automatic activity detection, or None if nothing was found."""
        if self.label in cfg.AUTO_TRIM_SKIP_LABELS:
            return None
        result = self.activity or self.detect_activity()
        if not result.found:
            return None
        return self.slice(result.start, result.end, STAGE_TRIMMED, {
            "trim_method": "auto",
            "trim_core_start_index": result.core_start,
            "trim_core_end_index": result.core_end,
            "trim_threshold_high": result.threshold_high,
            "trim_threshold_low": result.threshold_low,
            "trim_baseline": result.baseline,
        })

    # -- serialisation --------------------------------------------------------

    def metadata_dict(self) -> dict:
        """Everything about the recording except the samples themselves."""
        return {
            "uid": self.uid,
            "stage": self.stage,
            "label": self.label,
            "label_index": cfg.GESTURES.index(self.label) if self.label in cfg.GESTURES else -1,
            "person_id": self.person_id,
            "session_id": self.session_id,
            "repetition": self.repetition,
            "n_samples": self.n_samples,
            # Read off this recording, not off the active profile: a file
            # recorded on one band must keep describing itself correctly
            # after the config moves to another.
            "n_channels": self.n_channels,
            "channel_names": self.channel_names,
            "duration_s": round(self.duration, 6),
            "sample_rate_measured": round(self.fs, 3),
            "created_unix": self.created_unix,
            "created_iso": iso_time(self.created_unix),
            "start_unix": float(self.timestamps[0]) if self.n_samples else None,
            "end_unix": float(self.timestamps[-1]) if self.n_samples else None,
            "start_iso": iso_time(self.timestamps[0]) if self.n_samples else None,
            "end_iso": iso_time(self.timestamps[-1]) if self.n_samples else None,
            "label_start_index": self.label_start_index,
            "label_end_index": self.label_end_index,
            "amplitude_min": self.emg.min(axis=0).tolist() if self.n_samples else [],
            "amplitude_max": self.emg.max(axis=0).tolist() if self.n_samples else [],
            "quality_issues": [asdict(i) for i in self.issues],
            "quality_worst_level": self.worst_issue_level,
            "activity_detected": (asdict(self.activity) if self.activity else None),
            "files": dict(self.files),
            "meta": dict(self.meta),
            "config": recording_config_snapshot(),
        }

    def to_dataframe(self):
        """Tidy one-row-per-sample DataFrame (needs pandas)."""
        if not cfg.HAVE_PANDAS:
            raise RuntimeError("pandas is required for to_dataframe()")
        data = {
            "sample_index": np.arange(self.n_samples, dtype=np.int64),
            "t_rel_s": self.t_rel.astype(np.float64),
            "t_unix": self.timestamps,
        }
        for c, name in enumerate(self.channel_names):
            data[name.lower()] = self.emg[:, c]
        if cfg.CSV_INCLUDE_LABEL_COLUMNS:
            idx = np.arange(self.n_samples)
            data["in_label_window"] = (
                (idx >= self.label_start_index) & (idx < self.label_end_index)
            ).astype(np.int8)
            data["gesture"] = self.label
            data["person_id"] = self.person_id
            data["session_id"] = self.session_id
            data["repetition"] = self.repetition
        return cfg._pandas().DataFrame(data)


# ------------------------------------------------------------------------------
# Writing
# ------------------------------------------------------------------------------


def csv_header(n_channels: int = None):
    """
    The CSV column list for an `n_channels`-wide recording.

    Parameterised rather than global so a recording writes the columns IT
    has. A header that claimed eight channels for a sixteen-channel
    recording would produce a file whose rows and header disagree, which
    `load_recording_csv` has no way to detect.
    """
    names = cfg.channel_names_for(n_channels if n_channels is not None
                              else cfg.N_CHANNELS)
    cols = ["sample_index", "t_rel_s", "t_unix"] + [c.lower() for c in names]
    if cfg.CSV_INCLUDE_LABEL_COLUMNS:
        cols += ["in_label_window", "gesture", "person_id", "session_id", "repetition"]
    return cols


def write_recording_csv(rec: Recording, path: str) -> str:
    """
    Stream the samples out with the csv module.

    Integers are written as integers (no 1023.000000 noise) and timestamps keep
    microsecond resolution, so a round-trip through CSV is lossless.
    """
    path = unique_path(path)
    ensure_dir(os.path.dirname(path))
    t_rel = rec.t_rel
    emg = rec.emg

    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(csv_header(rec.n_channels))
        tail = ([rec.label, rec.person_id, rec.session_id, rec.repetition]
                if cfg.CSV_INCLUDE_LABEL_COLUMNS else [])
        for i in range(rec.n_samples):
            row = [i, f"{t_rel[i]:.6f}", f"{rec.timestamps[i]:.6f}"]
            row.extend(int(v) for v in emg[i])
            if cfg.CSV_INCLUDE_LABEL_COLUMNS:
                row.append(1 if rec.label_start_index <= i < rec.label_end_index else 0)
                row.extend(tail)
            writer.writerow(row)
    return path


def write_recording_npz(rec: Recording, path: str) -> str:
    """Compressed numpy archive: X, timestamps and the metadata as JSON."""
    path = unique_path(path)
    ensure_dir(os.path.dirname(path))
    np.savez_compressed(
        path,
        emg=rec.emg.astype(np.int32),
        t_unix=rec.timestamps,
        t_rel=rec.t_rel.astype(np.float32),
        label=np.array(rec.label),
        person_id=np.array(rec.person_id),
        session_id=np.array(rec.session_id),
        repetition=np.array(rec.repetition),
        channel_names=np.array(rec.channel_names),
        metadata_json=np.array(json.dumps(rec.metadata_dict(), default=str)),
    )
    return path


def write_recording_parquet(rec: Recording, path: str):
    if not cfg.HAVE_PANDAS:
        return None
    path = unique_path(path)
    ensure_dir(os.path.dirname(path))
    try:
        rec.to_dataframe().to_parquet(path, index=False)
    except Exception as exc:                                  # pragma: no cover
        print(f"[warn] parquet export failed ({exc}); CSV/NPZ are unaffected")
        return None
    return path


def write_metadata_json(rec: Recording, path: str) -> str:
    path = unique_path(path)
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(rec.metadata_dict(), fh, indent=2, ensure_ascii=False, default=str)
    return path


class DatasetWriter:
    """
    Owns the on-disk layout:

        <root>/<person_id>/<session_id>/session_config.json
        <root>/<person_id>/<session_id>/manifest.jsonl
        <root>/<person_id>/<session_id>/<GESTURE>/<base>_raw.csv
        <root>/<person_id>/<session_id>/<GESTURE>/<base>_raw.npz
        <root>/<person_id>/<session_id>/<GESTURE>/<base>_raw_meta.json
        <root>/<person_id>/<session_id>/<GESTURE>/<base>_trimmed.csv
        <root>/<person_id>/<session_id>/_graphs/<name>.png

    People stay in separate top-level folders, which is what makes honest
    leave-one-person-out validation possible later.
    """

    def __init__(self, root: str, person_id: str, session_id: str):
        self.root = os.path.abspath(root)
        self.person_id = safe_slug(person_id, "person")
        self.session_id = safe_slug(session_id, "session")
        ensure_dir(self.session_dir)

    # -- paths ----------------------------------------------------------------

    @property
    def person_dir(self) -> str:
        return os.path.join(self.root, self.person_id)

    @property
    def session_dir(self) -> str:
        return os.path.join(self.person_dir, self.session_id)

    @property
    def manifest_path(self) -> str:
        return os.path.join(self.session_dir, MANIFEST_NAME)

    @property
    def graphs_dir(self) -> str:
        return ensure_dir(os.path.join(self.session_dir, GRAPHS_DIRNAME))

    @property
    def takes_dir(self) -> str:
        """
        Where long free takes live. Starts with "_" on purpose: discovery
        prunes those folders, so a take can never be mistaken for training
        data no matter what stage selector is used.
        """
        return ensure_dir(os.path.join(self.session_dir, cfg.FREE_TAKE_FOLDER))

    def gesture_dir(self, label: str) -> str:
        return ensure_dir(os.path.join(self.session_dir, safe_slug(label, "UNLABELED")))

    def base_name(self, rec: Recording) -> str:
        return (f"{safe_slug(rec.label, 'UNLABELED')}"
                f"_rep{rec.repetition:03d}"
                f"_{timestamp_slug(rec.created_unix)}"
                f"_{rec.uid}")

    # -- writing --------------------------------------------------------------

    def write_session_config(self, extra: dict = None) -> str:
        """Session-level snapshot, refreshed whenever a session starts."""
        path = os.path.join(self.session_dir, SESSION_CONFIG_NAME)
        payload = {
            "person_id": self.person_id,
            "session_id": self.session_id,
            "written_iso": iso_time(time.time()),
            "handedness": cfg.HANDEDNESS,
            "arm_worn_on": cfg.ARM_WORN_ON,
            "device": cfg.DEVICE_NAME,
            "subject_notes": cfg.SUBJECT_NOTES,
            "channel_names": list(cfg.CHANNEL_NAMES),
            "config": recording_config_snapshot(),
        }
        payload.update(extra or {})
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False, default=str)
        return path

    def save(self, rec: Recording, stage: str = None):
        """
        Write one recording in every enabled format and append it to the
        manifest. Returns {format: path}. Existing files are never touched.
        """
        stage = stage or rec.stage
        folder = self.gesture_dir(rec.label)
        base = os.path.join(folder, f"{self.base_name(rec)}_{stage}")
        written = {}

        if cfg.SAVE_CSV:
            written["csv"] = write_recording_csv(rec, base + ".csv")
        if cfg.SAVE_NPZ:
            written["npz"] = write_recording_npz(rec, base + ".npz")
        if cfg.SAVE_PARQUET:
            parquet = write_recording_parquet(rec, base + ".parquet")
            if parquet:
                written["parquet"] = parquet

        for fmt, path in written.items():
            rec.files[f"{stage}:{fmt}"] = os.path.relpath(path, self.root)

        if cfg.SAVE_META_JSON:
            written["meta"] = write_metadata_json(rec, base + "_meta.json")
            rec.files[f"{stage}:meta"] = os.path.relpath(written["meta"], self.root)

        self.append_manifest(rec, stage, written)
        return written

    def append_manifest(self, rec: Recording, stage: str, written: dict) -> None:
        """One JSON line per written file set -- append-only, never rewritten."""
        entry = {
            "uid": rec.uid,
            "stage": stage,
            "label": rec.label,
            "repetition": rec.repetition,
            "person_id": rec.person_id,
            "session_id": rec.session_id,
            "n_samples": rec.n_samples,
            "duration_s": round(rec.duration, 4),
            "sample_rate_measured": round(rec.fs, 2),
            "created_iso": iso_time(rec.created_unix),
            "saved_iso": iso_time(time.time()),
            "quality_worst_level": rec.worst_issue_level,
            "quality_codes": [i.code for i in rec.issues],
            "notes": rec.meta.get("notes", ""),
            "files": {k: os.path.relpath(v, self.root) for k, v in written.items()},
        }
        with open(self.manifest_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")

    def read_manifest(self):
        """All manifest entries, oldest first. Returns [] if there is none."""
        if not os.path.exists(self.manifest_path):
            return []
        entries = []
        with open(self.manifest_path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return entries

    def count_existing(self, label: str = None) -> int:
        """How many raw recordings are already on disk for this session."""
        seen = set()
        for entry in self.read_manifest():
            if entry.get("stage") != STAGE_RAW:
                continue
            if label and entry.get("label") != label:
                continue
            seen.add(entry.get("uid"))
        return len(seen)

    def next_repetition(self, label: str) -> int:
        """Next free repetition number for a label, so restarts never collide."""
        return self.count_existing(label) + 1

    # -- free takes -----------------------------------------------------------

    def next_take_number(self) -> int:
        """Next free take number for this session, counted from the manifest."""
        seen = {entry.get("uid") for entry in self.read_manifest()
                if entry.get("stage") == STAGE_TAKE}
        return len(seen) + 1

    def take_base_name(self, rec: Recording, number: int) -> str:
        """`FIST_take003_20261012-164530-123_a1b2c3d4` -- no rep number.

        A take has no repetition: it is one continuous stretch, not the Nth
        instance of anything. The gesture in the name is the INTENDED one,
        recorded so you know what you sat down to collect.
        """
        return (f"{safe_slug(rec.label, 'TAKE')}"
                f"_take{int(number):03d}"
                f"_{timestamp_slug(rec.created_unix)}"
                f"_{rec.uid}")

    def save_take(self, rec: Recording, number: int = None):
        """
        Write one long free take into _takes/. Returns {format: path}.

        Identical mechanics to save() -- unique_path, every enabled format, a
        metadata sidecar and a manifest line -- but a different folder and the
        "take" stage, which together keep it out of the dataset.
        """
        number = self.next_take_number() if number is None else int(number)
        rec.stage = STAGE_TAKE
        base = os.path.join(self.takes_dir,
                            f"{self.take_base_name(rec, number)}_{STAGE_TAKE}")
        written = {}

        if cfg.SAVE_CSV:
            written["csv"] = write_recording_csv(rec, base + ".csv")
        if cfg.SAVE_NPZ:
            written["npz"] = write_recording_npz(rec, base + ".npz")
        if cfg.SAVE_PARQUET:
            parquet = write_recording_parquet(rec, base + ".parquet")
            if parquet:
                written["parquet"] = parquet

        for fmt, path in written.items():
            rec.files[f"{STAGE_TAKE}:{fmt}"] = os.path.relpath(path, self.root)

        if cfg.SAVE_META_JSON:
            written["meta"] = write_metadata_json(rec, base + "_meta.json")
            rec.files[f"{STAGE_TAKE}:meta"] = os.path.relpath(written["meta"],
                                                              self.root)

        self.append_manifest(rec, STAGE_TAKE, written)
        return written

    def list_takes(self):
        """
        Every take on disk for this session, newest first.

        Read from the folder rather than from the manifest, so takes recorded
        in an earlier run of the app -- or copied in from another machine --
        still show up in the history list.
        """
        folder = os.path.join(self.session_dir, cfg.FREE_TAKE_FOLDER)
        if not os.path.isdir(folder):
            return []

        # One entry per take, preferring .npz for loading (smaller, faster).
        by_base = {}
        for name in sorted(os.listdir(folder)):
            stem, ext = os.path.splitext(name)
            ext = ext.lower().lstrip(".")
            if ext not in ("npz", "csv", "parquet") or stem.endswith("_meta"):
                continue
            if _stage_from_filename(name) != STAGE_TAKE:
                continue
            by_base.setdefault(stem, {})[ext] = os.path.join(folder, name)

        takes = []
        for stem in sorted(by_base):
            formats = by_base[stem]
            path = (formats.get("npz") or formats.get("csv")
                    or formats.get("parquet"))
            info = {
                "stem": stem, "path": path, "formats": formats,
                "label": "", "uid": "", "created_unix": 0.0, "n_samples": 0,
                "duration_s": 0.0, "notes": "",
            }
            sidecar = os.path.join(folder, stem + "_meta.json")
            if os.path.exists(sidecar):
                try:
                    with open(sidecar, "r", encoding="utf-8") as handle:
                        meta = json.load(handle)
                    info.update({
                        "label": meta.get("label", ""),
                        "uid": meta.get("uid", ""),
                        "created_unix": meta.get("created_unix") or 0.0,
                        "n_samples": meta.get("n_samples", 0),
                        "duration_s": meta.get("duration_s", 0.0),
                        "notes": (meta.get("meta") or {}).get("notes", ""),
                    })
                except (json.JSONDecodeError, OSError):      # pragma: no cover
                    pass                                     # fall back to the name
            if not info["label"]:
                info["label"] = stem.split("_take")[0]
            if not info["uid"]:
                info["uid"] = take_uid_from_stem(stem)
            if not info["created_unix"]:
                info["created_unix"] = os.path.getmtime(path)
            takes.append(info)

        takes.sort(key=lambda t: t["created_unix"], reverse=True)
        return takes

    def clips_from_take(self, take_uid: str):
        """
        Paths of the recordings that were cut out of this take.

        Read from the clips' own metadata (`meta.parent_take_uid`), which is
        where EXTRACT CLIP writes provenance. Used to warn before a take is
        deleted: the clips are independent files and survive, but their
        `parent_take_file` then points into _deleted/.
        """
        take_uid = (take_uid or "").lower()
        if not take_uid or not os.path.isdir(self.session_dir):
            return []

        found = []
        for entry in sorted(os.listdir(self.session_dir)):
            folder = os.path.join(self.session_dir, entry)
            # Same rule as discovery: underscore folders are not the dataset.
            if entry.startswith("_") or not os.path.isdir(folder):
                continue
            for name in sorted(os.listdir(folder)):
                if not name.endswith("_meta.json"):
                    continue
                try:
                    with open(os.path.join(folder, name), "r",
                              encoding="utf-8") as handle:
                        meta = json.load(handle)
                except (json.JSONDecodeError, OSError):      # pragma: no cover
                    continue
                parent = (meta.get("meta") or {}).get("parent_take_uid") or ""
                if str(parent).lower() == take_uid:
                    found.append(os.path.join(folder, name))
        return found

    def delete_take(self, stem: str) -> dict:
        """
        Move one take's files out of _takes/ and into _deleted/.

        Deliberately not os.remove (safety rule 4): a take is minutes of a
        subject's time and the one thing clips cannot be re-cut from. A
        mis-click costs a folder move.

        Everything written for the take goes together -- csv, npz, parquet,
        the metadata sidecar and any notes sidecar -- matched as `<stem>.`
        or `<stem>_` so a different take whose name merely starts the same
        cannot be caught. Clips already cut from it are NOT touched: they are
        ordinary recordings in the gesture folders with their own uids, and
        deleting the take they came from does not make them wrong.

        Returns {"stem", "uid", "label", "moved": [(from, to), ...]}; `moved`
        is empty when nothing matched.
        """
        folder = os.path.join(self.session_dir, cfg.FREE_TAKE_FOLDER)
        stem = os.path.basename(str(stem))
        result = {"stem": stem, "uid": take_uid_from_stem(stem),
                  "label": stem.split("_take")[0], "moved": []}
        if not stem or not os.path.isdir(folder):
            return result

        names = [n for n in sorted(os.listdir(folder))
                 if n.startswith(stem + ".") or n.startswith(stem + "_")]
        if not names:
            return result

        # The label and uid from the sidecar beat the ones parsed out of the
        # filename, for the manifest line below.
        sidecar = os.path.join(folder, stem + "_meta.json")
        if os.path.exists(sidecar):
            try:
                with open(sidecar, "r", encoding="utf-8") as handle:
                    meta = json.load(handle)
                result["label"] = meta.get("label") or result["label"]
                result["uid"] = meta.get("uid") or result["uid"]
            except (json.JSONDecodeError, OSError):          # pragma: no cover
                pass

        target = ensure_dir(os.path.join(self.session_dir, DELETED_DIRNAME))
        for name in names:
            source = os.path.join(folder, name)
            destination = unique_path(os.path.join(target, name))
            try:
                os.replace(source, destination)
            except OSError:                                  # pragma: no cover
                continue
            result["moved"].append((
                os.path.relpath(source, self.root),
                os.path.relpath(destination, self.root),
            ))

        if result["moved"]:
            entry = {
                "uid": result["uid"],
                "stage": "deleted",
                "deleted_stage": STAGE_TAKE,
                "label": result["label"],
                "person_id": self.person_id,
                "session_id": self.session_id,
                "take_stem": stem,
                "deleted_iso": iso_time(time.time()),
                "files": {source: destination
                          for source, destination in result["moved"]},
            }
            with open(self.manifest_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False,
                                    default=str) + "\n")
        return result

    # -- markers and notes ----------------------------------------------------

    @property
    def markers_path(self) -> str:
        return os.path.join(self.session_dir, cfg.MARKERS_NAME)

    def append_marker(self, marker) -> str:
        """Append one marker to the session log. Never rewrites the file."""
        ensure_dir(self.session_dir)
        return append_marker_jsonl(self.markers_path, marker)

    def load_markers(self):
        """Every marker logged for this session, oldest first."""
        return load_markers(self.markers_path)

    def add_note_to_recording(self, rec, text: str, t_unix: float = None,
                              kind: str = "note"):
        """
        Attach a note to an already-saved recording.

        Writes a sidecar next to the recording's data file; the data file
        itself is never opened, let alone rewritten (safety rule 1). Returns
        (sidecar_path, marker).
        """
        target = rec.primary_path(self.root)
        if not target:
            raise ValueError("this recording has no written file to annotate")
        marker = Marker(
            t_unix=rec.created_unix if t_unix is None else t_unix,
            text=text, kind=kind,
            person_id=rec.person_id, session_id=rec.session_id,
        )
        path = append_recording_note(target, marker, rec_uid=rec.uid)
        return path, marker


def load_recording_npz(path: str) -> Recording:
    """Load a recording written by write_recording_npz()."""
    with np.load(path, allow_pickle=False) as data:
        meta = {}
        if "metadata_json" in data:
            try:
                meta = json.loads(str(data["metadata_json"]))
            except (json.JSONDecodeError, TypeError):
                meta = {}

        rec = Recording(
            emg=data["emg"],
            timestamps=data["t_unix"],
            label=str(data["label"]),
            person_id=str(data["person_id"]),
            session_id=str(data["session_id"]),
            repetition=int(data["repetition"]),
            stage=meta.get("stage", STAGE_RAW),
            uid=meta.get("uid"),
            created_unix=meta.get("created_unix"),
            label_start_index=meta.get("label_start_index", 0),
            label_end_index=meta.get("label_end_index"),
            meta=meta.get("meta", {}),
            source=path,
        )
    rec.issues = [QualityIssue(**i) for i in meta.get("quality_issues", [])]
    rec.files = meta.get("files", {})
    return rec


def load_recording_csv(path: str) -> Recording:
    """
    Load a recording from CSV. Used as the fallback when NPZ is disabled, and
    as the canonical reader for hand-edited files.
    """
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
        #  Take the channel columns from the HEADER, in file order, rather
        #  than from the active profile. The file knows how wide it is, and a
        #  recording made on an 8-channel band has to keep loading after
        #  EMG_DEVICE moves to a wider one -- otherwise changing the config
        #  would silently make the existing dataset unreadable. Sorted
        #  numerically so ch10 does not land between ch1 and ch2.
        cols = sorted((c for c in header
                       if re.fullmatch(r"ch\d+", (c or "").strip().lower())),
                      key=lambda c: int(c.strip().lower()[2:]))
        if not cols:
            #  No chN columns at all: fall back to the profile's names, which
            #  is what a hand-written file with custom captions would need.
            cols = [c.lower() for c in cfg.CHANNEL_NAMES]
            missing = [c for c in cols if c not in header]
            if missing:
                raise ValueError(f"{path}: missing channel columns {missing}")

        emg, times = [], []
        label = session = person = ""
        repetition = 0
        label_window = []
        for row in reader:
            emg.append([int(float(row[c])) for c in cols])
            times.append(float(row.get("t_unix") or row.get("t_rel_s") or 0.0))
            if "gesture" in row:
                label = row["gesture"]
                person = row.get("person_id", "")
                session = row.get("session_id", "")
                repetition = int(float(row.get("repetition") or 0))
            if "in_label_window" in row:
                label_window.append(int(float(row["in_label_window"])))

    # Fall back to the filename when the CSV carries no label columns.
    if not label:
        label, person, session, repetition = _parse_path_identity(path)

    start = end = None
    if label_window and any(label_window):
        marked = np.flatnonzero(np.asarray(label_window))
        start, end = int(marked[0]), int(marked[-1]) + 1

    meta_path = _sidecar_meta_path(path)
    meta = {}
    if meta_path and os.path.exists(meta_path):
        try:
            with open(meta_path, "r", encoding="utf-8") as fh:
                meta = json.load(fh)
        except (json.JSONDecodeError, OSError):
            meta = {}

    rec = Recording(
        emg=emg or np.zeros((0, cfg.N_CHANNELS)),
        timestamps=times or [],
        label=label, person_id=person, session_id=session, repetition=repetition,
        stage=meta.get("stage", _stage_from_filename(path)),
        uid=meta.get("uid"),
        created_unix=meta.get("created_unix"),
        label_start_index=start if start is not None else meta.get("label_start_index", 0),
        label_end_index=end if end is not None else meta.get("label_end_index"),
        meta=meta.get("meta", {}),
        source=path,
    )
    rec.issues = [QualityIssue(**i) for i in meta.get("quality_issues", [])]
    return rec


def load_recording(path: str) -> Recording:
    """Load a recording from .npz, .csv or .parquet by extension."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npz":
        return load_recording_npz(path)
    if ext == ".csv":
        return load_recording_csv(path)
    if ext == ".parquet":
        if not cfg.HAVE_PANDAS:
            raise RuntimeError("pandas is required to read parquet recordings")
        frame = cfg._pandas().read_parquet(path)
        label, person, session, repetition = _parse_path_identity(path)
        return Recording(
            emg=frame[[c.lower() for c in cfg.CHANNEL_NAMES]].to_numpy(),
            timestamps=frame["t_unix"].to_numpy(),
            label=frame["gesture"].iloc[0] if "gesture" in frame else label,
            person_id=frame["person_id"].iloc[0] if "person_id" in frame else person,
            session_id=frame["session_id"].iloc[0] if "session_id" in frame else session,
            repetition=int(frame["repetition"].iloc[0]) if "repetition" in frame else repetition,
            stage=_stage_from_filename(path),
            source=path,
        )
    raise ValueError(f"unsupported recording format: {path}")


#   ..._raw.csv  ..._trimmed.csv  ..._trimmed_2.csv  (the _2 comes from
#   unique_path() refusing to overwrite an existing file)
STAGE_SUFFIX_RE = re.compile(
    r"_(?P<stage>raw|trimmed|processed|augmented|take)(?:_\d+)?$"
)


def _stage_from_filename(path: str) -> str:
    """
    Read the stage back out of a filename.

    The optional trailing _2/_3 matters: unique_path() adds it when a name is
    already taken, and without handling it a second trimmed file would be
    mistaken for a raw one and quietly pulled into the dataset as raw data.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    match = STAGE_SUFFIX_RE.search(stem)
    return match.group("stage") if match else STAGE_RAW


def _sidecar_meta_path(path: str):
    stem = os.path.splitext(path)[0]
    return stem + "_meta.json"


def _parse_path_identity(path: str):
    """
    Recover (label, person, session, repetition) from the folder layout:
        .../<person>/<session>/<GESTURE>/<GESTURE>_rep007_...csv
    """
    parts = os.path.abspath(path).replace("\\", "/").split("/")
    label = parts[-2] if len(parts) >= 2 else ""
    session = parts[-3] if len(parts) >= 3 else ""
    person = parts[-4] if len(parts) >= 4 else ""
    match = re.search(r"_rep(\d+)_", os.path.basename(path))
    return label, person, session, int(match.group(1)) if match else 0


# ------------------------------------------------------------------------------
# MODEL PROVENANCE  (is a model trained, on what, and is it still current?)
# ------------------------------------------------------------------------------
#
# "Is this model up to date?" is answerable only if the model records which
# recordings went into it. So the trainers write a `data` block into the
# payload, and the app compares it against the dataset on disk.
#
# The comparison works off FILENAMES only -- every recording's uid and
# timestamp are in its name -- so it costs a directory walk, not a load of
# every signal file. That matters: this runs on a timer in the GUI.
#
# When the model predates the `data` block, the answer is "this model does
# not say", never a guess. An unknown provenance shown as "up to date" would
# be the worst possible outcome.

RECORDING_NAME_RE = re.compile(
    r"_rep(\d+)_(\d{8}-\d{6}-\d{3})_([0-9a-fA-F]+)_")

# Takes are named <LABEL>_takeNNN_<stamp>_<uid>_take, so RECORDING_NAME_RE
# (which wants _repNNN_) does not match them.
TAKE_NAME_RE = re.compile(
    r"_take(\d+)_(\d{8}-\d{6}-\d{3})_([0-9a-fA-F]+)")


def take_uid_from_stem(stem: str) -> str:
    """
    A take's uid recovered from its filename, or "" if the name is not one.

    The sidecar's `uid` is authoritative; this is the fallback for a take
    whose metadata file is missing or unreadable, so that deleting it still
    records *which* take went.
    """
    match = TAKE_NAME_RE.search(os.path.basename(str(stem)))
    return match.group(3).lower() if match else ""


def parse_recording_name(path: str):
    """
    `(repetition, stamp, uid)` from a recording's filename, or `None`.

    The filename is the cheapest identity we have: `unique_path()` guarantees
    it, and nothing rewrites it afterwards.
    """
    match = RECORDING_NAME_RE.search(os.path.basename(path))
    if not match:
        return None
    return int(match.group(1)), match.group(2), match.group(3).lower()


def stamp_to_unix(stamp: str):
    """`20261013-225148-614` -> unix seconds, or None if it is not one."""
    try:
        base = datetime.strptime(stamp[:15], "%Y%m%d-%H%M%S")
    except (ValueError, TypeError):
        return None
    millis = 0
    if len(stamp) >= 19:
        try:
            millis = int(stamp[16:19])
        except ValueError:
            millis = 0
    return base.timestamp() + millis / 1000.0
