# Dataset format

Everything written by `emg_dataset_tool.py`. Nothing here is ever rewritten in
place; new information always means a new file.

---

## 1. Folder layout

```
datasets/
  person_01/                       one folder per SUBJECT
    session_01/                    one folder per ARMBAND PLACEMENT
      session_config.json
      manifest.jsonl
      markers.jsonl
      _takes/                      long unlabelled free takes
        FIST_take001_20261012-164530-123_a1b2c3_take.csv
        FIST_take001_20261012-164530-123_a1b2c3_take.npz
        FIST_take001_20261012-164530-123_a1b2c3_take_meta.json
      REST/
      FIST/
        FIST_rep001_20261006-114530-123_a1b2c3_raw.csv
        FIST_rep001_20261006-114530-123_a1b2c3_raw.npz
        FIST_rep001_20261006-114530-123_a1b2c3_raw_meta.json
        FIST_rep001_20261006-114530-123_a1b2c3_raw_notes.json
        FIST_rep001_20261006-114530-123_a1b2c3_trimmed.csv
        FIST_rep001_20261006-114530-123_a1b2c3_trimmed.npz
        FIST_rep001_20261006-114530-123_a1b2c3_trimmed_meta.json
      OPEN/  THUMB_UP/  WAVE_LEFT/  WAVE_RIGHT/
      _graphs/                     exported PNG/SVG images
      _deleted/                    quarantined recordings
  person_02/
  person_03/
exports/
  dataset_20261006-120000-000.npz          combined windows + features
  dataset_20261006-120000-000_index.csv    one row per window
  dataset_20261006-120000-000_summary.json counts, labels, persons
```

**Person separation is structural.** It is what makes leave-one-person-out
validation possible without any bookkeeping. Never merge two subjects into one
`person_id` folder, and never reuse a `person_id` for a different person.

**Folders starting with `_` are excluded from dataset discovery.** That is how
`_deleted` keeps raw data recoverable without it re-entering the dataset.

## 2. Filename grammar

```
<GESTURE>_rep<NNN>_<YYYYMMDD-HHMMSS-mmm>_<uid8>_<stage>[.<dedupe>].<ext>

FIST       _rep003   _20261006-114530-123   _a1b2c3d4   _trimmed   .csv
  label      repetition      start timestamp      uid        stage    format
```

| Part | Meaning |
|---|---|
| `GESTURE` | the label, also the parent folder name |
| `rep<NNN>` | repetition number within the session, zero-padded |
| timestamp | local time the capture started, millisecond resolution |
| `uid8` | 8 hex chars. **Shared between a recording and its trimmed copies**, which is how a raw/trimmed pair is identified |
| `stage` | `raw`, `trimmed`, `processed`, `augmented`, `take` |
| dedupe | `_2`, `_3`, … appended by `unique_path()` when a name is taken. Never overwrites |

Stage detection tolerates the dedupe counter, so `..._trimmed_2.csv` is still
a trimmed file.

## 3. Stages

| Stage | Written by | Mutability |
|---|---|---|
| `raw` | the recorder, once | **immutable** |
| `trimmed` | TRIM dialog, AUTO-TRIM ALL, or `AUTO_TRIM_WRITE_FILE` | new file per trim |
| `processed` | reserved for filtered/normalised derivatives | — |
| `augmented` | reserved for synthetic data | — |
| `take` | START FREE TAKE, once | **immutable**, and **never loaded as data** |

Augmented data must always be generated into its own stage and never mixed
into `raw`.

### The `take` stage

A **take** is one long free recording — minutes, not seconds — saved
unlabelled so that labelled clips can be cut out of it afterwards. Its
filename has no repetition number, because a take is not the Nth instance of
anything:

```
<GESTURE>_take<NNN>_<YYYYMMDD-HHMMSS-mmm>_<uid8>_take.<ext>

FIST       _take001  _20261012-164530-123   _a1b2c3d4  _take  .npz
  intended    take nr     start timestamp      uid      stage
```

The gesture in the name is the **intended** one — what the operator sat down
to collect. It is not a label: the take contains rest, fidgeting and false
starts as well.

**A take is never training data.** Two independent mechanisms keep it out of
the dataset, and both are checked in `tests/test_free_take.py`:

1. it is written to `<session>/_takes/`, and `discover_recordings()` prunes
   every folder whose name starts with `_`;
2. it carries stage `take`, which no stage selector asks for — not `raw`, not
   `prefer_trimmed`, not the export bundle.

Clips cut out of a take are written as ordinary stage-`raw` recordings in the
gesture folder. Each gets its **own** uid, because uid is what pairs a raw
file with its trimmed copies. The link back to the take is in the metadata:

| Field | Meaning |
|---|---|
| `capture_style` | `"cut_from_take"` |
| `parent_take_uid` | the take's uid |
| `parent_take_label` | the take's intended gesture |
| `parent_take_file` | the take's path, relative to the dataset root |
| `take_start_index` / `take_end_index` | the clip's sample range inside the take |
| `take_start_s` / `take_end_s` | the same range in seconds from the take's start |

so any clip can be traced back and re-cut from the take, and the take itself
is never modified.

## 4. CSV schema

One row per sample. Lossless: integers are written as integers, timestamps keep
microsecond resolution.

| Column | Type | Meaning |
|---|---|---|
| `sample_index` | int | 0-based index within this file |
| `t_rel_s` | float | seconds since the first sample of this file |
| `t_unix` | float | absolute Unix timestamp (monotonic-derived) |
| `ch1` … `ch8` | int | **raw device counts, exactly as received** |
| `in_label_window` | 0/1 | 1 inside the cued gesture window, 0 in the pre/post roll |
| `gesture` | str | label |
| `person_id` | str | subject |
| `session_id` | str | session |
| `repetition` | int | repetition number |

The last five columns are controlled by `CSV_INCLUDE_LABEL_COLUMNS`. They are
redundant with the path, which is the point: a single CSV is self-describing
and a pile of them can be `pd.concat`-ed blindly.

```python
import pandas as pd, glob
df = pd.concat(pd.read_csv(f) for f in glob.glob("datasets/*/*/*/*_raw.csv"))
```

### The `in_label_window` column

```
|<- PRE_GESTURE_MS ->|<------ RECORD_TIME ------>|<- POST_GESTURE_MS ->|
         0                        1                        0
```

The pre/post roll exists so a fast gesture cannot lose its onset or its
release. Train on the label window (`make_windows(..., use_label_window=True)`,
the default) and keep the roll for trimming and for onset-detection
experiments.

## 5. NPZ schema

```python
with np.load(path, allow_pickle=False) as d:
    d["emg"]            # (n, 8)  int32   raw device counts
    d["t_unix"]         # (n,)    float64
    d["t_rel"]          # (n,)    float32
    d["label"]          # 0-d str
    d["person_id"]      # 0-d str
    d["session_id"]     # 0-d str
    d["repetition"]     # 0-d int
    d["channel_names"]  # (8,) str
    d["metadata_json"]  # 0-d str -- the full *_meta.json payload
```

`load_recording(path)` handles `.npz`, `.csv` and `.parquet` and gives back a
`Recording`. NPZ is the preferred read format (smaller and faster); CSV is for
humans.

## 6. `*_meta.json`

Per-recording sidecar. Key fields:

| Field | Meaning |
|---|---|
| `uid`, `stage`, `label`, `label_index` | identity |
| `person_id`, `session_id`, `repetition` | grouping |
| `n_samples`, `n_channels`, `duration_s` | shape |
| `sample_rate_measured` | **measured** from timestamps, not assumed: samples per second across the recording's span. The dongle delivers packets in bursts, so a per-interval estimate would be nonsense (it reported 9258 Hz for a 57 Hz stream before this was fixed). A long dropout lowers this value, which is honest — gaps are reported separately in `quality_issues`. |
| `start_unix`/`end_unix`, `start_iso`/`end_iso` | recording start and end |
| `label_start_index`, `label_end_index` | the cued window inside the samples |
| `amplitude_min`, `amplitude_max` | per-channel extremes |
| `quality_issues`, `quality_worst_level` | every finding from `check_quality` |
| `activity_detected` | the full `ActivityResult`, incl. thresholds used |
| `files` | `"stage:format"` → path relative to the dataset root |
| `meta` | handedness, arm, device, notes, plan position, disconnect flags |
| `config` | **complete config snapshot** (`recording_config_snapshot()`) |

The `config` block is why a dataset collected over several weeks stays
comparable: it records the timings, trim thresholds and EMG mode actually in
force when that file was written.

For a trimmed file, `meta` additionally carries `parent_uid`,
`parent_n_samples`, `parent_raw_file`, `trim_start_index`, `trim_end_index`,
`trim_removed_samples` and `trim_method` (`auto` or `manual`) — so any trim can
be traced back and redone from the raw file.

## 7. `manifest.jsonl`

Append-only, one JSON object per line, one line per written file set. Never
rewritten, so it doubles as the session's audit log.

```json
{"uid":"a1b2c3d4","stage":"raw","label":"FIST","repetition":3,
 "person_id":"person_01","session_id":"session_01","n_samples":130,
 "duration_s":2.58,"sample_rate_measured":50.3,
 "created_iso":"2026-10-06T11:45:30.123","saved_iso":"2026-10-06T11:45:33.001",
 "quality_worst_level":null,"quality_codes":[],"notes":"",
 "files":{"csv":"person_01/...","npz":"...","meta":"..."}}
```

A `"stage":"deleted"` line is appended when DELETE LAST quarantines a
recording. `DatasetWriter.count_existing()` and `next_repetition()` read this
file, which is how repetition numbers survive a restart without colliding.

## 7a. `markers.jsonl`

Append-only, one JSON object per line, one line per marker. Written while
recording by ADD NOTE / BREAK. Never rewritten: a marker is never edited or
deleted, and a correction is simply another marker.

```json
{"uid":"9f3c1a77","kind":"note","text":"electrode slipped",
 "t_unix":1760274330.42,"t_iso":"2026-10-12T14:25:30.420",
 "created_unix":1760274330.44,"created_iso":"2026-10-12T14:25:30.440",
 "person_id":"person_01","session_id":"session_01"}
```

| Field | Meaning |
|---|---|
| `kind` | `"note"` (an annotation) or `"break"` (a separator). Both are cut boundaries. |
| `text` | Free text. May be empty for a `break`. |
| `t_unix` | Where the marker sits **in the signal** — the timestamp of the newest sample when it was dropped, not the GUI clock. |
| `created_unix` | When the operator actually pressed the button. Differs from `t_unix` by the GUI frame latency. |

A damaged line is skipped on read rather than being fatal. `load_markers()`
returns them oldest first; `markers_between(markers, t0, t1)` filters by span.

Markers falling inside a recording's time span are also copied into that
recording's `*_meta.json` under `markers`, so the context travels with the
data even if the session folder is split up later.

## 7b. `*_notes.json`

Notes attached to one recording, after the fact. A sidecar next to the data
file, because `*_raw.*` is immutable (safety rule 1) and a note written hours
later must not force a rewrite of the signal it describes.

```json
{"recording_file":"FIST_rep001_20261006-114530-123_a1b2c3_raw.csv",
 "recording_uid":"a1b2c3d4",
 "updated_iso":"2026-10-12T15:02:11.900",
 "notes":[{"uid":"...","kind":"note","text":"subject said this felt weak",
           "t_unix":1760274330.42,"t_iso":"...",
           "created_unix":1760277731.9,"created_iso":"...",
           "person_id":"person_01","session_id":"session_01"}]}
```

The suffix follows the stage, so `*_raw_notes.json` and
`*_trimmed_notes.json` are separate files: notes belong to the take they
describe, and a trimmed copy does not inherit the raw take's notes. Written
via a temp file and `os.replace`, so an interrupted save cannot leave a
half-written sidecar. Nothing in this path opens the signal file for writing.

## 8. `session_config.json`

Written when a session starts: subject metadata, channel names and the same
config snapshot. If a session was interrupted and restarted, this reflects the
most recent start — the per-recording `*_meta.json` is the authoritative record
for any individual file.

## 9. Export bundle

The EXPORT DATASET button writes a ready-to-train bundle:

```python
with np.load("exports/dataset_....npz", allow_pickle=False) as d:
    d["X"]              # (n_windows, window_samples, 8) float32
    d["y"]              # (n_windows,) int16 -> index into label_names
    d["groups"]         # (n_windows,) str  -> person_id   <-- use this for CV
    d["label_names"]    # (n_labels,) str
    d["features"]       # (n_windows, n_features) float32
    d["feature_names"]  # (n_features,) str
    d["window_samples"], d["step"], d["fs"], d["channel_names"]
```

`..._index.csv` maps every window back to its source recording;
`..._summary.json` holds the counts per label and per person.

By default the export prefers the trimmed version of a recording and falls
back to raw (`stage="prefer_trimmed"`).

## 10. Amplitude ranges

| `EMG_MODE` | Encoding | Range | Notes |
|---|---|---|---|
| `filtered` | uint16 | 0 … 2047 | on-device filtered/rectified, non-negative, ~50 Hz |
| `raw` | int8 | −128 … 127 | unfiltered ADC counts, 200 Hz, zero-centred |

The filtered ceiling is **empirical**: measured on this armband, resting
baseline sits around 15–40 and a maximum-effort fist reached 1100–1444. It is
*not* clipped at 1023. Only the `raw` range is a true hardware limit, so the
saturation quality check is meaningful there and effectively inert for
`filtered`.

Both are stored as `int32` on disk so one loader handles either. No scaling,
no clipping and no smoothing is ever applied to the stored channels — the
normalised and envelope views are computed on demand and never written over
the raw data.
