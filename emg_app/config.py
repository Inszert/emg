"""
USER CONFIGURATION: every tunable, the device profile and the optional-
dependency probes.

Other modules read these as `cfg.NAME` at call time, so a value changed at
runtime (by the GUI, a test, or `emg_dataset_tool.NAME = ...`) is seen
everywhere.
"""

import importlib.util
import os
from typing import List


# ==============================================================================
# ==============================================================================
#                             USER CONFIGURATION
#
#   Everything you are likely to want to change lives in this block.
#   Nothing below this block needs to be edited for normal data collection.
# ==============================================================================
# ==============================================================================

# ------------------------------------------------------------------------------
# WHERE THE DATA GOES
# ------------------------------------------------------------------------------

# Root folder for all datasets (created automatically, relative to this file).
DATA_FOLDER = "datasets"

# Identity of the person currently being recorded.
# IMPORTANT: use a NEW person id for every new subject -- the dataset keeps
# people in separate folders so that subject-independent validation stays honest.
PERSON_ID = "Dano"

# One recording session = one armband placement. If you take the armband off
# and put it back on, start a NEW session id (electrode shift changes the data).
SESSION_ID = "session_01"

# Optional subject / setup metadata, stored with every recording.
HANDEDNESS = "right"            # "right" | "left"
ARM_WORN_ON = "right"           # which arm the armband is on
DEVICE_NAME = "Myo armband"     # free text
SUBJECT_NOTES = ""              # free text, editable in the GUI

# ------------------------------------------------------------------------------
# GESTURES / LABELS
# ------------------------------------------------------------------------------

# The label set IN USE. Add or remove gestures here -- the GUI, the folder
# structure and the automatic recording sequence all follow this list. It is
# also editable at runtime through GESTURES... (Ctrl+Shift+G), which writes
# back to this global so the metadata on disk matches what was recorded.
GESTURES = [
    "REST",
    "FIST",
    "OPEN",
    "THUMB_UP",
    "WAVE_LEFT",
    "WAVE_RIGHT",
]

# Gestures the tool KNOWS but does not record unless you switch them on, in
# GESTURES... or by moving one into the list above.
#
# They are off by default on purpose. Every extra class costs recording time
# from every future subject AND makes the problem harder -- six classes on an
# 8-pod forearm band already collide (CLAUDE.md 8L measured OPEN against
# WAVE_RIGHT at 0.978 cosine, i.e. nearly the same signal), so a seventh is
# only worth it if it is driven by a different muscle group rather than by a
# different-looking hand.
#
# The value is the cue shown during PREPARE, so it can say what to do rather
# than just naming a constant.
OPTIONAL_GESTURES = {
    # Thumb folded across the index finger. Driven by the thumb adductor and
    # flexor pollicis, which sit deeper and more medially than the finger
    # flexors FIST uses -- so it should separate from FIST rather than being
    # a weaker version of it.
    "THUMB_OVER_INDEX": "thumb folded flat across the index finger",
    # Thumb and index fingertips together, the other fingers relaxed. The
    # classic prosthetics precision grip.
    "PINCH": "thumb and index fingertips together, other fingers loose",
    # Index extended, the rest closed. Mixes extensor indicis with the
    # flexors holding the other fingers down, so it is not simply OPEN or
    # FIST at half effort.
    "POINT": "index finger extended, the other fingers closed",
}

# Cues for the built-in six, same idea. A gesture with no entry here is cued
# by its own name.
GESTURE_CUES = {
    "REST": "relax the hand completely",
    "FIST": "close the hand firmly",
    "OPEN": "spread the fingers wide",
    "THUMB_UP": "thumb up, other fingers closed",
    "WAVE_LEFT": "bend the wrist towards the thumb",
    "WAVE_RIGHT": "bend the wrist towards the little finger",
}


# Gestures added in the GUI (GESTURES... -> Add your own): name -> cue. They
# and the in-use list are saved to <DATA_FOLDER>/GESTURE_SETUP_FILE and
# restored on start (emg_app/gestures.py). Adding one never retrains a model.
CUSTOM_GESTURES = {}
GESTURE_SETUP_FILE = "gestures.json"

# Gestures switched OFF in the LIVE classifiers -- the ML model and the neural
# net alike (settings icon -> Gestures -> "Use in classifier"). A switched-off
# gesture is masked out of the answer: its probability is zeroed and the rest
# renormalised, which is what the model would say if it had never been asked
# to tell that gesture apart. NOTHING is retrained and nothing on disk or in a
# model changes, so switching it back on is instant. Only a gesture the model
# has never been trained on needs a retrain (emg_app/class_mask.py).
# Saved with the gesture set in <DATA_FOLDER>/GESTURE_SETUP_FILE.
CLASSIFIER_OFF = []

# ------------------------------------------------------------------------------
# MODEL SELECTION, TRAINING OPTIONS AND PROMPTS  (settings icon -> Models / Training)
# ------------------------------------------------------------------------------
# All of these are saved in <DATA_FOLDER>/app_settings.json (emg_app/settings_store.py)
# and restored at startup. With the defaults below the application behaves as it
# did before they existed.
#
# MODEL SELECTION. Off = the original behaviour: the ML classifier is MODEL_PATH
# and the neural net is NN_MODEL_PATH. On = the models picked in the settings
# from the model zoo (models/zoo/<id>/, emg_app/zoo.py). One model picked =
# that model; several = their SOFT VOTE (the mean of their probabilities) --
# for the ML models and, separately, for the nets. The two families are never
# blended with each other (CLAUDE.md safety rule 13).
MODEL_SELECTION_ENABLED = False
SELECTED_ML = []                    # zoo ids, e.g. ["ml-et-500", "ml-rf-500-leaf3"]
SELECTED_NN = []                    # zoo ids, e.g. ["nn-gru"]

# TRAINING OPTIONS, read by the trainers the GUI starts (RETRAIN, TRAIN NEURAL NET).
#   augmentation  names from training/nn_augment.AUG_TYPES, or [] for none
#   adaptation    train with per-person standardisation (emg_app/adaptation.py)
# One switch per FAMILY (the neural nets, the ML classifiers) because the
# ablation study measured them separately and they do not agree. The ML
# switches act on models trained by the model zoo (training/train_zoo.py); the
# classic RETRAIN sweep (train_model.py) trains plain models.
# The defaults come from the ablation study (docs/REPORT.md, "defaults"): a
# switch is only on by default if it was MEASURED to help.
TRAIN_AUGMENT_NN = []
TRAIN_AUGMENT_ML = []
TRAIN_ADAPT_NN = False
TRAIN_ADAPT_ML = True       # measured; tests/test_defaults.py checks it against results/experiments.json

# ADAPTATION AT RUN TIME. Only has an effect on a model that was trained with
# adaptation; with it off such a model uses its stored global statistics.
ADAPT_LIVE = True

# PROMPTS
#   ASK_CALIBRATION_BEFORE_PREDICT   before PREDICT / START NN: "calibrate the
#       hand first?" -- with an option to reuse the last calibration, offered
#       only if the app was restarted while the band stayed on the arm.
#   ASK_RETRAIN_ON_NEW_DATA          when new gestures or new recordings make a
#       model out of date: "retrain all models or only the selected?"
ASK_CALIBRATION_BEFORE_PREDICT = True
ASK_RETRAIN_ON_NEW_DATA = True
APP_SETTINGS_FILE = "app_settings.json"

# The application icon (window, dialogs, taskbar). Off = the toolkit's default
# icon. Read at startup, so a change applies at the next start.
APP_ICON_ENABLED = True


def known_gestures():
    """
    Every gesture the tool can record: the ones in use, then the ones added
    in the GUI, then the optional ones.

    Order matters for the GUI list -- the ones in use come first, in their
    configured order, so the list reads as "what you are recording" followed
    by "what else is available".
    """
    out = list(GESTURES)
    for name in list(CUSTOM_GESTURES) + list(OPTIONAL_GESTURES):
        if name not in out:
            out.append(name)
    return out


def gesture_cue(name: str) -> str:
    """The instruction shown while cueing `name`, or "" if there is none."""
    return (CUSTOM_GESTURES.get(name) or GESTURE_CUES.get(name)
            or OPTIONAL_GESTURES.get(name) or "")

# How many repetitions of EACH gesture the automatic session should collect.
REPETITIONS = 6

# Order in which the automatic sequence walks through the plan:
#   "blocked"      -> all 30 FIST, then all 30 OPEN, ...  (fastest for the subject)
#   "interleaved"  -> FIST, OPEN, ..., FIST, OPEN, ...    (best against drift/fatigue)
#   "random"       -> fully shuffled                      (best against order effects)
GESTURE_ORDER = "interleaved"

# Seed for the shuffling so a session can be reproduced.
RANDOM_SEED = 42

# Hard safety cap on the number of recordings in one session (0 = no cap).
MAX_RECORDINGS = 0

# ------------------------------------------------------------------------------
# AUTOMATIC RECORDING SEQUENCE (seconds)
# ------------------------------------------------------------------------------
#
#   ... REST (relax) -> "GET READY: FIST" -> 3 2 1 -> RECORD -> pause ...
#
REST_TIME = 2.0          # relax / return-to-neutral time before each repetition
PREPARE_TIME = 0.8       # "GET READY: FIST" announcement before the countdown
COUNTDOWN_TIME = 3.0     # length of the 3-2-1 countdown (3.0 = three ticks)
# onset -> settle -> RELAXED HOLD -> release (CLAUDE.md 8f R4).
# 2.5 s, set 2026-10-19: with "blocked" order the subject does every
# repetition of one gesture back to back, and 4 s holds x 5 reps is enough
# sustained contraction to drive measurable fatigue drift (CLAUDE.md 8f
# measured FIST losing 59 % of its amplitude across a 52 s hold). 2.5 s still
# yields ~7 windows per repetition at 0.3 s / 0.5 overlap, and it matches the
# 3 s median of the clips already in the dataset closely enough to compare.
# Adjustable per session in the PLAN panel.
RECORD_TIME = 2.5
PAUSE_TIME = 1.0         # short pause after the recording (save + breathe)

# Extra signal kept AROUND the labeled record window. It is captured into the
# raw file so that nothing is cut off a fast/explosive gesture and so that
# automatic trimming has context to work with. The labeled window itself stays
# marked in the metadata (label_start_index / label_end_index).
PRE_GESTURE_MS = 300     # ms of signal kept BEFORE the "RECORD" cue
POST_GESTURE_MS = 300    # ms of signal kept AFTER the record window ends

# Save each repetition automatically as soon as it finishes.
AUTO_SAVE = True

# Move on to the next repetition without asking.
# Ignored when REVIEW_EACH_RECORDING is True (the review step decides).
AUTO_ADVANCE = True

# ------------------------------------------------------------------------------
# REVIEW STEP -- you confirm every recording before the next one starts
# ------------------------------------------------------------------------------

# After each repetition, freeze the graph on what was just captured, draw the
# proposed cut over it, and wait. Nothing advances until you press
# ACCEPT & NEXT (or Space / Enter) or REDO (or R).
#   True  -> you are in control of the pace, and you see every take
#   False -> fully automatic, runs straight through the plan
REVIEW_EACH_RECORDING = True

# Draw the auto-detected gesture region as a draggable green band during the
# review. Drag either edge to adjust what gets kept before accepting.
REVIEW_SHOW_TRIM = True

# On ACCEPT, write the trimmed copy using the region shown in the review.
# The raw file is already on disk and is never touched by this.
REVIEW_SAVE_TRIM_ON_ACCEPT = True

# On REDO, the rejected take is MOVED to the session's _deleted folder (never
# erased) and the same repetition is recorded again.
REVIEW_QUARANTINE_ON_REDO = True

# How the countdown is drawn on top of the live graph:
#   "corner" -> compact badge in the top-right; the graph stays fully visible
#   "center" -> large centred digits (dramatic, but covers the traces)
#   "off"    -> nothing on the graph; only the banner above it
COUNTDOWN_OVERLAY = "corner"

# Short haptic buzz on the armband at RECORD start / stop (eyes-free cue).
VIBRATE_CUES = True

# ------------------------------------------------------------------------------
# MYO / SIGNAL ACQUISITION
# ------------------------------------------------------------------------------

# Serial port of the Myo BLED112 dongle. None = auto-detect (USB PID 2458:0001).
MYO_PORT = None                  # e.g. "COM3"

# Which EMG stream to use:
#
#   "filtered" -> the stream the existing working code uses: 8 x uint16,
#                 on-device filtered/rectified. SAFE DEFAULT.
#   "raw"      -> the Myo's true unfiltered ADC stream: 8 x int8, 200 Hz.
#                 Better for fast/explosive gestures and for CNN/LSTM models.
#
# Both modes go through the same myo_raw.MyoRaw connection logic.
EMG_MODE = "filtered"

# Nominal sample rate per mode (Hz). The real rate is measured at runtime and
# the measured value is what gets written into the recording metadata.
NOMINAL_SAMPLE_RATE = {
    "filtered": 50,
    "raw": 200,
}

# Valid amplitude range per mode, used by the quality checks and the plot.
#
# The "filtered" ceiling is EMPIRICAL, not from a datasheet: measured on this
# armband, a maximum-effort fist reached 1100, with nothing above 2047 over
# thousands of samples. It is NOT hard-clipped at 1023, which is why that
# earlier guess produced spurious "out_of_range" warnings. Widen it further if
# you ever see that warning on a signal that looks healthy.
EMG_RANGE = {
    "filtered": (0, 2047),       # on-device filtered EMG is non-negative
    "raw": (-128, 127),          # signed 8-bit ADC counts -- a true hard limit
}

# ------------------------------------------------------------------------------
# BAND-PASS FILTER
# ------------------------------------------------------------------------------
#
# Per mode: (high_pass_Hz, low_pass_Hz). None disables that half.
#
# READ THIS BEFORE CHANGING THE "filtered" ROW. The textbook EMG band-pass is
# 20-450 Hz, and NEITHER end of that is applicable to the default stream:
#
#   * The Myo already band-passes and rectifies on-device. What arrives in
#     "filtered" mode is an ENVELOPE, not a bipolar waveform -- the 20-450 Hz
#     stage has already happened in hardware, upstream of anything here.
#   * The envelope arrives at ~50 Hz, so Nyquist is ~25 Hz. A 450 Hz low-pass
#     is not merely unnecessary, it is unrepresentable; a 20 Hz high-pass sits
#     at 0.8x Nyquist, which is meaningless on an envelope.
#   * A high-pass on an envelope removes its DC, and DC here IS the activation
#     level. This project measured that twice and both times it cost accuracy:
#     median-centring dropped the Fisher ratio 25.5 -> 15.1 and accuracy
#     43.2 % -> 33.5 %, and baseline subtraction dropped Fisher 24.9 ->
#     11.5-17.2. See the "Absolute amplitude is kept" decision in CLAUDE.md
#     section 13.
#
# So "filtered" defaults to a LOW-PASS ONLY: an 8 Hz smoother on the envelope,
# which removes window-to-window jitter without touching the level. The
# high-pass is left at None on purpose, not by omission.
#
# "raw" is the opposite case: a true 200 Hz bipolar ADC stream with nothing
# applied to it, where a band-pass is both standard and genuinely absent.
# 20-95 Hz fits inside its 100 Hz Nyquist.
EMG_BANDPASS_HZ = {
    "filtered": (None, 8.0),
    "raw": (20.0, 95.0),
}

# Filter order per half. 2 is a gentle 12 dB/octave roll-off; the zero-phase
# forward-backward pass doubles the effective order to 4.
EMG_BANDPASS_ORDER = 2

# A sample at/above this fraction of the range limit counts as "saturated".
# This check is meaningful for "raw" (int8 really does clip at +-127) and is
# effectively inert for "filtered", whose on-device envelope never approaches
# the ceiling above.
SATURATION_FRACTION = 0.98

# ------------------------------------------------------------------------------
# BUFFERS / MEMORY
# ------------------------------------------------------------------------------

# How far past "now" the graph may be panned, in seconds. A sliver rather
# than zero, so the newest sample is not flush against the right edge.
PLOT_FUTURE_MARGIN_S = 0.5

# Length of the rolling in-RAM display buffer, in seconds. Bounded on purpose:
# long sessions must not grow memory without limit.
#
# This is also how far back you can still select and label a gesture in
# LABEL ON GRAPH mode, so it must comfortably exceed PLOT_WINDOW_S.
#
# 10 minutes. The cost is linear and small: 8 channels at ~50 Hz is about
# 30k samples, a couple of MB, and it buys you the ability to scroll back and
# label something you did five minutes ago instead of losing it.
RING_BUFFER_SECONDS = 600.0

# ------------------------------------------------------------------------------
# LABEL ON GRAPH (free recording)
# ------------------------------------------------------------------------------
# Instead of being cued by a countdown, you stream continuously, perform a
# gesture whenever you like, then drag a box around it on the graph and label
# it. Nothing is recorded until you select it, so hesitation, false starts and
# fidgeting never enter the dataset.
#
# This is the better workflow for data quality: you see the signal before you
# commit to it, and you label what actually happened rather than what the
# countdown told you to do.

# Shortest selection that may be labelled (seconds).
LABEL_MIN_SELECTION_S = 0.25

# How many labelled marks stay drawn on the graph. They disappear on their own
# once they scroll off the left edge; this only caps the drawing cost.
LABEL_MAX_VISIBLE_MARKS = 14

# Default length of a cut, in seconds: the width of a fresh selection box
# both on the live graph (SELECT LAST n s) and in the cut dialog that carves
# clips out of a free take. The GUI mirrors its spin box back into this value,
# so whatever you set there is what lands in every *_meta.json.
LABEL_DEFAULT_SELECTION_S = 2.0

# Colour per gesture for the labelled bands. Falls back to the channel
# palette for gestures not listed here.
LABEL_COLORS = {
    "REST": "#64748b",
    "FIST": "#ef4444",
    "OPEN": "#22c55e",
    "THUMB_UP": "#f59e0b",
    "WAVE_LEFT": "#38bdf8",
    "WAVE_RIGHT": "#a78bfa",
}

# Colours for gestures that are not in LABEL_COLORS (the ones added in the
# GUI, the optional ones, anything found on disk). Chosen to stay apart from
# the six above; handed out in known_gestures() order so a new gesture keeps
# its colour and never takes one already in use. See label_color().
EXTRA_LABEL_COLORS = ["#ec4899", "#14b8a6", "#84cc16", "#f97316", "#6366f1",
                      "#a16207", "#06b6d4", "#e11d48"]


def label_color(name: str) -> str:
    """The one colour of gesture `name`, everywhere it is drawn."""
    if name in LABEL_COLORS:
        return LABEL_COLORS[name]
    extras = [n for n in known_gestures() if n not in LABEL_COLORS]
    index = (extras.index(name) if name in extras
             else len(extras) + sum(ord(c) for c in str(name)))
    return EXTRA_LABEL_COLORS[index % len(EXTRA_LABEL_COLORS)]


# Longest single recording that may be held in RAM (safety valve, seconds).
# This caps a cued or manual take. A FREE TAKE has its own, much larger cap --
# see FREE_TAKE_MAX_MINUTES below.
MAX_RECORDING_SECONDS = 120.0

# ------------------------------------------------------------------------------
# WINDOW SIZING
# ------------------------------------------------------------------------------
# Every window is clamped to the screen it opens on, so none of them can ever
# appear larger than the display or spill under the taskbar. The sizes below
# are what the app WANTS; what it gets is the smaller of that and the screen's
# available area.
#
# `availableGeometry()` is used rather than the raw screen size, because it
# already excludes the taskbar and is reported in the same logical pixels Qt
# lays widgets out in -- which is what makes this correct under Windows
# display scaling, where a "1920x1080" screen is 1280x720 logical pixels at
# 150 %.

# Preferred size of the main window. Shrunk to fit a smaller screen.
WINDOW_WIDTH = 1680
WINDOW_HEIGHT = 1000

# Fraction of the screen's available area a window may occupy when it has to
# be shrunk. Slightly below 1.0 so the title bar and borders stay on-screen.
WINDOW_FIT_FRACTION = 0.97

# Open the main window maximised when the screen is too small for the
# preferred size anyway. Saves the first manual resize on a laptop.
WINDOW_MAXIMIZE_IF_SMALL = True

# ------------------------------------------------------------------------------
# FREE TAKE  (record the whole session now, cut it up later)
# ------------------------------------------------------------------------------
# Press START FREE TAKE, stream for as long as you like -- minutes, not
# seconds -- then STOP & SAVE. The whole stretch is written to disk in one
# piece, unlabelled, and you carve the useful parts out of it afterwards with
# OPEN TAKE: select a span, pick a gesture, EXTRACT CLIP, repeat.
#
# Why this is a separate thing from LABEL ON GRAPH: label mode can only reach
# back RING_BUFFER_SECONDS, because nothing is on disk until you select it. A
# free take is on disk the moment you stop, so "cut it up later" can mean
# tomorrow, on a machine with no armband attached.
#
# A take is NOT training data and must never be loaded as if it were: it is
# mostly rest, fidgeting and false starts. Two things keep it out of the
# dataset:
#   * it is written into the session's _takes/ folder, and discovery prunes
#     every folder whose name starts with "_" (same rule as _deleted);
#   * it carries stage "take", which no stage selector asks for.
# Clips EXTRACTED from a take are written as ordinary stage-"raw" recordings
# in the gesture folder, and those are the training data.

# Master switch for the whole feature.
FREE_TAKE_ENABLED = True

# Hard cap on one free take, in minutes. The samples are held in RAM until you
# stop, so this is a memory bound: 8 channels at ~50 Hz costs roughly 1 MB per
# 10 minutes, and the recorder stops itself when the cap is reached rather than
# growing without limit.
FREE_TAKE_MAX_MINUTES = 15.0

# Folder for takes inside the session folder. Must start with "_" so dataset
# discovery prunes it -- see the note above.
FREE_TAKE_FOLDER = "_takes"

# Warn in the status line once this fraction of the cap is used up.
FREE_TAKE_WARN_FRACTION = 0.85

# Shortest free take worth keeping (seconds). Below this, STOP & SAVE throws it
# away rather than littering _takes/ with accidental button presses.
FREE_TAKE_MIN_SECONDS = 2.0

# ------------------------------------------------------------------------------
# MARKERS AND NOTES
# ------------------------------------------------------------------------------
# A marker is a timestamped piece of text dropped onto the live stream while
# you record: "electrode slipped", "subject fatigued", "block 2 starts here".
# It is drawn on the graph as a vertical separator with its text, so a long
# free-running take stops being an undifferentiated wall of signal.
#
# Markers are never destructive. They go to their own append-only file next to
# the manifest, so dropping one touches no recording, and a marker placed by
# mistake is corrected by dropping another -- consistent with the rest of the
# project, where nothing on disk is ever rewritten.
#
# Two kinds, and the difference is only intent:
#   "note"  -- an annotation. Something worth knowing about this moment.
#   "break" -- a separator. Where one block of work ends and the next begins.
# Both are cut boundaries: CUT SINCE LAST MARKER turns the span between the
# newest marker and now into a labelled training recording.

# Master switch for the whole markers/notes feature.
MARKERS_ENABLED = True

# Append-only marker log, one JSON object per line, in the session folder.
MARKERS_NAME = "markers.jsonl"

# Whether cutting the span between markers into a training recording is
# offered. Turn it off for a pure observation session where you want the
# annotations but nothing cut from them. Selectable in the GUI at any time.
MARKER_CUT_ENABLED = True

# How many markers stay drawn on the graph at once (drawing cost only -- the
# file keeps every one of them).
MARKER_MAX_VISIBLE = 12

# How much of a marker's text is drawn on the graph before it is elided. The
# full text always goes to the file.
MARKER_LABEL_MAX_CHARS = 28

# Colour per kind.
MARKER_COLORS = {
    "note": "#38bdf8",
    "break": "#f59e0b",
}

# Per-recording notes live in a sidecar beside the data file, because
# `*_raw.*` is immutable (safety rule 1) and a note written hours later must
# not force a rewrite of the signal it describes.
NOTES_SIDECAR_SUFFIX = "_notes.json"

# ------------------------------------------------------------------------------
# REAL-TIME GRAPH
# ------------------------------------------------------------------------------

GUI_FPS = 60                     # graph refresh rate (Hz); lower on weak PCs
PLOT_WINDOW_S = 30.0             # visible time window (seconds)
PLOT_AUTOSCALE = True            # auto vertical scaling per channel
PLOT_Y_PADDING = 1.15            # headroom factor when autoscaling
PLOT_ANTIALIAS = False           # off = noticeably faster
PLOT_USE_OPENGL = False          # try True if you have a good GPU
PLOT_LINE_WIDTH = 1              # >1 is much slower in PyQtGraph
SHOW_ENVELOPE = False            # overlay a moving-RMS envelope
ENVELOPE_MS = 50.0               # window of that envelope / of the RMS readout

# Per-channel colours (8 channels).
CHANNEL_COLORS = [
    "#4cc9f0", "#4895ef", "#4361ee", "#7209b7",
    "#b5179e", "#f72585", "#ff6d00", "#ffba08",
]

# ------------------------------------------------------------------------------
# WINDOWING (for later ML training)
# ------------------------------------------------------------------------------

WINDOW_MS = 200                  # analysis window length
WINDOW_OVERLAP = 0.5             # 0.5 = 50 % overlap between windows

# ------------------------------------------------------------------------------
# AUTOMATIC TRIMMING (EMG activity detection)
# ------------------------------------------------------------------------------

# Run activity detection on every recording and store the suggestion.
AUTO_TRIM = True

# Write the trimmed file automatically, or only suggest it in the TRIM dialog.
# The RAW file is ALWAYS written first and is never touched by trimming.
AUTO_TRIM_WRITE_FILE = False

# Never auto-trim these labels (REST has no "activity" to find by definition).
AUTO_TRIM_SKIP_LABELS = ["REST"]

# Envelope used for detection: moving RMS of the baseline-removed signal.
TRIM_ENVELOPE_MS = 50.0

# Baseline (resting noise) is estimated from this much signal at the start.
TRIM_BASELINE_MS = 200.0

# Hysteresis thresholds, as a multiple of the baseline envelope level. The
# signal must exceed HIGH to start a gesture and fall below LOW to end it.
TRIM_THRESHOLD_HIGH = 3.0
TRIM_THRESHOLD_LOW = 1.8

# Absolute floor for the threshold, in signal units. Stops a dead-quiet
# baseline from making the detector hair-trigger.
TRIM_ABS_FLOOR = {
    "filtered": 12.0,
    "raw": 3.0,
}

# A burst shorter than this is noise, not a gesture.
TRIM_MIN_ACTIVE_MS = 120.0

# A quiet stretch shorter than this does not end a gesture (it bridges the dip
# between two phases of e.g. a fast wave, or the brief release in the middle of
# a double contraction). Raise it if a two-phase gesture gets split in half and
# only the louder half survives trimming; lower it to separate rapid repeats.
TRIM_MIN_INACTIVE_MS = 250.0

# Padding kept around the detected region.
PRE_PADDING_MS = 100.0
POST_PADDING_MS = 150.0

# How many channels must agree that something is happening (1..8).
TRIM_MIN_ACTIVE_CHANNELS = 2

# ------------------------------------------------------------------------------
# DATA QUALITY CHECKS
# ------------------------------------------------------------------------------
# A failed check NEVER deletes anything. It raises a visible warning and the
# recording is flagged in its metadata; keeping or discarding it is your call.

QC_ENABLED = True

# When the automatic sequence should stop and ask you what to do:
#   "never"   -> never interrupt; problems are only logged and flagged
#   "error"   -> pause on serious problems (disconnect, dead channel)
#   "warning" -> pause on anything suspicious (thorough but slow)
QC_PAUSE_ON = "error"

# Warn if the recording has fewer than this fraction of the expected samples.
QC_MIN_SAMPLE_FRACTION = 0.80

# Warn if two consecutive samples are further apart than this (dropped packets).
QC_MAX_GAP_MS = 120.0

# Warn if more than this fraction of samples sit at the saturation level.
QC_MAX_SATURATED_FRACTION = 0.02

# Warn if a channel never changes (dead electrode / lost contact).
QC_FLAT_CHANNEL_TOLERANCE = 1e-9

# Warn if the resting baseline RMS is above this (bad contact, mains hum).
QC_MAX_BASELINE_RMS = {
    "filtered": 60.0,
    "raw": 12.0,
}

# Warn if a non-REST recording never gets louder than this (gesture missing).
QC_MIN_PEAK_AMPLITUDE = {
    "filtered": 40.0,
    "raw": 8.0,
}

# ------------------------------------------------------------------------------
# FILE FORMATS
# ------------------------------------------------------------------------------

SAVE_CSV = True                  # human-inspectable, one row per sample
SAVE_NPZ = True                  # compact, ML-friendly (numpy)
SAVE_PARQUET = False             # needs pyarrow; nice for very large datasets
SAVE_META_JSON = True            # per-recording metadata sidecar

CSV_FLOAT_FORMAT = "%.6f"

# Include the identifying columns in every CSV row. Slightly redundant, but it
# makes a single CSV self-describing and lets you pd.concat them blindly.
CSV_INCLUDE_LABEL_COLUMNS = True

# ------------------------------------------------------------------------------
# RECONNECTION
# ------------------------------------------------------------------------------

# ---- staying connected -------------------------------------------------------
# The armband drops the link for three different reasons, and each has its own
# setting here. Measured on this hardware, these three together turn a link
# that dropped every couple of minutes into one that holds.

# 1. The armband puts ITSELF to sleep after a short idle period, and streaming
#    EMG does not keep it awake. Sending set_sleep_mode(never_sleep) stops
#    that. Costs battery, so charge it between sessions.
MYO_NEVER_SLEEP = True

# 2. The BLE supervision timeout -- how long the radio tolerates silence
#    before declaring the link dead. myo_raw's original 640 ms is short
#    enough that a hand over the dongle or a busy 2.4 GHz band drops it.
#    Raised here; range is 100 ms to 32000 ms.
MYO_CONN_TIMEOUT_MS = 3000

# 3. Connection interval (ms). Shorter = lower latency, more airtime.
#    7.5 ms is the Myo default and is what the working code used.
MYO_CONN_INTERVAL_MS = 7.5

# ---- latency ------------------------------------------------------------------
# Drain the serial port in bulk instead of one byte at a time.
#
# myo_raw's BT.recv_packet() calls ser.read() for a SINGLE byte and re-assigns
# ser.timeout once per packet; on Windows each assignment reconfigures the COM
# port. That only just keeps up with the armband, so any hiccup leaves a
# backlog in the OS buffer that never drains -- and every sample you see is
# that far behind reality. Measured symptom: gestures appearing ~8 s late while
# the reported sample rate still looked correct, because a constant delay does
# not change the spacing between samples.
#
# With this on, each pass reads everything waiting in one call and parses it
# with myo_raw's own byte parser, so the reader outruns the armband and the
# backlog stays at zero.
MYO_BULK_READ = True

# Discard anything already queued in the port before streaming starts, so the
# session begins at "now" rather than replaying the connect handshake.
MYO_FLUSH_ON_START = True

# If the backlog ever exceeds this many bytes (roughly 25 bytes per sample),
# the link is running behind. Outside a recording the buffer is flushed to
# restore responsiveness; during a recording it is only logged, because a gap
# in a recording is worse than latency and the quality checks will flag it.
# 0 disables the guard.
MYO_MAX_BACKLOG_BYTES = 4096

# Reconnect straight to the armband's known address instead of re-scanning.
# The scan only succeeds while the armband is advertising, which it stops
# doing shortly after coming off, and it is the slow part of recovery:
# about 1 s with this on versus 5-10 s with a scan.
MYO_FAST_RECONNECT = True

AUTO_RECONNECT = True

# Pause a running session when the link drops, and resume it automatically once
# the armband is back. Without this, repetitions would keep counting down while
# no signal is arriving and you would collect empty recordings.
PAUSE_SESSION_ON_LINK_LOSS = True

RECONNECT_DELAY_S = 2.0          # wait before the first retry

# Give up a BLE scan after this long and start over with a clear message.
# The armband stops advertising a minute or two after you take it off, and
# MyoRaw.connect() would otherwise sit in its scan loop indefinitely while the
# UI just says "Opening the Myo dongle...". 0 disables the timeout.
SCAN_TIMEOUT_S = 20.0
RECONNECT_MAX_DELAY_S = 10.0     # back-off ceiling
RECONNECT_MAX_ATTEMPTS = 0       # 0 = retry forever

# No EMG packet for this long while connected => treat the link as lost.
# Kept comfortably above the supervision timeout above: tearing the connection
# down and rebuilding it costs more than waiting out a brief gap.
STALL_TIMEOUT_S = 6.0

# ------------------------------------------------------------------------------
# LIVE CLASSIFIER
# ------------------------------------------------------------------------------
# NO fake predictions are ever displayed. Every number in the prediction panel
# comes from a real model's predict()/predict_proba(). With no model file the
# panel says so and the buttons stay disabled.
#
# Train one with:   python training/train_model.py
ENABLE_LIVE_PREDICTION = True
MODEL_PATH = "models/gesture_model.joblib"

# How the PREDICT button behaves:
#   "one_shot"   -> one prediction per press, then it waits for the next press
#   "continuous" -> keeps predicting until you press STOP
# Switchable in the GUI; this is just the startup default.
PREDICTION_MODE = "one_shot"

# Majority vote over this many consecutive windows in continuous mode.
# Higher = steadier but laggier. 1 disables smoothing.
PREDICTION_SMOOTHING = 5

# How often continuous mode runs the classifier (ms).
PREDICTION_INTERVAL_MS = 150

# One-shot timing: after the press you get this long to get into the gesture,
# then the window used for the prediction is captured.
ONE_SHOT_DELAY_S = 0.8
ONE_SHOT_HOLD_S = 3.0            # how long the result stays on screen

# ------------------------------------------------------------------------------
# NEURAL NETWORK  (a SECOND, SEPARATE opinion -- never part of the answer above)
# ------------------------------------------------------------------------------
# The panel above is the deployed classifier: one scikit-learn estimator, no
# torch, no CUDA, destined for an SBC on a robot. This section is a different
# model entirely -- a small sequence net that reads how the envelope CHANGES
# across the window instead of summarising it into statistics.
#
# IT IS DELIBERATELY NOT FUSED WITH IT. Separate file, separate START button,
# separate probability bars, separate statistics dialog. `LivePredictor` never
# opens the checkpoint, the net is not a member of the DECISION TRAIL's soft
# vote, and turning it on changes nothing about what LIVE CLASSIFIER answers.
# Two independently validated models that disagree tell you something; one
# blended number that hides which half was wrong does not.
#
# Train it with:   python training/train_nn.py            (CUDA; --device cpu to force)
ENABLE_NEURAL_NET = True
NN_MODEL_PATH = "models/gesture_nn.pt"

# THE RECORDER NEVER IMPORTS TORCH, and that is not negotiable (section 13):
# the import costs ~7 s and attaches a GPU context to the process whose one
# job is to not drop EMG samples. So inference runs in a subprocess --
# `nn_serve.py` -- which the app talks to over stdin/stdout, exactly the way
# the GPU/CPU badge is probed. `tests/test_no_torch.py` still passes with the
# net enabled, because nothing here imports it.
NN_WORKER_SCRIPT = os.path.join("training", "nn_serve.py")

# "cuda" asks the worker for the GPU and lets it FAIL rather than fall back:
# a silent CPU fallback looks identical to a working run. The panel then says
# why, in words, instead of quietly running slower.
#   "cuda" -> require CUDA,  "cpu" -> force CPU,  "auto" -> CUDA if present
NN_DEVICE = "auto"

# How long to wait for the worker to load torch and the checkpoint before
# giving up on it. The torch import alone is several seconds on a cold cache.
NN_START_TIMEOUT_S = 40.0

# How often the net classifies a window while running (ms). Its own timer,
# independent of PREDICTION_INTERVAL_MS, so the two models can run at
# different cadences -- or one of them alone.
NN_INTERVAL_MS = 150

# Soft-vote smoothing over this many consecutive windows, like the sklearn
# panel's. 1 disables it. Kept separate so a change here cannot alter what
# the deployed classifier displays.
NN_SMOOTHING = 5

# A stand-in for the model grid until a checkpoint says what it really is.
# Both trainers resample to 60 Hz (train_model.MODEL_FS), but THAT number is
# authoritative and arrives in the payload -- this is only what the engine
# reports before it has loaded one, so that the field is never empty.
NN_MODEL_FS_FALLBACK = 60.0

# Drop a request if the worker has not answered the previous one. The net is
# a diagnostic, not a recorder: skipping a window is better than queueing up
# a backlog that answers questions about the past.
NN_MAX_IN_FLIGHT = 1

# ...and give up on a request the worker never answered after this long.
# Without it, one lost reply would hold the single in-flight slot forever and
# the panel would freeze on its last answer while looking perfectly healthy.
# Generous against a measured ~25 ms round trip: this is a stuck-request
# valve, not a latency budget.
NN_RESPONSE_TIMEOUT_S = 5.0

# ------------------------------------------------------------------------------
# PER-WEARER CALIBRATION  (optional; off until you press CALIBRATE)
# ------------------------------------------------------------------------------
# The armband reads a different number on every wearer for the same effort:
# skin impedance, muscle mass, band tightness and where pod 1 ended up all
# scale the signal. Measured on this dataset (CLAUDE.md section 8k): the two
# wearers' pooled p95 differ by only 9 % OVERALL, but per channel they differ
# by up to 1.85x -- so what actually varies between people here is the
# per-channel profile, not the master volume.
#
# A calibration is four numbers per channel measured once per wearer: a REST
# level and a REFERENCE (strong-contraction) level. Together they define an
# affine map onto a common scale, so "half effort" means the same number for
# everybody.
#
# IT IS OFF BY DEFAULT, and that is the honest default. Nothing about the
# pipeline changes until a calibration exists AND a mode other than "off" is
# chosen; `CALIBRATION_MODE = "off"` reproduces every number this project has
# ever reported. Whether calibration HELPS is measured by the sweep like
# every other representation choice -- see section 8k -- not asserted here.
#
# One calibration is valid until it is replaced. RECALIBRATE in the GUI
# measures a new one; nothing expires on its own, nothing is re-estimated per
# recording, and a stored calibration survives restarts because it lives in
# the dataset next to the wearer's data.
#  False = every side panel starts folded; click a header to open it.
#  True restores the old defaults (SUBJECT, FREE TAKE, LABEL, RECORDINGS open).
SIDE_PANELS_START_OPEN = False
#  Side panels not shown at all (keys: subject, free_take, label, markers,
#  plan, channels, recordings, prediction, neural, training, log). They are
#  still built, so shortcuts and everything else keep working; remove a key
#  here to bring its panel back.
SIDE_PANELS_HIDDEN = {"label", "markers"}
CALIBRATION_ENABLED = True

# How a calibration is applied, if one exists:
#   "off"         -> not applied at all. The signal reaches the features in
#                    device counts, exactly as before.
#   "gain"        -> ONE scalar divisor for all eight channels. The most
#                    conservative mode: it fixes overall effort/gain between
#                    wearers and leaves the spatial profile -- the thing that
#                    actually distinguishes the gestures -- bit-for-bit
#                    proportional.
#   "per_channel" -> each channel gets its own divisor. Also equalises the
#                    per-channel differences that come from where the pods sat,
#                    at the cost of flattening profile differences that may be
#                    real.
#   "affine"      -> per-channel divisor AND the wearer's rest level
#                    subtracted first. The most aggressive.
#
# NOTE on the rest subtraction in "affine": CLAUDE.md section 13 records that
# baseline subtraction measurably HURT (Fisher 24.9 -> 11.5-17.2). That was a
# PER-RECORDING baseline, estimated from the recording itself, which partly
# cancels the gesture it is supposed to be a baseline for. A calibration
# baseline is one constant per wearer, measured once from a dedicated rest
# capture, and cannot cancel a gesture because it never sees one. Different
# operation, different risk -- which is why it is a mode you can measure
# rather than a default.
#
# SET TO "affine" ON 2026-10-16, because the full sweep said so. Measured on
# the 85 recordings of two wearers, subject-grouped per window: off 80.2 %,
# gain 79.7 %, affine **86.8 %** -- and the gap between the subject-grouped
# and recording-grouped protocols, which IS the subject-specificity, narrowed
# from 17.6 to 11.4 points. The full table and the caveats are in CLAUDE.md
# section 8k. The price of this default is real and is enforced rather than
# hoped for: every wearer must calibrate before the saved model will predict
# for them, and LivePredictor refuses rather than answering on the wrong
# scale. Set this back to "off" and retrain to undo it exactly.
CALIBRATION_MODE = "affine"

# Which calibration the TRAINERS apply to recorded data: "derived" (from each
# wearer's own recordings, labels unused -- one method for everybody) or
# "saved" (calibration.json, which may be a live REST+SQUEEZE). Predicting
# always uses the saved one. See training_calibrations().
TRAINING_CALIBRATION_SOURCE = "derived"

# File names. The record lives in the WEARER's folder, not next to the model,
# because it belongs to the person: that is what lets the trainer calibrate
# each training subject with their own numbers and the GUI calibrate whoever
# is wearing the band right now.
#       <root>/<person_id>/calibration.json          the current one
#       <root>/<person_id>/calibration_log.jsonl     append-only history
CALIBRATION_NAME = "calibration.json"
CALIBRATION_LOG_NAME = "calibration_log.jsonl"

# How long each of the two captures runs, in seconds.
CALIBRATION_REST_SECONDS = 3.0
CALIBRATION_EFFORT_SECONDS = 3.0

# Fewer samples than this in either capture and the calibration is REFUSED.
# At ~50 Hz this is about half a second.
CALIBRATION_MIN_SAMPLES = 24

# Which percentile of each capture becomes the stored level. The rest level
# is a median (robust to a twitch); the reference is a high percentile rather
# than the maximum, because one spike should not define a wearer's full scale.
CALIBRATION_REST_PERCENTILE = 50.0
CALIBRATION_EFFORT_PERCENTILE = 95.0

# What a calibrated reference contraction maps to. The value is arbitrary but
# it is NOT free: the level features include absolute terms (`log_mean`,
# `total_log`) and the quality checks have absolute thresholds, so calibrated
# signal is scaled back into roughly the same numeric range the Myo delivers
# instead of into 0..1. 600 is about a strong contraction in device counts on
# this hardware.
CALIBRATION_TARGET_LEVEL = 600.0

# A calibration is refused unless the effort capture is at least this many
# times the rest capture, and the reference is at least this many counts.
# Both guard the same mistake -- a "maximum effort" capture where the wearer
# did nothing -- which would otherwise divide the whole dataset by noise.
CALIBRATION_MIN_CONTRAST = 2.0
CALIBRATION_MIN_REFERENCE = 40.0
# Percentiles used when a calibration is DERIVED from recordings that already
# exist instead of measured live (calibration_from_recordings). All of a
# wearer's samples are pooled with their labels ignored, so the low percentile
# has to stand in for "relaxed" and the high one for "strong effort". 10/95
# rather than 50/95: pooled over a whole sitting the median is not rest, it is
# whatever the wearer spent most of the time doing.
CALIBRATION_DERIVE_REST_PERCENTILE = 10.0
CALIBRATION_DERIVE_EFFORT_PERCENTILE = 95.0
# ---- the guided gesture sweep ------------------------------------------------
# The better ritual, and the default one: instead of one generic clench, cue
# REST and then each gesture in turn, and take each channel's reference as the
# loudest that channel ever got across the set. One gesture only drives the
# muscles that gesture uses, so a single-clench reference sets the quiet
# channels' scale from nearly nothing -- see calibration_from_gesture_captures.
CALIBRATION_GESTURE_SECONDS = 2.5

# Which gestures the sweep cues. None = every entry in GESTURES except REST,
# which is always captured first and separately because it sets the floor.
# A shorter list is a valid trade: three well-spread gestures cover the eight
# channels better than six sloppy ones.
CALIBRATION_GESTURES = None

# Show the guided gesture sweep in the CALIBRATE dialog. It was hidden earlier
# on 2026-10-22 (calibration = REST + one SQUEEZE) and switched back on the same
# day at the operator's request: calibrate from every gesture in GESTURES, not
# just rest and a squeeze. One squeeze under-scales the channels it leaves
# quiet (CLAUDE.md 8k). The quick path stays below the sweep.
CALIBRATION_GUIDED_SWEEP = True

# A short "GET READY: FIST" beat before each capture, so the countdown does
# not start while the hand is still moving.
CALIBRATION_PREPARE_SECONDS = 1.2

# One capture is only accepted as evidence when its peak channel reaches this
# multiple of the wearer's rest level. Below it the capture is kept on screen,
# marked, and excluded from the reference -- "that one did not register, do it
# again" rather than silently storing a shrug.
CALIBRATION_MIN_GESTURE_CONTRAST = 2.0

# ...and its peak must reach this many counts in absolute terms. The ratio
# alone is not enough: noise on a channel sitting near rest inflates a high
# percentile, so a shrug at 32 counts against a rest of 21 can read as 2.5x
# and sail through. Set from the measured spread on this hardware -- REST p95
# is 24-27 counts and the WEAKEST real gesture any wearer produced is 183
# (Betka's THUMB_UP) -- so 60 has clear headroom on both sides. It is
# deliberately higher than CALIBRATION_MIN_REFERENCE (40), which guards a
# whole record: a calibration capture is a deliberate strong contraction, not
# the relaxed hold the recording protocol asks for.
CALIBRATION_MIN_GESTURE_LEVEL = 60.0

# The REST capture is judged too, because it is the floor everything else is
# measured against: `affine` subtracts it from every sample, and every
# gesture's contrast is a ratio over it. A "rest" held with the hand still
# tense would raise that floor and quietly fail the gestures after it. Measured
# relaxed levels on this hardware are 19-26 counts per channel (median), so 50
# is twice that while staying under CALIBRATION_MIN_GESTURE_LEVEL -- a rest
# must never be loud enough to pass as a gesture.
CALIBRATION_MAX_REST_LEVEL = 50.0

# No sample for this long and the dialog treats the stream as down. It is
# checked against the newest SAMPLE's age rather than the link state, because
# what a capture needs is samples, and a connected armband that has gone to
# sleep supplies none.
CALIBRATION_STALE_SECONDS = 2.0



# ------------------------------------------------------------------------------
# MODEL SPECIFICATION  (which classifiers compete, and how they vote)
# ------------------------------------------------------------------------------
# The sweep in train_model.py trains every ENABLED entry below, scores them all
# with the same honest protocol, and keeps the winner. MODEL SPEC... in the GUI
# edits this list -- which models take part, their hyperparameters, and whether
# they also vote as an ensemble -- and writes it to models/model_spec.json,
# which the trainer reads via --spec.
#
# Why a file and not just these globals: training runs as a subprocess (see
# CLAUDE.md section 13), so the GUI cannot hand it live Python objects. The
# spec file is the contract, and it is written next to the model so you can
# always see what produced it.
#
# Nothing here invents numbers: a spec selects and parameterises real
# scikit-learn estimators, and every score still comes from the same
# cross-validation. Switching a model on does not make it good -- the sweep
# decides that.

# The trainer's exit code for "the dataset cannot support a model yet".
# Distinct from 1 (a real failure) so the GUI can give advice instead of
# reporting a crash.
TRAIN_EXIT_NEED_MORE_DATA = 2

# Name of the spec file, written next to the model.
MODEL_SPEC_NAME = "model_spec.json"

# Name of the decision panel, written next to the model by train_model.py.
# It holds every swept candidate fitted on the whole dataset, plus their
# out-of-fold decision trail, and drives MODEL STATS -> DECISION TRAIL.
#
# It is deliberately a SEPARATE file from the model: gesture_model.joblib
# stays one estimator because that is what ships to the SBC on the robot,
# and the live prediction path never touches this file. Delete it and the
# only thing that changes is that the DECISION TRAIL view says so.
PANEL_FILENAME = "gesture_panel.joblib"

# Load the panel's members and run them beside the deployed model while
# predicting. Costs one predict_proba per candidate per window -- fine on a
# desktop, which is where this view is used. Set False on a small board.
ENABLE_DECISION_PANEL = True

# Also build a voting ensemble out of the enabled models.
MODEL_VOTING_ENABLED = True

# "soft" averages predicted probabilities (better when the members are
# calibrated, and the usual choice); "hard" counts votes.
MODEL_VOTING_MODE = "soft"

# True  -> the ensemble is always what gets saved, even if a single model
#          scored higher.
# False -> the ensemble competes in the sweep like any other candidate and the
#          best score wins. This is the honest default: forcing a vote that
#          scored worse than its own best member is a choice, not an
#          improvement.
MODEL_VOTING_ALWAYS = False

# What a hyperparameter may be. The GUI builds its editor from this, and
# validate_model_spec() coerces and clamps against it, so a hand-edited spec
# file cannot push a nonsense value into the trainer.
#   int    -> min, max, step
#   float  -> min, max, decimals, step
#   choice -> values
#   ints   -> a tuple of ints, written "256, 128" (MLP layer sizes)
MODEL_KINDS = {
    "lda": {
        "label": "LDA  (linear discriminant)",
        "note": "The myoelectric-control baseline. No real hyperparameters to "
                "overfit, trains in milliseconds.",
        "scaler": "standard",
        "params": {
            "solver": {"type": "choice", "values": ["lsqr", "svd", "eigen"],
                       "default": "lsqr"},
            "shrinkage": {"type": "choice", "values": ["auto", "none"],
                          "default": "auto",
                          "note": "ignored by the svd solver"},
        },
    },
    "svm_rbf": {
        "label": "SVM  (RBF kernel)",
        "note": "Strong at this dataset size. C is INVERSE regularisation: "
                "lower C = stronger L2 penalty = smoother boundary.",
        "scaler": "standard",
        "params": {
            "C": {"type": "float", "default": 10.0, "min": 0.01, "max": 1000.0,
                  "decimals": 2, "step": 1.0,
                  "note": "lower = more regularisation"},
            "gamma": {"type": "choice", "values": ["scale", "auto"],
                      "default": "scale"},
            "kernel": {"type": "choice", "values": ["rbf", "poly", "sigmoid"],
                       "default": "rbf"},
            "class_weight": {"type": "choice", "values": ["none", "balanced"],
                             "default": "none",
                             "note": "balanced = reweight rare gestures"},
        },
    },
    "svm_linear": {
        "label": "SVM  (linear kernel)",
        "note": "A linear decision boundary -- fewer ways to overfit than RBF.",
        "scaler": "standard",
        "params": {
            "C": {"type": "float", "default": 1.0, "min": 0.01, "max": 1000.0,
                  "decimals": 2, "step": 0.5,
                  "note": "lower = more regularisation"},
            "class_weight": {"type": "choice", "values": ["none", "balanced"],
                             "default": "none"},
        },
    },
    "random_forest": {
        "label": "Random Forest",
        "note": "Needs no scaling. More trees never hurts accuracy, only time.",
        "scaler": "none",
        "params": {
            "n_estimators": {"type": "int", "default": 400, "min": 10,
                             "max": 2000, "step": 50},
            "max_depth": {"type": "int", "default": 0, "min": 0, "max": 100,
                          "step": 1, "note": "0 = unlimited"},
            "min_samples_leaf": {"type": "int", "default": 1, "min": 1,
                                 "max": 50, "step": 1,
                                 "note": "higher = more regularisation"},
            "max_features": {"type": "choice",
                             "values": ["sqrt", "log2", "none"],
                             "default": "sqrt"},
            "class_weight": {"type": "choice",
                             "values": ["none", "balanced",
                                        "balanced_subsample"],
                             "default": "none",
                             "note": "reweight rare gestures"},
        },
    },
    "extra_trees": {
        "label": "Extra Trees",
        "note": "Like a random forest but with random split points. Often the "
                "best of the tree models on EMG features.",
        "scaler": "none",
        "params": {
            "n_estimators": {"type": "int", "default": 400, "min": 10,
                             "max": 2000, "step": 50},
            "max_depth": {"type": "int", "default": 0, "min": 0, "max": 100,
                          "step": 1, "note": "0 = unlimited"},
            "min_samples_leaf": {"type": "int", "default": 1, "min": 1,
                                 "max": 50, "step": 1,
                                 "note": "higher = more regularisation"},
            "class_weight": {"type": "choice",
                             "values": ["none", "balanced",
                                        "balanced_subsample"],
                             "default": "none"},
        },
    },
    "hist_gradient_boost": {
        "label": "Histogram Gradient Boosting",
        "note": "Boosted trees with built-in early stopping: it holds back a "
                "validation slice and stops when it stops improving.",
        "scaler": "none",
        "params": {
            "learning_rate": {"type": "float", "default": 0.1, "min": 0.001,
                              "max": 1.0, "decimals": 3, "step": 0.01},
            "max_iter": {"type": "int", "default": 200, "min": 10, "max": 2000,
                         "step": 10, "note": "boosting rounds (the cap)"},
            "max_leaf_nodes": {"type": "int", "default": 31, "min": 2,
                               "max": 255, "step": 1,
                               "note": "lower = more regularisation"},
            "l2_regularization": {"type": "float", "default": 0.0, "min": 0.0,
                                  "max": 100.0, "decimals": 3, "step": 0.1,
                                  "note": "L2 penalty on the leaf values"},
            "early_stopping": {"type": "choice",
                               "values": ["auto", "yes", "no"],
                               "default": "auto",
                               "note": "auto = on above 10k samples"},
            "validation_fraction": {"type": "float", "default": 0.1,
                                    "min": 0.02, "max": 0.5, "decimals": 2,
                                    "step": 0.05,
                                    "note": "held back for early stopping"},
            "n_iter_no_change": {"type": "int", "default": 10, "min": 1,
                                 "max": 100, "step": 1,
                                 "note": "patience before stopping"},
            "class_weight": {"type": "choice", "values": ["none", "balanced"],
                             "default": "none"},
        },
    },
    "logreg": {
        "label": "Logistic Regression",
        "note": "Linear, well-calibrated probabilities -- a good soft-voting "
                "member even when its own accuracy is middling. This is where "
                "the L1/L2/elastic-net choice lives.",
        "scaler": "standard",
        "params": {
            "C": {"type": "float", "default": 1.0, "min": 0.001, "max": 1000.0,
                  "decimals": 3, "step": 0.1,
                  "note": "INVERSE strength: lower = more regularisation"},
            "penalty": {"type": "choice",
                        "values": ["l2", "l1", "elasticnet", "none"],
                        "default": "l2",
                        "note": "L1 drives weights to zero (feature "
                                "selection); L2 shrinks them"},
            "l1_ratio": {"type": "float", "default": 0.5, "min": 0.0,
                         "max": 1.0, "decimals": 2, "step": 0.1,
                         "note": "elasticnet only: 0 = pure L2, 1 = pure L1"},
            "solver": {"type": "choice",
                       "values": ["auto", "lbfgs", "saga", "liblinear"],
                       "default": "auto",
                       "note": "auto picks one that supports the penalty"},
            "max_iter": {"type": "int", "default": 2000, "min": 100,
                         "max": 20000, "step": 100},
            "class_weight": {"type": "choice", "values": ["none", "balanced"],
                             "default": "none"},
        },
    },
    "knn": {
        "label": "k-Nearest Neighbours",
        "note": "No training to speak of. Degrades fast when a new subject's "
                "signal is offset from the training set.",
        "scaler": "standard",
        "params": {
            "n_neighbors": {"type": "int", "default": 5, "min": 1, "max": 100,
                            "step": 1},
            "weights": {"type": "choice", "values": ["distance", "uniform"],
                        "default": "distance"},
        },
    },
    "mlp": {
        "label": "MLP  (small neural net)",
        "note": "Stands in for 'neural network'. This is where early "
                "stopping, L2 (alpha) and the batch size live. The CNN/LSTM "
                "architectures the literature reports need far more data "
                "than this.",
        "scaler": "standard",
        "params": {
            "hidden_layer_sizes": {"type": "ints", "default": [128],
                                   "max_items": 4, "min": 4, "max": 1024,
                                   "note": "one number per layer"},
            "alpha": {"type": "float", "default": 0.0001, "min": 0.0,
                      "max": 1.0, "decimals": 5, "step": 0.0001,
                      "note": "L2 penalty: higher = more regularisation"},
            "solver": {"type": "choice", "values": ["adam", "sgd", "lbfgs"],
                       "default": "adam"},
            "batch_size": {"type": "int", "default": 0, "min": 0, "max": 4096,
                           "step": 16, "note": "0 = auto (min(200, n))"},
            "learning_rate_init": {"type": "float", "default": 0.001,
                                   "min": 0.00001, "max": 1.0, "decimals": 5,
                                   "step": 0.0005,
                                   "note": "adam/sgd only"},
            "max_iter": {"type": "int", "default": 1200, "min": 100,
                         "max": 20000, "step": 100,
                         "note": "epoch cap"},
            "early_stopping": {"type": "choice", "values": ["yes", "no"],
                               "default": "yes",
                               "note": "hold back a slice and stop when it "
                                       "stops improving"},
            "validation_fraction": {"type": "float", "default": 0.1,
                                    "min": 0.02, "max": 0.5, "decimals": 2,
                                    "step": 0.05,
                                    "note": "early stopping only"},
            "n_iter_no_change": {"type": "int", "default": 10, "min": 1,
                                 "max": 200, "step": 1,
                                 "note": "patience before stopping"},
        },
    },
}

# What AUTO mode tries for each kind. Hand-picked rather than a dense grid:
# these are the values that actually move the needle on EMG features, and a
# wide grid on 500 windows finds noise, not structure.
MODEL_SEARCH_SPACES = {
    "lda": {"solver": ["lsqr", "svd"], "shrinkage": ["auto", "none"]},
    "svm_rbf": {"C": [1.0, 10.0, 100.0], "gamma": ["scale", "auto"],
                "class_weight": ["none", "balanced"]},
    "svm_linear": {"C": [0.1, 1.0, 10.0, 100.0],
                   "class_weight": ["none", "balanced"]},
    "random_forest": {"n_estimators": [400, 800],
                      "min_samples_leaf": [1, 2, 4],
                      "max_features": ["sqrt", "log2"],
                      "class_weight": ["none", "balanced"]},
    "extra_trees": {"n_estimators": [400, 800],
                    "min_samples_leaf": [1, 2, 4],
                    "class_weight": ["none", "balanced"]},
    "hist_gradient_boost": {"learning_rate": [0.03, 0.1, 0.3],
                            "max_leaf_nodes": [15, 31, 63],
                            "l2_regularization": [0.0, 1.0]},
    "logreg": {"C": [0.01, 0.1, 1.0, 10.0],
               "penalty": ["l2", "l1"],
               "class_weight": ["none", "balanced"]},
    "knn": {"n_neighbors": [3, 5, 9, 15], "weights": ["distance", "uniform"]},
    "mlp": {"hidden_layer_sizes": [[64], [128], [256, 128]],
            "alpha": [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.01]},
}

# AUTO mode defaults.
MODEL_AUTO_ENABLED = False
# Cap on the combinations generated PER MODEL. The full grid is sampled down
# to this many (seeded, so a rerun is reproducible). Keep it modest: every
# combination is a full cross-validation.
MODEL_AUTO_MAX_PER_MODEL = 8

# The scalers a model may be wrapped in. "none" is right for tree models,
# which are invariant to monotone rescaling.
MODEL_SCALERS = ("standard", "robust", "minmax", "none")

# The default roster. Reproduces the sweep train_model.py has always run, so
# the first launch after this change behaves exactly as before.
DEFAULT_MODEL_SPEC_MODELS = [
    ("LDA-shrinkage", "lda", True, "standard",
     {"solver": "lsqr", "shrinkage": "auto"}),
    ("LDA-svd", "lda", True, "standard",
     {"solver": "svd", "shrinkage": "none"}),
    ("SVM-rbf-C10", "svm_rbf", True, "standard", {"C": 10.0}),
    ("RandomForest-400", "random_forest", True, "none", {"n_estimators": 400}),
    ("ExtraTrees-400", "extra_trees", True, "none", {"n_estimators": 400}),
    ("HistGradBoost", "hist_gradient_boost", True, "none", {}),
    ("LogReg", "logreg", True, "standard", {"C": 1.0, "max_iter": 2000}),
    ("SVM-rbf-C1", "svm_rbf", False, "standard", {"C": 1.0}),
    ("SVM-rbf-C100", "svm_rbf", False, "standard", {"C": 100.0}),
    ("SVM-rbf-robustscale", "svm_rbf", False, "robust", {"C": 10.0}),
    ("kNN-5", "knn", False, "standard", {"n_neighbors": 5}),
    ("kNN-15", "knn", False, "standard", {"n_neighbors": 15}),
    ("MLP-128", "mlp", False, "standard", {"hidden_layer_sizes": [128]}),
    ("MLP-256-128", "mlp", False, "standard",
     {"hidden_layer_sizes": [256, 128]}),
    ("RandomForest-800-deep", "random_forest", False, "none",
     {"n_estimators": 800, "min_samples_leaf": 1}),
]

# Below this confidence the result is shown as uncertain rather than as a
# confident label. 0 disables the distinction.
PREDICTION_MIN_CONFIDENCE = 0.45

# Draw the prediction over the live graph as well as in the side panel.
PREDICTION_OVERLAY = True

# Printed before numpy/Qt/pyqtgraph are pulled in. On a cold file cache those
# imports take a few seconds with no window and no output, which reads as a
# hang; this is the first thing on screen. Only when run as a script, so an
# `import emg_dataset_tool` from a trainer or a notebook stays silent.

APP_NAME = "Myo EMG Dataset Tool"
APP_VERSION = "1.0.0"
# The project root: one level above this package.
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ------------------------------------------------------------------------------
# DEVICE PROFILE  --  how many channels, and what they are called
# ------------------------------------------------------------------------------
# `N_CHANNELS` used to be a bare 8 sitting in the middle of the module, which
# made a second armband a search-and-replace across 44 sites. It is now
# DERIVED from the active profile, so a different band is a new entry here
# plus `EMG_DEVICE`.
#
# What a profile does NOT do is make the app talk to that band: acquisition
# still goes through `myo_raw.py`, which is Myo-specific and must stay
# working (CLAUDE.md section 3). A profile makes the DATA side -- the stored
# schema, the loaders, the features, the calibration, the quality checks and
# the plot -- stop assuming eight. That is the half that would otherwise make
# the existing dataset unreadable the moment a second band arrived.
#
# To add a band: add a profile, set EMG_DEVICE, and write its acquisition
# backend. Everything below this block follows the profile.
DEVICE_PROFILES = {
    "myo": {
        "label": "Thalmic Myo armband",
        "channels": 8,
        # Names are per profile because a band with a different geometry
        # should not be forced to call its pods CH1..CHn if it names them
        # something meaningful. None = CH1..CHn.
        "channel_names": None,
        "ble_name": "Myo_TUKE",
    },
    # Example of what a second band looks like. Unused until EMG_DEVICE
    # names it; kept here so the shape of a profile is obvious rather than
    # having to be inferred from the one entry.
    "generic_16": {
        "label": "generic 16-channel EMG band (placeholder)",
        "channels": 16,
        "channel_names": None,
        "ble_name": None,
    },
}

EMG_DEVICE = "myo"


def device_profile(name: str = None) -> dict:
    """The active profile, or a named one. Unknown names fall back to myo."""
    return DEVICE_PROFILES.get(name or EMG_DEVICE, DEVICE_PROFILES["myo"])


def set_device(name: str) -> dict:
    """
    Switch the active device profile and recompute what derives from it.

    The derived globals (`N_CHANNELS`, `CHANNEL_NAMES`) are recomputed rather
    than recalculated at every use, because they are read on hot paths --
    per-window feature naming, per-frame plotting. Anything that can instead
    follow the DATA's own width does that already, so a recording made on one
    band still loads correctly after switching to another.
    """
    global EMG_DEVICE, N_CHANNELS, CHANNEL_NAMES
    if name not in DEVICE_PROFILES:
        raise ValueError(f"unknown device {name!r}; known: "
                         f"{', '.join(sorted(DEVICE_PROFILES))}")
    EMG_DEVICE = name
    profile = DEVICE_PROFILES[name]
    N_CHANNELS = int(profile["channels"])
    names = profile.get("channel_names")
    CHANNEL_NAMES = (list(names) if names
                     else [f"CH{i + 1}" for i in range(N_CHANNELS)])
    return profile


def channel_names_for(n: int) -> List[str]:
    """
    Names for an array that is `n` channels wide, whatever the config says.

    Used by anything that has real data in hand: a file recorded on an
    8-channel band must keep reading as eight channels after the config moves
    to a sixteen-channel one. Falls back to CH1..CHn beyond the profile's own
    names rather than raising -- a name is a caption, and refusing to load a
    recording over a caption would be the wrong trade.
    """
    n = int(n)
    if n == len(CHANNEL_NAMES):
        return list(CHANNEL_NAMES)
    return [f"CH{i + 1}" for i in range(n)]


N_CHANNELS = int(device_profile()["channels"])
CHANNEL_NAMES = [f"CH{i + 1}" for i in range(N_CHANNELS)]

# Optional dependencies -- the data layer works without Qt so that the dataset
# functions can be imported from a training script or a notebook.
#
# These are probed with find_spec rather than imported. Importing pandas costs
# ~3.6 s and scipy.signal ~7.3 s on a cold cache, and both were being paid on
# every single start before the window appeared -- which made the app look
# hung and invited a Ctrl+C mid-build. find_spec only locates the module, it
# does not execute it, so the HAVE_* flags stay honest for a few milliseconds.
# pandas is loaded on first real use by _pandas(); nothing in this file uses
# scipy at all, so HAVE_SCIPY is advertised capability only.
import importlib
import importlib.util


def _module_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):                    # pragma: no cover
        return False


HAVE_PANDAS = _module_available("pandas")
HAVE_SCIPY = _module_available("scipy")

_pd = None


def _pandas():
    """Import pandas on first use. Raises RuntimeError if it is missing."""
    global _pd
    if _pd is None:
        if not HAVE_PANDAS:
            raise RuntimeError("pandas is required for this operation")
        _pd = importlib.import_module("pandas")
    return _pd
