"""
================================================================================
 MYO EMG DATASET TOOL
================================================================================

 Professional EMG recording / visualisation / labeling / dataset-generation tool
 for the Thalmic Myo armband (BLED112 dongle, COM port, myo_raw.py wrapper).

 Design goals (in priority order):

     1. Reliable Myo connection   -> reuses the proven myo_raw.MyoRaw layer
     2. No data loss              -> samples are captured in the acquisition
                                     thread itself, never in the GUI thread
     3. RAW data preservation     -> *_raw.csv is written once and never
                                     modified; trimming/processing always
                                     produces NEW files
     4. Fast real-time graph      -> PyQtGraph, 8 stacked channel bands,
                                     peak-preserving downsampling
     5. Automatic recording       -> REST / 3 / 2 / 1 / RECORD state machine
     6. Easy labeling             -> gesture list lives in the CONFIG block
     7. Trimming                  -> manual region select + automatic activity
                                     detection, saved separately
     8. Multi-person datasets     -> datasets/<person>/<session>/<GESTURE>/...
     9. ML-ready data             -> load_dataset() / make_windows() /
                                     extract_features() / leave-one-person-out
    10. Professional UI           -> dark Qt desktop application

 Run:        python emg_dataset_tool.py
 Install:    python -m pip install -r requirements.txt

 See CLAUDE.md and docs/ for architecture, dataset format and ML pipeline.
================================================================================
"""

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

# The label set. Add or remove gestures here -- the GUI, the folder structure
# and the automatic recording sequence all follow this list.
GESTURES = [
    "REST",
    "FIST",
    "OPEN",
    "THUMB_UP",
    "WAVE_LEFT",
    "WAVE_RIGHT",
]

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
RECORD_TIME = 4.0        # onset -> settle -> RELAXED HOLD -> release (CLAUDE.md 8f R4)
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
# Train one with:   python train_model.py
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

# ==============================================================================
#                          END OF USER CONFIGURATION
# ==============================================================================


import csv
import hashlib
import json
import math
import os
import random
import re
import signal
import struct
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import deque
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# Printed before numpy/Qt/pyqtgraph are pulled in. On a cold file cache those
# imports take a few seconds with no window and no output, which reads as a
# hang; this is the first thing on screen. Only when run as a script, so an
# `import emg_dataset_tool` from a trainer or a notebook stays silent.
if __name__ == "__main__":
    print("Myo EMG Dataset Tool -- loading libraries ...", flush=True)

import numpy as np

APP_NAME = "Myo EMG Dataset Tool"
APP_VERSION = "1.0.0"
N_CHANNELS = 8
CHANNEL_NAMES = [f"CH{i + 1}" for i in range(N_CHANNELS)]
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))

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


# ==============================================================================
# SECTION 1 -- SMALL HELPERS
# ==============================================================================


def nominal_rate(mode: str = None) -> float:
    """Configured nominal sample rate for an EMG mode."""
    mode = mode or EMG_MODE
    return float(NOMINAL_SAMPLE_RATE.get(mode, 200))


def emg_range(mode: str = None) -> Tuple[float, float]:
    mode = mode or EMG_MODE
    return EMG_RANGE.get(mode, (-128, 127))


def mode_value(table: dict, mode: str = None, default=None):
    """Look a per-mode configuration value up with a sane fallback."""
    mode = mode or EMG_MODE
    if isinstance(table, dict):
        return table.get(mode, default if default is not None else next(iter(table.values())))
    return table


def ms_to_samples(ms: float, fs: float) -> int:
    """Convert milliseconds to a whole number of samples (at least 1)."""
    return max(1, int(round(ms * 1e-3 * fs)))


def safe_slug(text: str, fallback: str = "unnamed") -> str:
    """Make a string safe for use as a file/folder name."""
    text = (text or "").strip()
    text = re.sub(r"[^\w.\-]+", "_", text)
    text = text.strip("._-")
    return text or fallback


def timestamp_slug(t: float = None) -> str:
    """Compact sortable timestamp: 20261006-114530-123"""
    dt = datetime.fromtimestamp(t if t is not None else time.time())
    return dt.strftime("%Y%m%d-%H%M%S-") + f"{dt.microsecond // 1000:03d}"


def iso_time(t: float) -> str:
    return datetime.fromtimestamp(t).isoformat(timespec="milliseconds")


def unique_path(path: str) -> str:
    """Never overwrite: append _2, _3, ... until the path is free."""
    if not os.path.exists(path):
        return path
    stem, ext = os.path.splitext(path)
    n = 2
    while os.path.exists(f"{stem}_{n}{ext}"):
        n += 1
    return f"{stem}_{n}{ext}"


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def fmt_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    m, s = divmod(int(seconds), 60)
    return f"{m:d}:{s:02d}"


# ==============================================================================
# SECTION 2 -- SIGNAL PROCESSING
#
# Nothing in this section ever modifies its input array. Every function returns
# a new array, so the raw signal that came off the armband stays byte-identical
# from acquisition all the way to disk.
# ==============================================================================


def moving_rms(x: np.ndarray, win: int) -> np.ndarray:
    """
    Moving RMS along axis 0, same length as the input.

    Implemented with a cumulative sum so it stays O(n) even for long
    recordings, and centred so that peaks do not shift in time.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
        squeeze = True
    else:
        squeeze = False

    n = x.shape[0]
    win = int(max(1, min(win, n)))

    sq = x * x
    # Reflect-pad so the first/last samples are not biased towards zero.
    pad_lo = win // 2
    pad_hi = win - 1 - pad_lo
    padded = np.pad(sq, ((pad_lo, pad_hi), (0, 0)), mode="reflect")
    csum = np.cumsum(padded, axis=0)
    csum = np.vstack([np.zeros((1, x.shape[1])), csum])
    out = np.sqrt((csum[win:win + n] - csum[0:n]) / float(win))

    return out[:, 0] if squeeze else out


def moving_mean(x: np.ndarray, win: int) -> np.ndarray:
    """Centred moving average along axis 0, same length as the input."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return moving_mean(x[:, None], win)[:, 0]

    n = x.shape[0]
    win = int(max(1, min(win, n)))
    pad_lo = win // 2
    pad_hi = win - 1 - pad_lo
    padded = np.pad(x, ((pad_lo, pad_hi), (0, 0)), mode="reflect")
    csum = np.vstack([np.zeros((1, x.shape[1])), np.cumsum(padded, axis=0)])
    return (csum[win:win + n] - csum[0:n]) / float(win)


def _butter2_coeffs(fc: float, fs: float, kind: str):
    """
    One 2-pole Butterworth section as `(b, a)`, via the bilinear transform.

    Written out rather than taken from `scipy.signal`. The HAVE_SCIPY comment
    above records that `scipy.signal` costs ~7.3 s to import on a cold cache
    and that nothing here actually imports it; a filter that runs in the live
    prediction path is not the place to start, and a 2-pole Butterworth is
    four lines of algebra.

    `fc` is pre-warped with tan() so the digital cutoff lands exactly where it
    was asked for instead of where the s-plane prototype would have put it.
    """
    nyquist = fs / 2.0
    fc = float(min(max(fc, 1e-6), nyquist * 0.999))
    k = math.tan(math.pi * fc / fs)          # pre-warped cutoff
    k2 = k * k
    root2 = math.sqrt(2.0)
    norm = 1.0 / (1.0 + root2 * k + k2)

    if kind == "low":
        b = np.array([k2, 2.0 * k2, k2]) * norm
    elif kind == "high":
        b = np.array([1.0, -2.0, 1.0]) * norm
    else:                                               # pragma: no cover
        raise ValueError(f"kind must be 'low' or 'high', not {kind!r}")

    a = np.array([1.0,
                  2.0 * (k2 - 1.0) * norm,
                  (1.0 - root2 * k + k2) * norm])
    return b, a


def _lfilter(b, a, x: np.ndarray) -> np.ndarray:
    """Direct-form-II transposed IIR along axis 0. Columns are independent."""
    x = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x)
    z1 = np.zeros(x.shape[1])
    z2 = np.zeros(x.shape[1])
    b0, b1, b2 = b
    a1, a2 = a[1], a[2]
    for i in range(x.shape[0]):
        xi = x[i]
        yi = b0 * xi + z1
        z1 = b1 * xi - a1 * yi + z2
        z2 = b2 * xi - a2 * yi
        out[i] = yi
    return out


def _filtfilt(b, a, x: np.ndarray, pad: int) -> np.ndarray:
    """
    Zero-phase filtering: forward, then backward over the reversed signal.

    Phase matters here. A causal filter delays the envelope, and a delayed
    envelope moves a gesture's onset relative to the label window it was cut
    with -- so a one-directional filter would shift the very boundaries this
    dataset is built on. Running it both ways cancels the phase exactly, at
    the cost of doubling the effective order.

    Reflect-padded for the same reason `moving_rms` is: a zero-padded edge
    reads as a sudden drop to silence, and the filter rings on it.
    """
    n = x.shape[0]
    pad = int(max(0, min(pad, n - 1)))
    if pad:
        head = 2.0 * x[0] - x[pad:0:-1]
        tail = 2.0 * x[-1] - x[-2:-pad - 2:-1]
        work = np.vstack([head, x, tail])
    else:
        work = x
    work = _lfilter(b, a, work)
    work = _lfilter(b, a, work[::-1])[::-1]
    return work[pad:pad + n] if pad else work


def bandpass_filter(x: np.ndarray, fs: float, high_pass: float = None,
                    low_pass: float = None, order: int = None,
                    report: dict = None) -> np.ndarray:
    """
    Zero-phase Butterworth band-pass. Returns a filtered COPY.

    `high_pass` / `low_pass` are cutoffs in Hz; None skips that half. Defaults
    come from `EMG_BANDPASS_HZ[EMG_MODE]`, whose comment explains why the
    "filtered" default is a low-pass only -- in short, the Myo has already
    band-passed that stream in hardware, and a high-pass on the envelope it
    returns would delete the activation level that carries most of the signal.

    **A cutoff at or above Nyquist is SKIPPED, not clamped**, and the reason is
    recorded in `report` when one is passed. Clamping 450 Hz down to 24.9 Hz on
    a 50 Hz stream would quietly apply a completely different filter from the
    one that was asked for and still report it as 20-450 Hz. At these sample
    rates that is not a corner case -- it is what happens the first time
    someone copies a textbook band from a 1 kHz setup.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        return bandpass_filter(x[:, None], fs, high_pass, low_pass, order,
                               report)[:, 0]
    if report is None:
        report = {}
    notes = report.setdefault("notes", [])
    report.setdefault("applied", [])

    defaults = EMG_BANDPASS_HZ.get(EMG_MODE, (None, None))
    if high_pass is None and low_pass is None:
        high_pass, low_pass = defaults
    order = EMG_BANDPASS_ORDER if order is None else int(order)
    fs = float(fs or nominal_rate())
    nyquist = fs / 2.0

    if x.shape[0] < 9 or fs <= 0:
        notes.append(f"too short to filter ({x.shape[0]} samples)")
        return x.copy()

    #  3x the slowest cutoff's period, so the reflected edge covers the
    #  filter's own settling time.
    slowest = min([c for c in (high_pass, low_pass) if c] or [1.0])
    pad = int(min(max(9, round(3.0 * fs / slowest)), x.shape[0] - 1))

    out = x
    for cutoff, kind in ((high_pass, "high"), (low_pass, "low")):
        if not cutoff:
            continue
        if cutoff >= nyquist:
            notes.append(
                f"{kind}-pass {cutoff:.1f} Hz SKIPPED: at or above the "
                f"{nyquist:.1f} Hz Nyquist of this {fs:.1f} Hz stream")
            continue
        sections = max(1, order // 2)
        b, a = _butter2_coeffs(cutoff, fs, kind)
        for _ in range(sections):
            out = _filtfilt(b, a, out, pad)
        report["applied"].append({"kind": kind, "hz": float(cutoff),
                                  "order": int(sections * 2),
                                  "zero_phase": True})
    return np.ascontiguousarray(out)


def baseline_level(x: np.ndarray, n_samples: int) -> np.ndarray:
    """
    Per-channel DC/offset estimate from the first n_samples.

    The median is used instead of the mean because it is not pulled around by
    an early muscle burst -- important when the subject starts too soon.
    """
    x = np.asarray(x, dtype=np.float64)
    n = min(max(1, int(n_samples)), x.shape[0])
    return np.median(x[:n], axis=0)


def envelope(x: np.ndarray, fs: float, win_ms: float = None,
             baseline_ms: float = None) -> np.ndarray:
    """
    Linear-envelope representation: moving RMS of the baseline-removed signal.

    This is a *derived* view used for display, activity detection and feature
    extraction. It is never written over the raw channels.
    """
    win_ms = ENVELOPE_MS if win_ms is None else win_ms
    baseline_ms = TRIM_BASELINE_MS if baseline_ms is None else baseline_ms

    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()

    base = baseline_level(x, ms_to_samples(baseline_ms, fs))
    return moving_rms(x - base, ms_to_samples(win_ms, fs))


def normalize_per_channel(x: np.ndarray, mode: str = "mad",
                          ref: np.ndarray = None) -> Tuple[np.ndarray, dict]:
    """
    Produce a normalised representation WITHOUT clipping or crushing peaks.

    mode:
      "mad"    -> centre on the median, scale by the median absolute deviation.
                  Robust: a single explosive burst does not shrink the whole
                  recording the way max-normalisation would.
      "range"  -> scale by the device's full dynamic range (fully reversible).
      "zscore" -> classic mean/std.

    Returns (normalised_copy, scaling_parameters). The scaling parameters are
    stored with the data so the operation can always be undone.
    """
    x = np.asarray(x, dtype=np.float64)
    src = x if ref is None else np.asarray(ref, dtype=np.float64)

    if mode == "range":
        lo, hi = emg_range()
        centre = np.full(x.shape[1], (hi + lo) / 2.0)
        scale = np.full(x.shape[1], max(1e-9, (hi - lo) / 2.0))
    elif mode == "zscore":
        centre = src.mean(axis=0)
        scale = np.maximum(src.std(axis=0), 1e-9)
    else:  # "mad"
        centre = np.median(src, axis=0)
        mad = np.median(np.abs(src - centre), axis=0) * 1.4826
        scale = np.maximum(mad, 1e-9)

    return (x - centre) / scale, {
        "mode": mode,
        "centre": centre.tolist(),
        "scale": scale.tolist(),
    }


# ------------------------------------------------------------------------------
# Activity detection / automatic trimming
# ------------------------------------------------------------------------------


@dataclass
class ActivityResult:
    """Outcome of automatic gesture-onset/offset detection."""
    found: bool
    start: int = 0                  # inclusive, padding already applied
    end: int = 0                    # exclusive, padding already applied
    core_start: int = 0             # detected region without padding
    core_end: int = 0
    threshold_high: float = 0.0
    threshold_low: float = 0.0
    baseline: float = 0.0
    n_candidates: int = 0
    reason: str = ""

    @property
    def length(self) -> int:
        return max(0, self.end - self.start)


def _merge_and_filter_runs(mask: np.ndarray, min_active: int,
                           min_inactive: int) -> List[Tuple[int, int]]:
    """
    Turn a boolean activity mask into a list of (start, end) runs, bridging
    gaps shorter than min_inactive and dropping runs shorter than min_active.
    """
    if mask.size == 0 or not mask.any():
        return []

    padded = np.concatenate([[False], mask, [False]])
    edges = np.flatnonzero(padded[1:] != padded[:-1])
    runs = [(int(edges[i]), int(edges[i + 1])) for i in range(0, len(edges), 2)]

    # Bridge short inactive gaps.
    merged: List[List[int]] = []
    for start, end in runs:
        if merged and start - merged[-1][1] < min_inactive:
            merged[-1][1] = end
        else:
            merged.append([start, end])

    # Drop bursts that are too short to be a gesture.
    return [(s, e) for s, e in merged if (e - s) >= min_active]


def detect_activity(x: np.ndarray, fs: float, mode: str = None) -> ActivityResult:
    """
    Find the gesture in a recording using a hysteresis moving-RMS detector.

    Deliberately conservative about fast gestures:
      * the envelope window is short (TRIM_ENVELOPE_MS), so a 100 ms burst
        still produces a full-height envelope peak;
      * short inactive gaps are bridged, so a two-phase wave is one gesture;
      * generous padding is added back afterwards.

    Returns an ActivityResult. `found == False` means "do not trim", never
    "throw the recording away".
    """
    mode = mode or EMG_MODE
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]

    if n < 8:
        return ActivityResult(False, 0, n, 0, n, reason="recording too short")

    env = envelope(x, fs, TRIM_ENVELOPE_MS, TRIM_BASELINE_MS)

    n_base = ms_to_samples(TRIM_BASELINE_MS, fs)
    base_per_ch = np.median(env[:min(n_base, n)], axis=0)
    floor = float(mode_value(TRIM_ABS_FLOOR, mode, 1.0))

    hi_per_ch = np.maximum(base_per_ch * TRIM_THRESHOLD_HIGH, floor)
    lo_per_ch = np.maximum(base_per_ch * TRIM_THRESHOLD_LOW, floor * 0.6)

    need = int(np.clip(TRIM_MIN_ACTIVE_CHANNELS, 1, x.shape[1]))
    above_hi = (env > hi_per_ch).sum(axis=1) >= need
    above_lo = (env > lo_per_ch).sum(axis=1) >= need

    min_active = ms_to_samples(TRIM_MIN_ACTIVE_MS, fs)
    min_inactive = ms_to_samples(TRIM_MIN_INACTIVE_MS, fs)

    # Hysteresis: grow every confirmed high-threshold run out to the
    # surrounding low-threshold run, so onset and offset slopes are kept.
    seeds = _merge_and_filter_runs(above_hi, min_active, min_inactive)
    if not seeds:
        return ActivityResult(
            False, 0, n, 0, n,
            threshold_high=float(np.mean(hi_per_ch)),
            threshold_low=float(np.mean(lo_per_ch)),
            baseline=float(np.mean(base_per_ch)),
            reason="no sample crossed the high threshold",
        )

    grown: List[Tuple[int, int]] = []
    for s, e in seeds:
        a = s
        while a > 0 and above_lo[a - 1]:
            a -= 1
        b = e
        while b < n and above_lo[b]:
            b += 1
        grown.append((a, b))

    regions = _merge_and_filter_runs(
        _runs_to_mask(grown, n), min_active, min_inactive
    )
    if not regions:
        regions = grown

    # Keep the strongest region (largest envelope energy), not simply the
    # longest -- an explosive 150 ms burst must win over 1 s of drift.
    energies = [float(env[s:e].sum()) for s, e in regions]
    core_start, core_end = regions[int(np.argmax(energies))]

    pre = ms_to_samples(PRE_PADDING_MS, fs)
    post = ms_to_samples(POST_PADDING_MS, fs)
    start = int(max(0, core_start - pre))
    end = int(min(n, core_end + post))

    return ActivityResult(
        True, start, end, int(core_start), int(core_end),
        threshold_high=float(np.mean(hi_per_ch)),
        threshold_low=float(np.mean(lo_per_ch)),
        baseline=float(np.mean(base_per_ch)),
        n_candidates=len(regions),
        reason="ok",
    )


def _runs_to_mask(runs: Sequence[Tuple[int, int]], n: int) -> np.ndarray:
    mask = np.zeros(n, dtype=bool)
    for s, e in runs:
        mask[max(0, s):min(n, e)] = True
    return mask


# ------------------------------------------------------------------------------
# Feature extraction
# ------------------------------------------------------------------------------

# Time-domain features, in the order extract_features() returns them.
TIME_FEATURES = [
    "rms", "mav", "var", "std", "wl", "zc", "ssc",
    "peak", "iemg", "ptp", "skew",
]
FREQ_FEATURES = ["mnf", "mdf"]


def feature_names(include_freq: bool = True) -> List[str]:
    """Flat feature names, matching the extract_features() output order."""
    names = []
    for feat in TIME_FEATURES + (FREQ_FEATURES if include_freq else []):
        names.extend(f"{ch}_{feat}" for ch in CHANNEL_NAMES)
    return names


def extract_features(window: np.ndarray, fs: float = None,
                     include_freq: bool = True,
                     zc_threshold: float = 1e-6) -> np.ndarray:
    """
    Classic EMG feature vector for one (n_samples, 8) window.

    Returns a flat float32 array ordered feature-major: all 8 channels of
    `rms`, then all 8 of `mav`, and so on. The names are given by
    feature_names(), so a DataFrame can be built directly from the pair.

    The window is centred on its own median before the shape-based features are
    computed, so a DC offset (which the filtered Myo stream always has) does not
    masquerade as signal. Amplitude features stay on the original scale.
    """
    fs = float(fs or nominal_rate())
    x = np.asarray(window, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] < 2:
        n_feat = len(TIME_FEATURES) + (len(FREQ_FEATURES) if include_freq else 0)
        return np.zeros(n_feat * x.shape[1], dtype=np.float32)

    centred = x - np.median(x, axis=0)
    diff = np.diff(centred, axis=0)

    rms = np.sqrt(np.mean(centred ** 2, axis=0))
    mav = np.mean(np.abs(centred), axis=0)
    var = np.var(centred, axis=0)
    std = np.std(centred, axis=0)
    wl = np.sum(np.abs(diff), axis=0)                       # waveform length
    iemg = np.sum(np.abs(centred), axis=0)                  # integrated EMG
    peak = np.max(np.abs(centred), axis=0)
    ptp = np.ptp(x, axis=0)

    # Zero crossings (with a dead-band so noise does not inflate the count).
    sign = np.sign(centred)
    sign[np.abs(centred) < zc_threshold] = 0
    zc = np.sum((sign[:-1] * sign[1:]) < 0, axis=0).astype(np.float64)

    # Slope sign changes.
    if diff.shape[0] >= 2:
        ssc = np.sum((diff[:-1] * diff[1:]) < 0, axis=0).astype(np.float64)
    else:
        ssc = np.zeros(x.shape[1])

    with np.errstate(divide="ignore", invalid="ignore"):
        skew = np.nan_to_num(
            np.mean(centred ** 3, axis=0) / np.maximum(std ** 3, 1e-12)
        )

    parts = [rms, mav, var, std, wl, zc, ssc, peak, iemg, ptp, skew]

    if include_freq:
        mnf, mdf = _spectral_features(centred, fs)
        parts.extend([mnf, mdf])

    return np.concatenate(parts).astype(np.float32)


# Amplitude / spatial-profile features, in the order extract_level_features()
# returns them. Each is computed per channel.
LEVEL_FEATURES = ["mean", "median", "p90", "std", "share", "log_mean"]
LEVEL_GLOBAL_FEATURES = ["total_log", "focality"]


def level_feature_names() -> List[str]:
    names = []
    for feat in LEVEL_FEATURES:
        names.extend(f"{ch}_{feat}" for ch in CHANNEL_NAMES)
    names.extend(LEVEL_GLOBAL_FEATURES)
    return names


def extract_level_features(window: np.ndarray, fs: float = None) -> np.ndarray:
    """
    Amplitude and spatial-profile features for an ALREADY-RECTIFIED stream.

    This exists because `extract_features()` is wrong for the Myo's
    "filtered" mode. That stream is an on-device rectified envelope, not a
    zero-centred oscillation, and `extract_features()` subtracts each
    channel's median before computing its shape features -- correct for raw
    bipolar EMG, but here it deletes exactly what identifies a gesture: HOW
    HARD each channel is firing.

    Measured on this dataset, the between-class/within-class Fisher ratio is
    25.5 for the per-channel mean and only 15.1 after that median-centring.

    `share` (each channel's fraction of the total activation) is the spatial
    pattern with absolute amplitude divided out. That is the part that should
    transfer between people and between sessions, since it survives a
    stronger contraction, a tighter band, or different skin impedance.
    """
    x = np.asarray(window, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    if x.shape[0] == 0:
        return np.zeros(len(LEVEL_FEATURES) * x.shape[1]
                        + len(LEVEL_GLOBAL_FEATURES), dtype=np.float32)

    # The stream is non-negative already; abs() makes this safe if a caller
    # passes a signed (raw-mode) window instead.
    mag = np.abs(x)

    mean = mag.mean(axis=0)
    median = np.median(mag, axis=0)
    p90 = np.percentile(mag, 90, axis=0)
    std = mag.std(axis=0)

    total = mean.sum()
    share = mean / total if total > 0 else np.full_like(mean, 1.0 / mean.size)
    log_mean = np.log1p(mean)

    total_log = np.log1p(total)
    focality = float(mean.max() / mean.mean()) if mean.mean() > 0 else 0.0

    return np.concatenate([
        mean, median, p90, std, share, log_mean,
        np.array([total_log, focality]),
    ]).astype(np.float32)


def extract_all_features(window: np.ndarray, fs: float = None,
                         include_freq: bool = True) -> np.ndarray:
    """Classic shape features plus the amplitude/profile features."""
    return np.concatenate([
        extract_features(window, fs, include_freq=include_freq),
        extract_level_features(window, fs),
    ])


def all_feature_names(include_freq: bool = True) -> List[str]:
    return feature_names(include_freq) + level_feature_names()


# ------------------------------------------------------------------------------
# The single feature path shared by training and live inference
# ------------------------------------------------------------------------------
#
# train_model.py and LivePredictor both call model_feature_vector(). Keeping
# one implementation is not tidiness: if the two drifted apart by even one
# feature the live predictions would silently become garbage while the
# reported cross-validation score stayed perfect.


def model_feature_vector(window: np.ndarray, representation: str = "raw",
                         feature_set: str = "level",
                         include_freq: bool = True,
                         fs: float = None) -> np.ndarray:
    """One analysis window -> the feature vector a trained model expects."""
    block = np.asarray(window, dtype=np.float64)
    fs = float(fs or nominal_rate())

    if representation == "normalized":
        block = normalize_per_channel(block, "mad")[0]
    elif representation == "envelope":
        block = envelope(block, fs)
    elif representation == "bandpass":
        #  One representation, so the trainer and LivePredictor filter
        #  identically -- the whole reason this function is shared.
        block = bandpass_filter(block, fs)

    if feature_set == "level":
        return extract_level_features(block, fs)
    if feature_set == "both":
        return extract_all_features(block, fs, include_freq)
    return extract_features(block, fs, include_freq=include_freq)


def model_feature_names(feature_set: str = "level",
                        include_freq: bool = True) -> List[str]:
    if feature_set == "level":
        return level_feature_names()
    if feature_set == "both":
        return all_feature_names(include_freq)
    return feature_names(include_freq)


def _spectral_features(x: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Mean frequency and median frequency per channel, from the periodogram."""
    n = x.shape[0]
    win = np.hanning(n)[:, None] if n > 2 else np.ones((n, 1))
    spec = np.abs(np.fft.rfft(x * win, axis=0)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / fs)[:, None]

    total = spec.sum(axis=0)
    total_safe = np.where(total > 0, total, 1.0)
    mnf = (spec * freqs).sum(axis=0) / total_safe

    cumulative = np.cumsum(spec, axis=0)
    half = total_safe / 2.0
    idx = (cumulative < half).sum(axis=0)
    idx = np.clip(idx, 0, spec.shape[0] - 1)
    mdf = freqs[:, 0][idx]

    mnf = np.where(total > 0, mnf, 0.0)
    mdf = np.where(total > 0, mdf, 0.0)
    return mnf, mdf


# ------------------------------------------------------------------------------
# Windowing
# ------------------------------------------------------------------------------


def sliding_windows(x: np.ndarray, win: int, step: int) -> np.ndarray:
    """
    (n_windows, win, n_channels) view-based windowing. No copy, no padding --
    a trailing partial window is dropped rather than zero-filled, because a
    zero-filled tail would teach a model to recognise padding.
    """
    x = np.asarray(x)
    n = x.shape[0]
    win = int(max(1, win))
    step = int(max(1, step))
    if n < win:
        return np.empty((0, win) + x.shape[1:], dtype=x.dtype)

    n_win = 1 + (n - win) // step
    return np.lib.stride_tricks.as_strided(
        x,
        shape=(n_win, win) + x.shape[1:],
        strides=(x.strides[0] * step,) + x.strides,
        writeable=False,
    )


def resample_uniform(emg: np.ndarray, t: np.ndarray, n_out: int) -> np.ndarray:
    """
    Resample an (n, 8) block onto `n_out` evenly spaced samples.

    This exists because the Myo's filtered stream is NOT evenly sampled: the
    dongle delivers samples in pairs about 32 ms apart, and the effective rate
    measured across this dataset ranges from 28 Hz to 70 Hz depending on link
    conditions. A model trained on "13 samples" would therefore see windows
    covering anywhere from 185 ms to 460 ms of real time.

    Resampling onto a fixed grid makes a window mean a fixed DURATION, so the
    same feature vector means the same thing at training and at inference
    regardless of how the link happened to behave. Linear interpolation is
    deliberate: it does not invent peaks the way a spline can.
    """
    emg = np.asarray(emg, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).reshape(-1)
    n_out = int(max(2, n_out))

    if emg.shape[0] < 2:
        return np.repeat(emg.reshape(1, -1) if emg.size else
                         np.zeros((1, N_CHANNELS)), n_out, axis=0)

    # Guard against duplicate timestamps from burst delivery.
    t = np.maximum.accumulate(t)
    span = t[-1] - t[0]
    if span <= 0:
        grid = np.linspace(0, emg.shape[0] - 1, n_out)
        src = np.arange(emg.shape[0], dtype=np.float64)
    else:
        grid = np.linspace(t[0], t[-1], n_out)
        src = t

    return np.stack([np.interp(grid, src, emg[:, c])
                     for c in range(emg.shape[1])], axis=1)


def window_params(fs: float, window_ms: float = None,
                  overlap: float = None) -> Tuple[int, int]:
    """(window_length, step) in samples, from the ms/overlap configuration."""
    window_ms = WINDOW_MS if window_ms is None else window_ms
    overlap = WINDOW_OVERLAP if overlap is None else overlap
    win = ms_to_samples(window_ms, fs)
    step = max(1, int(round(win * (1.0 - float(np.clip(overlap, 0.0, 0.95))))))
    return win, step


# ------------------------------------------------------------------------------
# Augmentation (never applied to raw recordings -- callers build new arrays)
# ------------------------------------------------------------------------------


def augment_amplitude(x: np.ndarray, factor_range=(0.9, 1.1),
                      rng: np.random.Generator = None) -> np.ndarray:
    """Global gain jitter: models different contraction strength / skin impedance."""
    rng = rng or np.random.default_rng()
    return np.asarray(x, dtype=np.float32) * float(rng.uniform(*factor_range))


def augment_time_warp(x: np.ndarray, factor_range=(0.9, 1.1),
                      rng: np.random.Generator = None) -> np.ndarray:
    """Resample in time: models a faster or slower performance of the gesture."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    factor = float(rng.uniform(*factor_range))
    m = max(2, int(round(n * factor)))
    src = np.linspace(0.0, n - 1.0, n)
    dst = np.linspace(0.0, n - 1.0, m)
    out = np.stack([np.interp(dst, src, x[:, c]) for c in range(x.shape[1])], axis=1)
    return out.astype(np.float32)


def augment_noise(x: np.ndarray, sigma_fraction: float = 0.02,
                  rng: np.random.Generator = None) -> np.ndarray:
    """Additive Gaussian noise scaled to each channel's own RMS."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=np.float32)
    rms = np.sqrt(np.mean((x - x.mean(axis=0)) ** 2, axis=0))
    return x + rng.normal(0.0, np.maximum(rms, 1e-6) * sigma_fraction, x.shape).astype(np.float32)


def augment_channel_dropout(x: np.ndarray, n_drop: int = 1,
                            rng: np.random.Generator = None) -> np.ndarray:
    """Zero out random channels: models a lifted electrode / rotated armband."""
    rng = rng or np.random.default_rng()
    out = np.asarray(x, dtype=np.float32).copy()
    if out.shape[1] and n_drop > 0:
        for c in rng.choice(out.shape[1], size=min(n_drop, out.shape[1]), replace=False):
            out[:, c] = np.median(out[:, c])
    return out


def augment_baseline_shift(x: np.ndarray, shift_fraction: float = 0.05,
                           rng: np.random.Generator = None) -> np.ndarray:
    """Per-channel DC offset: models electrode drift between sessions."""
    rng = rng or np.random.default_rng()
    x = np.asarray(x, dtype=np.float32)
    span = np.ptp(x, axis=0)
    return x + rng.uniform(-1, 1, x.shape[1]).astype(np.float32) * span * shift_fraction


def augment_rotate_channels(x: np.ndarray, shift: int = 1) -> np.ndarray:
    """
    Circularly rotate the channel order: models the armband being worn rotated
    by one or more pod positions. Very effective for cross-subject robustness.
    """
    return np.roll(np.asarray(x, dtype=np.float32), int(shift), axis=1)


AUGMENTATIONS: Dict[str, Callable] = {
    "amplitude": augment_amplitude,
    "time_warp": augment_time_warp,
    "noise": augment_noise,
    "channel_dropout": augment_channel_dropout,
    "baseline_shift": augment_baseline_shift,
    "rotate_channels": augment_rotate_channels,
}


# ==============================================================================
# SECTION 3 -- QUALITY CHECKS
#
# These never delete or silently alter a recording: they return a list of
# findings which the GUI shows and which get stored in the metadata.
# ==============================================================================


@dataclass
class QualityIssue:
    level: str          # "error" | "warning" | "info"
    code: str
    message: str

    def __str__(self) -> str:
        return f"[{self.level.upper()}] {self.message}"


def check_quality(x: np.ndarray, timestamps: np.ndarray, label: str,
                  expected_seconds: float = None, mode: str = None,
                  had_disconnect: bool = False) -> List[QualityIssue]:
    """Run every configured sanity check over one recording."""
    mode = mode or EMG_MODE
    issues: List[QualityIssue] = []

    if not QC_ENABLED:
        return issues

    x = np.asarray(x, dtype=np.float64)
    timestamps = np.asarray(timestamps, dtype=np.float64)
    n = x.shape[0]

    if n == 0:
        issues.append(QualityIssue("error", "empty", "Recording contains no samples."))
        return issues

    if had_disconnect:
        issues.append(QualityIssue(
            "error", "disconnect",
            "The Myo link dropped during this recording -- the signal is incomplete.",
        ))

    # --- invalid values -------------------------------------------------------
    if not np.isfinite(x).all():
        issues.append(QualityIssue(
            "error", "invalid_values",
            f"{int((~np.isfinite(x)).sum())} non-finite (NaN/Inf) sample values.",
        ))

    lo, hi = emg_range(mode)
    out_of_range = int(((x < lo) | (x > hi)).sum())
    if out_of_range:
        issues.append(QualityIssue(
            "warning", "out_of_range",
            f"{out_of_range} values outside the expected device range [{lo}, {hi}].",
        ))

    # --- sample count ---------------------------------------------------------
    fs_nominal = nominal_rate(mode)
    if expected_seconds:
        expected = expected_seconds * fs_nominal
        if expected > 0 and n < expected * QC_MIN_SAMPLE_FRACTION:
            issues.append(QualityIssue(
                "warning", "missing_samples",
                f"Only {n} samples, expected about {int(expected)} "
                f"({n / expected * 100:.0f} %). Packets were probably dropped.",
            ))

    # --- gaps -----------------------------------------------------------------
    if timestamps.size >= 2:
        gaps_ms = np.diff(timestamps) * 1000.0
        worst = float(gaps_ms.max())
        if worst > QC_MAX_GAP_MS:
            n_bad = int((gaps_ms > QC_MAX_GAP_MS).sum())
            issues.append(QualityIssue(
                "warning", "gap",
                f"{n_bad} gap(s) in the stream, worst {worst:.0f} ms "
                f"(limit {QC_MAX_GAP_MS:.0f} ms).",
            ))
        measured = measured_rate(timestamps, fs_nominal)
        if measured < fs_nominal * 0.7 or measured > fs_nominal * 1.4:
            issues.append(QualityIssue(
                "info", "rate_mismatch",
                f"Measured sample rate {measured:.0f} Hz differs from the "
                f"configured {fs_nominal:.0f} Hz for mode '{mode}'.",
            ))

    # --- saturation -----------------------------------------------------------
    sat_hi = lo + (hi - lo) * SATURATION_FRACTION
    sat_lo = lo + (hi - lo) * (1.0 - SATURATION_FRACTION)
    saturated = (x >= sat_hi) | (x <= sat_lo) if lo < 0 else (x >= sat_hi)
    frac = float(saturated.mean())
    if frac > QC_MAX_SATURATED_FRACTION:
        worst_ch = int(saturated.mean(axis=0).argmax())
        issues.append(QualityIssue(
            "warning", "saturated",
            f"{frac * 100:.1f} % of samples are clipped at the device limit "
            f"(worst: {CHANNEL_NAMES[worst_ch]}). Peaks may be lost.",
        ))

    # --- constant / dead channels --------------------------------------------
    spread = x.max(axis=0) - x.min(axis=0)
    dead = [CHANNEL_NAMES[c] for c in range(x.shape[1])
            if spread[c] <= QC_FLAT_CHANNEL_TOLERANCE]
    if dead:
        issues.append(QualityIssue(
            "error", "flat_channel",
            f"Channel(s) {', '.join(dead)} never change -- lost electrode contact.",
        ))

    # --- baseline noise -------------------------------------------------------
    fs = measured_rate(timestamps, fs_nominal)
    n_base = ms_to_samples(TRIM_BASELINE_MS, fs)
    if n > n_base:
        base = x[:n_base]
        base_rms = float(np.sqrt(np.mean((base - np.median(base, axis=0)) ** 2)))
        limit = float(mode_value(QC_MAX_BASELINE_RMS, mode, 1e9))
        if base_rms > limit:
            issues.append(QualityIssue(
                "warning", "noisy_baseline",
                f"Resting baseline RMS {base_rms:.1f} exceeds {limit:.0f}. "
                "Check skin contact and move away from mains interference.",
            ))

    # --- amplitude sanity -----------------------------------------------------
    env = envelope(x, fs)
    peak = float(env.max())
    min_peak = float(mode_value(QC_MIN_PEAK_AMPLITUDE, mode, 0.0))
    if label not in ("REST",) and peak < min_peak:
        issues.append(QualityIssue(
            "warning", "no_activity",
            f"Peak envelope only {peak:.1f} (expected at least {min_peak:.0f}) -- "
            f"the '{label}' contraction may be missing.",
        ))
    if label == "REST" and peak > min_peak * 2.5 and min_peak > 0:
        issues.append(QualityIssue(
            "info", "rest_not_quiet",
            f"REST recording has a peak envelope of {peak:.1f}; the muscle was "
            "probably not fully relaxed.",
        ))

    return issues


def measured_rate(timestamps: np.ndarray, fallback: float = None) -> float:
    """
    Effective sample rate of a recording: samples per second over its span.

    Deliberately NOT 1/median(diff): the BLE dongle delivers two to four
    buffered packets at once, microseconds apart, then goes quiet until the
    next burst. The median interval is therefore an intra-burst interval, and
    estimating from it reports absurd rates (measured here: 9258 Hz for a
    57 Hz stream). The span average is immune to that clustering.

    A long dropout lowers this estimate, which is correct -- it is the rate
    actually achieved. Gaps are reported separately by check_quality().
    """
    fallback = float(fallback or nominal_rate())
    t = np.asarray(timestamps, dtype=np.float64)
    if t.size < 3:
        return fallback
    span = float(t[-1] - t[0])
    if span <= 0:
        return fallback
    return float((t.size - 1) / span)


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
        "app_version": APP_VERSION,
        "emg_mode": EMG_MODE,
        "nominal_sample_rate": nominal_rate(),
        "emg_range": list(emg_range()),
        "gestures": list(GESTURES),
        "repetitions": REPETITIONS,
        "gesture_order": GESTURE_ORDER,
        "random_seed": RANDOM_SEED,
        "rest_time": REST_TIME,
        "prepare_time": PREPARE_TIME,
        "countdown_time": COUNTDOWN_TIME,
        "record_time": RECORD_TIME,
        "pause_time": PAUSE_TIME,
        "pre_gesture_ms": PRE_GESTURE_MS,
        "post_gesture_ms": POST_GESTURE_MS,
        "window_ms": WINDOW_MS,
        "window_overlap": WINDOW_OVERLAP,
        "auto_trim": AUTO_TRIM,
        "default_cut_seconds": LABEL_DEFAULT_SELECTION_S,
        "model_spec_file": MODEL_SPEC_NAME,
        "min_selection_seconds": LABEL_MIN_SELECTION_S,
        "free_take_max_minutes": FREE_TAKE_MAX_MINUTES,
        "trim": {
            "envelope_ms": TRIM_ENVELOPE_MS,
            "baseline_ms": TRIM_BASELINE_MS,
            "threshold_high": TRIM_THRESHOLD_HIGH,
            "threshold_low": TRIM_THRESHOLD_LOW,
            "abs_floor": mode_value(TRIM_ABS_FLOOR),
            "min_active_ms": TRIM_MIN_ACTIVE_MS,
            "min_inactive_ms": TRIM_MIN_INACTIVE_MS,
            "pre_padding_ms": PRE_PADDING_MS,
            "post_padding_ms": POST_PADDING_MS,
            "min_active_channels": TRIM_MIN_ACTIVE_CHANNELS,
        },
        "quality": {
            "enabled": QC_ENABLED,
            "min_sample_fraction": QC_MIN_SAMPLE_FRACTION,
            "max_gap_ms": QC_MAX_GAP_MS,
            "max_saturated_fraction": QC_MAX_SATURATED_FRACTION,
            "max_baseline_rms": mode_value(QC_MAX_BASELINE_RMS),
            "min_peak_amplitude": mode_value(QC_MIN_PEAK_AMPLITUDE),
        },
    }


# ------------------------------------------------------------------------------
# Model specification
# ------------------------------------------------------------------------------
#
# A spec is plain JSON so it can sit next to the model, be diffed, and be
# handed to a subprocess:
#
#     {"version": 1,
#      "voting": {"enabled": true, "mode": "soft", "always": false},
#      "models": [{"name": "...", "kind": "lda", "enabled": true,
#                  "scaler": "standard", "params": {...}}]}
#
# Everything in here is Qt-free and sklearn-free: the GUI edits it, the trainer
# turns it into estimators, and neither needs the other's dependencies.

MODEL_SPEC_VERSION = 1


def default_model_spec() -> dict:
    """The roster the sweep has always run, as a spec."""
    return {
        "version": MODEL_SPEC_VERSION,
        "voting": {
            "enabled": bool(MODEL_VOTING_ENABLED),
            "mode": str(MODEL_VOTING_MODE),
            "always": bool(MODEL_VOTING_ALWAYS),
        },
        "auto": {
            "enabled": bool(MODEL_AUTO_ENABLED),
            "max_per_model": int(MODEL_AUTO_MAX_PER_MODEL),
        },
        "models": [
            {"name": name, "kind": kind, "enabled": bool(enabled),
             "scaler": scaler, "params": dict(params)}
            for name, kind, enabled, scaler, params in DEFAULT_MODEL_SPEC_MODELS
        ],
    }


def model_param_default(kind: str, param: str):
    """The schema default for one hyperparameter."""
    spec = MODEL_KINDS.get(kind, {}).get("params", {}).get(param)
    if not spec:
        return None
    value = spec.get("default")
    return list(value) if isinstance(value, (list, tuple)) else value


def _coerce_param(rule: dict, value, problems: list, where: str):
    """One hyperparameter, coerced and clamped to its schema rule."""
    kind = rule.get("type")
    default = rule.get("default")

    if kind == "choice":
        allowed = [str(v) for v in rule.get("values", [])]
        text = str(value)
        if text not in allowed:
            problems.append(f"{where}: {text!r} is not one of {allowed}; "
                            f"using {default!r}")
            return default
        return text

    if kind == "int":
        try:
            number = int(round(float(value)))
        except (TypeError, ValueError):
            problems.append(f"{where}: {value!r} is not a number; "
                            f"using {default!r}")
            return default
        low, high = rule.get("min"), rule.get("max")
        if low is not None and number < low:
            problems.append(f"{where}: {number} is below {low}; clamped")
            number = int(low)
        if high is not None and number > high:
            problems.append(f"{where}: {number} is above {high}; clamped")
            number = int(high)
        return number

    if kind == "float":
        try:
            number = float(value)
        except (TypeError, ValueError):
            problems.append(f"{where}: {value!r} is not a number; "
                            f"using {default!r}")
            return default
        low, high = rule.get("min"), rule.get("max")
        if low is not None and number < low:
            problems.append(f"{where}: {number} is below {low}; clamped")
            number = float(low)
        if high is not None and number > high:
            problems.append(f"{where}: {number} is above {high}; clamped")
            number = float(high)
        return number

    if kind == "ints":
        raw = value
        if isinstance(raw, str):
            raw = [part for part in re.split(r"[,\s]+", raw.strip()) if part]
        if not isinstance(raw, (list, tuple)):
            raw = [raw]
        out = []
        for item in raw:
            try:
                number = int(round(float(item)))
            except (TypeError, ValueError):
                problems.append(f"{where}: {item!r} is not a number; dropped")
                continue
            low, high = rule.get("min"), rule.get("max")
            if low is not None:
                number = max(int(low), number)
            if high is not None:
                number = min(int(high), number)
            out.append(number)
        limit = rule.get("max_items")
        if limit and len(out) > limit:
            problems.append(f"{where}: more than {limit} layers; truncated")
            out = out[:limit]
        if not out:
            problems.append(f"{where}: empty; using {default!r}")
            return list(default or [])
        return out

    return value                                             # unknown rule type


def validate_model_spec(spec: dict):
    """
    Return (clean_spec, problems).

    Never raises and never returns something the trainer cannot build: an
    unknown model kind is dropped, a bad value is coerced or clamped, and a
    spec with nothing enabled falls back to the defaults. `problems` is a list
    of human-readable strings that the GUI shows and the trainer logs.
    """
    problems = []
    spec = spec if isinstance(spec, dict) else {}

    raw_voting = spec.get("voting")
    raw_voting = raw_voting if isinstance(raw_voting, dict) else {}
    mode = str(raw_voting.get("mode", MODEL_VOTING_MODE))
    if mode not in ("soft", "hard"):
        problems.append(f"voting mode {mode!r} is not 'soft' or 'hard'; "
                        "using 'soft'")
        mode = "soft"
    voting = {
        "enabled": bool(raw_voting.get("enabled", MODEL_VOTING_ENABLED)),
        "mode": mode,
        "always": bool(raw_voting.get("always", MODEL_VOTING_ALWAYS)),
    }

    raw_auto = spec.get("auto")
    raw_auto = raw_auto if isinstance(raw_auto, dict) else {}
    try:
        per_model = int(raw_auto.get("max_per_model",
                                     MODEL_AUTO_MAX_PER_MODEL))
    except (TypeError, ValueError):
        problems.append("auto.max_per_model is not a number; using "
                        f"{MODEL_AUTO_MAX_PER_MODEL}")
        per_model = MODEL_AUTO_MAX_PER_MODEL
    if per_model < 1:
        problems.append("auto.max_per_model below 1; using 1")
        per_model = 1
    if per_model > 64:
        problems.append("auto.max_per_model above 64; clamped -- every "
                        "combination costs a full cross-validation")
        per_model = 64
    auto = {"enabled": bool(raw_auto.get("enabled", MODEL_AUTO_ENABLED)),
            "max_per_model": per_model}

    models = []
    seen_names = set()
    for index, entry in enumerate(spec.get("models") or []):
        if not isinstance(entry, dict):
            problems.append(f"model #{index + 1} is not an object; dropped")
            continue
        kind = str(entry.get("kind", ""))
        if kind not in MODEL_KINDS:
            problems.append(
                f"model #{index + 1}: unknown kind {kind!r}; dropped "
                f"(known: {', '.join(sorted(MODEL_KINDS))})")
            continue

        schema = MODEL_KINDS[kind]
        name = str(entry.get("name") or kind).strip() or kind
        if name in seen_names:
            # Two candidates with one name would collide in the results table
            # and in the saved report, so make it unique rather than drop it.
            suffix = 2
            while f"{name}-{suffix}" in seen_names:
                suffix += 1
            problems.append(f"duplicate model name {name!r}; "
                            f"renamed to {name}-{suffix}")
            name = f"{name}-{suffix}"
        seen_names.add(name)

        scaler = str(entry.get("scaler", schema.get("scaler", "none")))
        if scaler not in MODEL_SCALERS:
            problems.append(f"{name}: unknown scaler {scaler!r}; "
                            f"using {schema.get('scaler', 'none')!r}")
            scaler = schema.get("scaler", "none")

        given = entry.get("params")
        given = given if isinstance(given, dict) else {}
        params = {}
        for param, rule in schema["params"].items():
            if param in given:
                params[param] = _coerce_param(
                    rule, given[param], problems, f"{name}.{param}")
            else:
                default = rule.get("default")
                params[param] = (list(default)
                                 if isinstance(default, (list, tuple))
                                 else default)
        for unknown in sorted(set(given) - set(schema["params"])):
            problems.append(f"{name}: {unknown!r} is not a parameter of "
                            f"{kind}; ignored")

        models.append({"name": name, "kind": kind,
                       "enabled": bool(entry.get("enabled", True)),
                       "scaler": scaler, "params": params,
                       # Per-model AUTO: tune only the ones worth tuning.
                       "auto": bool(entry.get("auto", True))})

    if not models:
        problems.append("no usable models in the spec; using the defaults")
        return default_model_spec(), problems
    if not any(m["enabled"] for m in models):
        problems.append("every model was switched off; enabling the first one "
                        "so there is something to train")
        models[0]["enabled"] = True

    return {"version": MODEL_SPEC_VERSION, "voting": voting,
            "auto": auto, "models": models}, problems


def model_search_space(kind: str) -> dict:
    """What AUTO mode will try for one kind. {} = nothing to tune."""
    return {param: list(values)
            for param, values in MODEL_SEARCH_SPACES.get(kind, {}).items()
            if param in MODEL_KINDS.get(kind, {}).get("params", {})}


def auto_combinations(entry: dict, limit: int, seed: int = 0):
    """
    The parameter combinations AUTO mode will try for one model.

    The full grid is sampled down to `limit` with a seeded shuffle, so a
    rerun produces the same candidates -- a sweep you cannot reproduce is not
    a result. The model's own configured values are always combination #1, so
    turning AUTO on can never lose the setting you had.
    """
    space = model_search_space(entry["kind"])
    if not space or limit <= 1:
        return [dict(entry.get("params") or {})]

    base = dict(entry.get("params") or {})
    grid = [dict(base)]
    for param, values in sorted(space.items()):
        grown = []
        for partial in grid:
            for value in values:
                candidate = dict(partial)
                candidate[param] = (list(value)
                                    if isinstance(value, (list, tuple))
                                    else value)
                grown.append(candidate)
        grid = grown

    # De-duplicate, keeping the configured one first.
    seen, unique = set(), []
    for candidate in [base] + grid:
        key = json.dumps(candidate, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)

    if len(unique) <= limit:
        return unique
    rest = unique[1:]
    random.Random(seed).shuffle(rest)
    return [unique[0]] + rest[:limit - 1]


def expand_auto_spec(spec: dict, seed: int = 0):
    """
    Turn AUTO mode into ordinary candidates, and return (spec, note).

    Every combination becomes its own named entry, so it is trained and scored
    by exactly the same cross-validation as a hand-written one and shows up as
    its own row in the results. There is no second, hidden evaluation path --
    which is the point: a tuned model has to earn its score the same way.

    Caller beware, and the GUI says so: choosing the best of many
    combinations on the same folds makes the WINNER's score optimistic. More
    combinations, more optimism. The honest number for a tuned pipeline is a
    fresh subject, which is what leave-one-subject-out reports.
    """
    clean, _ = validate_model_spec(spec)
    auto = clean.get("auto") or {}
    if not auto.get("enabled"):
        return clean, "auto mode off"

    limit = int(auto.get("max_per_model", MODEL_AUTO_MAX_PER_MODEL))
    expanded, grown = [], 0
    for entry in clean["models"]:
        if not entry["enabled"] or not entry.get("auto", True):
            expanded.append(entry)
            continue

        combos = auto_combinations(entry, limit, seed=seed)
        if len(combos) <= 1:
            expanded.append(entry)
            continue

        space = model_search_space(entry["kind"])
        for index, params in enumerate(combos):
            # Name each variant by what actually differs, so the results
            # table is readable instead of being full of "RF-3".
            bits = []
            for param in sorted(space):
                value = params.get(param)
                if isinstance(value, list):
                    value = "x".join(str(v) for v in value)
                bits.append(f"{param}={value}")
            expanded.append({
                "name": f"{entry['name']}[{', '.join(bits)}]" if bits
                        else f"{entry['name']}#{index + 1}",
                "kind": entry["kind"], "enabled": True,
                "scaler": entry["scaler"], "params": params,
                "auto": False,
                "auto_from": entry["name"],
            })
            grown += 1

    clean = dict(clean)
    clean["models"] = expanded
    final, _ = validate_model_spec(clean)
    # validate_model_spec drops keys it does not know, so put the provenance
    # back: the report needs to say which entry a variant came from.
    by_name = {m["name"]: m for m in expanded}
    for model in final["models"]:
        source = by_name.get(model["name"], {}).get("auto_from")
        if source:
            model["auto_from"] = source
    return final, (f"auto mode: {grown} combination(s) generated, "
                   f"{len([m for m in final['models'] if m['enabled']])} "
                   f"candidates in total")


def enabled_models(spec: dict):
    """The entries the sweep will actually train, in order."""
    clean, _ = validate_model_spec(spec)
    return [m for m in clean["models"] if m["enabled"]]


# ------------------------------------------------------------------------------
# COMPUTE DEVICE  (is a CUDA GPU actually going to be used?)
# ------------------------------------------------------------------------------
#
# Only torch can answer this, and the recorder deliberately does not import
# torch (CLAUDE.md section 13: no torch dependency in the application -- it
# would add seconds to startup and a GPU context to a process whose job is to
# not drop EMG samples). So the probe runs in a short-lived subprocess, the
# same way training does, and the answer is cached for the session.
#
# The split: this module owns the probe snippet and the parser, so both are
# testable headless; the GUI owns running it without blocking the event loop.

COMPUTE_PROBE_TIMEOUT_S = 30.0

#  Printed as one JSON line on stdout. Deliberately tiny and dependency-free:
#  anything that raises must still produce a parseable answer.
COMPUTE_PROBE_SNIPPET = (
    "import json\n"
    "out = {'probed': True, 'torch': None, 'cuda': False, 'gpu': None,\n"
    "       'cuda_runtime': None, 'capability': None,\n"
    "       'total_memory_mb': None, 'reason': ''}\n"
    "try:\n"
    "    import torch\n"
    "    out['torch'] = torch.__version__\n"
    "    if torch.cuda.is_available():\n"
    "        p = torch.cuda.get_device_properties(0)\n"
    "        out.update(cuda=True, gpu=torch.cuda.get_device_name(0),\n"
    "                   cuda_runtime=torch.version.cuda,\n"
    "                   capability='%d.%d' % (p.major, p.minor),\n"
    "                   total_memory_mb=int(p.total_memory / 1048576))\n"
    "    elif '+cu' not in torch.__version__:\n"
    "        out['reason'] = 'CPU-only torch build'\n"
    "    else:\n"
    "        out['reason'] = 'no usable GPU or driver'\n"
    "except ImportError:\n"
    "    out['reason'] = 'torch is not installed'\n"
    "except Exception as exc:\n"
    "    out['reason'] = '%s: %s' % (type(exc).__name__, exc)\n"
    "print('COMPUTE ' + json.dumps(out))\n"
)


def blank_compute_info(reason: str = "not checked yet",
                       failed: bool = False) -> dict:
    """
    The shape every consumer can rely on, with nothing known yet.

    `failed` separates "the probe has not run yet" from "the probe ran and
    could not tell us". Both mean the device is unknown, but only the first
    is worth waiting out, and neither may be shown as "CPU".
    """
    return {"probed": False, "failed": bool(failed), "torch": None,
            "cuda": False, "gpu": None, "cuda_runtime": None,
            "capability": None, "total_memory_mb": None, "reason": reason}


def parse_compute_probe(text: str) -> dict:
    """
    Pull the probe's answer out of its stdout.

    Never raises: a probe that printed nothing usable means "unknown", which
    is a different thing from "no GPU" and is reported as such.
    """
    for line in reversed(str(text or "").splitlines()):
        line = line.strip()
        if not line.startswith("COMPUTE "):
            continue
        try:
            found = json.loads(line[len("COMPUTE "):])
        except (ValueError, TypeError):
            break
        if not isinstance(found, dict):
            break
        info = blank_compute_info("")
        info.update({k: v for k, v in found.items() if k in info})
        info["probed"] = True
        return info
    return blank_compute_info("the device probe did not answer", failed=True)


def detect_compute_device(timeout_s: float = None) -> dict:
    """
    Blocking probe. Used by headless callers and the tests; the GUI runs the
    same snippet through QProcess instead so it never stalls the event loop.
    """
    try:
        done = subprocess.run(
            [sys.executable, "-c", COMPUTE_PROBE_SNIPPET],
            capture_output=True, text=True,
            timeout=timeout_s or COMPUTE_PROBE_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return blank_compute_info("the device probe timed out", failed=True)
    except OSError as exc:
        return blank_compute_info(f"could not start the device probe: {exc}",
                                  failed=True)
    return parse_compute_probe(done.stdout or "")


def compute_badge(info: dict) -> tuple:
    """
    `(headline, detail, severity)` for the GUI's compute badge.

    severity is "good" / "warn" / "dim", which the GUI maps to a colour.

    The headline says what the HARDWARE offers. The detail says what each
    trainer actually uses, because those are not the same thing: the
    scikit-learn sweep behind RETRAIN has no GPU backend and runs on the CPU
    whatever card is installed. A badge that said "GPU" next to a button that
    only ever uses the CPU would be a lie.
    """
    info = info or blank_compute_info()
    if not info.get("probed"):
        if info.get("failed"):
            return ("DEVICE UNKNOWN",
                    f"{info.get('reason') or 'the probe failed'}.\n"
                    "Run `python train_torch.py --quick --no-save` in a "
                    "terminal to see what the GPU situation is.",
                    "warn")
        return ("checking compute device...", info.get("reason", ""), "dim")

    if info.get("cuda"):
        head = f"GPU READY  --  {info.get('gpu') or 'CUDA device'}"
        bits = [b for b in (
            f"CUDA {info['cuda_runtime']}" if info.get("cuda_runtime") else "",
            f"torch {info['torch']}" if info.get("torch") else "",
            f"{info['total_memory_mb']} MB" if info.get("total_memory_mb") else "",
            f"capability {info['capability']}" if info.get("capability") else "",
        ) if b]
        detail = (", ".join(bits) + "\n"
                  "RETRAIN (scikit-learn) runs on the CPU -- sklearn has no "
                  "GPU backend.\ntrain_torch.py uses this GPU.")
        return (head, detail, "good")

    reason = info.get("reason") or "no CUDA GPU found"
    head = "CPU ONLY"
    detail = f"{reason}.\nEvery trainer runs on the CPU."
    if info.get("torch"):
        detail = f"torch {info['torch']}, {detail}"
    else:
        detail += ("\nFor CUDA: python -m pip install torch==2.9.1+cu128 "
                   "--index-url https://download.pytorch.org/whl/cu128")
    return (head, detail, "warn")


def model_spec_path(model_path: str = None) -> str:
    """`models/model_spec.json` -- next to the model it describes."""
    target = model_path or MODEL_PATH
    if not os.path.isabs(target):
        target = os.path.join(PROJECT_DIR, target)
    return os.path.join(os.path.dirname(os.path.abspath(target)),
                        MODEL_SPEC_NAME)


def panel_path(model_path: str = None) -> str:
    """`models/gesture_panel.joblib` -- next to the model it explains."""
    target = model_path or MODEL_PATH
    if not os.path.isabs(target):
        target = os.path.join(PROJECT_DIR, target)
    return os.path.join(os.path.dirname(os.path.abspath(target)),
                        PANEL_FILENAME)


def save_model_spec(path: str, spec: dict) -> str:
    """
    Write a spec, validated first.

    Overwrites deliberately: unlike a recording, a spec is a setting, and
    keeping fifteen numbered copies of it would make it impossible to tell
    which one the trainer will read. The file it produces is copied into every
    model's payload, so the provenance of a trained model is not lost.
    """
    clean, problems = validate_model_spec(spec)
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    temp = path + ".tmp"
    with open(temp, "w", encoding="utf-8") as handle:
        json.dump(clean, handle, indent=2, ensure_ascii=False, sort_keys=False)
    os.replace(temp, path)
    return path


def load_model_spec(path: str = None):
    """
    Read a spec from disk. Returns (spec, problems).

    A missing file is not an error -- it means "never edited", and the
    defaults are the right answer.
    """
    path = path or model_spec_path()
    if not os.path.exists(path):
        return default_model_spec(), []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        return default_model_spec(), [f"could not read {os.path.basename(path)}"
                                      f" ({exc}); using the defaults"]
    return validate_model_spec(payload)


# Which constructor arguments are worth showing per class. Printing every
# sklearn default would bury the three numbers that matter.
ESTIMATOR_KEY_PARAMS = {
    "LinearDiscriminantAnalysis": ("solver", "shrinkage"),
    "SVC": ("kernel", "C", "gamma", "class_weight"),
    "RandomForestClassifier": ("n_estimators", "max_depth", "min_samples_leaf",
                               "max_features", "class_weight"),
    "ExtraTreesClassifier": ("n_estimators", "max_depth", "min_samples_leaf",
                             "class_weight"),
    "HistGradientBoostingClassifier": ("learning_rate", "max_iter",
                                       "max_leaf_nodes", "l2_regularization",
                                       "early_stopping", "class_weight"),
    "LogisticRegression": ("penalty", "C", "l1_ratio", "solver", "max_iter",
                           "class_weight"),
    "KNeighborsClassifier": ("n_neighbors", "weights"),
    "MLPClassifier": ("hidden_layer_sizes", "alpha", "solver", "batch_size",
                      "learning_rate_init", "early_stopping", "max_iter"),
    "StandardScaler": (),
    "RobustScaler": (),
    "MinMaxScaler": (),
}


def _estimator_params(estimator) -> str:
    """The interesting constructor arguments of one estimator, as text."""
    name = type(estimator).__name__
    wanted = ESTIMATOR_KEY_PARAMS.get(name)
    try:
        params = estimator.get_params(deep=False)
    except (AttributeError, TypeError):                      # pragma: no cover
        return ""
    if wanted is None:
        # An unknown class: show everything short rather than nothing, so a
        # model trained by a future version is still readable here.
        wanted = tuple(sorted(params))
    bits = []
    for key in wanted:
        if key not in params:
            continue
        value = params[key]
        if value is None and key == "class_weight":
            continue                                 # the default, not news
        if isinstance(value, float):
            text = f"{value:g}"
        elif isinstance(value, (tuple, list)):
            text = "x".join(str(v) for v in value)
        else:
            text = str(value)
        bits.append(f"{key}={text}")
    return ", ".join(bits)


def describe_estimator(estimator, indent: int = 0, max_depth: int = 4):
    """
    The real structure of a fitted estimator, as lines of text.

    Read off the OBJECT, not off the spec that was meant to produce it: this
    is what will actually run on the next prediction, so if the two ever
    disagree, this is the truth. A Pipeline becomes its ordered stages, a
    VotingClassifier becomes its members plus how their outputs are combined,
    and anything else becomes one line.
    """
    pad = "    " * indent
    if estimator is None:
        return [f"{pad}(no model)"]
    if indent > max_depth:                                   # pragma: no cover
        return [f"{pad}..."]

    name = type(estimator).__name__
    lines = []

    steps = getattr(estimator, "steps", None)
    if steps is not None:                                    # a Pipeline
        lines.append(f"{pad}Pipeline  ({len(steps)} stages, in order)")
        for position, (step_name, step) in enumerate(steps, start=1):
            arrow = "->" if position > 1 else "  "
            lines.append(f"{pad}  {arrow} [{position}] {step_name}: "
                         f"{type(step).__name__}")
            detail = _estimator_params(step)
            if detail:
                lines.append(f"{pad}         {detail}")
            if getattr(step, "steps", None) or getattr(step, "estimators", None):
                lines.extend(describe_estimator(step, indent + 3, max_depth))
        return lines

    members = getattr(estimator, "estimators", None)
    if members is not None and name.startswith("Voting"):
        mode = getattr(estimator, "voting", "?")
        weights = getattr(estimator, "weights", None)
        lines.append(f"{pad}{name}   --   {len(members)} models, "
                     f"{mode} voting")
        if mode == "soft":
            lines.append(f"{pad}  combines: average of each model's "
                         "predicted probabilities, highest average wins")
        else:
            lines.append(f"{pad}  combines: one vote per model, "
                         "most votes wins (ties go to the first label)")
        lines.append(f"{pad}  weights : "
                     + ("equal" if not weights
                        else ", ".join(f"{w:g}" for w in weights)))
        for position, item in enumerate(members, start=1):
            member_name, member = (item if isinstance(item, tuple)
                                   else (f"model{position}", item))
            lines.append(f"{pad}  member {position}: {member_name}")
            lines.extend(describe_estimator(member, indent + 2, max_depth))
        return lines

    lines.append(f"{pad}{name}")
    detail = _estimator_params(estimator)
    if detail:
        lines.append(f"{pad}  {detail}")
    return lines


def _html_escape(text) -> str:
    return (str(text).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


def estimator_html(estimator, depth: int = 0) -> str:
    """
    A fitted estimator drawn as the chain of stages it actually is.

    The text version (describe_estimator) is exact but reads as a wall; this
    is the same information laid out so the SHAPE is visible at a glance --
    what goes in, what each stage does, and, for an ensemble, that several
    models run in parallel and how their answers are merged.

    Read off the object, like its text counterpart: this is what will run.
    """
    if estimator is None:
        return f'<p style="color:{COL_TEXT_DIM}">No model loaded.</p>'

    name = type(estimator).__name__

    steps = getattr(estimator, "steps", None)
    if steps is not None:
        rows = []
        for position, (step_name, step) in enumerate(steps, start=1):
            detail = _estimator_params(step)
            inner = ""
            if getattr(step, "steps", None) or getattr(step, "estimators", None):
                inner = estimator_html(step, depth + 1)
            rows.append(
                f'<tr>'
                f'<td style="padding:6px 10px;color:{COL_TEXT_DIM};'
                f'font-size:{FS_SM}px;white-space:nowrap">step {position}</td>'
                f'<td style="padding:6px 10px">'
                f'<b style="color:{COL_ACCENT}">{_html_escape(type(step).__name__)}</b>'
                f'<span style="color:{COL_TEXT_DIM}"> &nbsp;({_html_escape(step_name)})</span>'
                + (f'<br><span style="color:{COL_TEXT_DIM};font-size:{FS_SM}px">'
                   f'{_html_escape(detail)}</span>' if detail else "")
                + inner +
                f'</td></tr>'
            )
        return (f'<table width="100%" cellspacing="0" cellpadding="0" '
                f'style="background:{COL_PANEL};border:1px solid {COL_BORDER}">'
                + "".join(rows) + '</table>')

    members = getattr(estimator, "estimators", None)
    if members is not None and name.startswith("Voting"):
        mode = getattr(estimator, "voting", "?")
        weights = getattr(estimator, "weights", None)
        combine = ("every model outputs a probability per gesture; these are "
                   "averaged and the highest average wins"
                   if mode == "soft" else
                   "every model names one gesture; the most-named wins")
        cards = []
        for position, item in enumerate(members, start=1):
            member_name, member = (item if isinstance(item, tuple)
                                   else (f"model{position}", item))
            cards.append(
                f'<div style="margin:6px 0;padding:8px;'
                f'background:{COL_PANEL};border:1px solid {COL_BORDER}">'
                f'<div style="color:{COL_ACCENT};font-weight:700">'
                f'{position}. {_html_escape(member_name)}</div>'
                + estimator_html(member, depth + 1) + '</div>')
        return (
            f'<div style="padding:10px;background:{COL_PANEL_2};'
            f'border:1px solid {COL_ACCENT}">'
            f'<div style="font-size:{FS_LG}px;font-weight:700;color:{COL_TEXT}">'
            f'{len(members)} models vote &nbsp;&middot;&nbsp; '
            f'{_html_escape(mode)}</div>'
            f'<div style="color:{COL_TEXT_DIM};font-size:{FS_SM}px;'
            f'padding:4px 0 2px 0">How they are combined: {combine}.</div>'
            f'<div style="color:{COL_TEXT_DIM};font-size:{FS_SM}px">'
            f'Weights: {"equal" if not weights else _html_escape(weights)}</div>'
            + "".join(cards) + '</div>')

    detail = _estimator_params(estimator)
    return (f'<div style="padding:8px;background:{COL_PANEL};'
            f'border:1px solid {COL_BORDER}">'
            f'<b style="color:{COL_ACCENT}">{_html_escape(name)}</b>'
            + (f'<br><span style="color:{COL_TEXT_DIM};font-size:{FS_SM}px">'
               f'{_html_escape(detail)}</span>' if detail else "")
            + '</div>')


def model_report_html(predictor, stats: dict) -> str:
    """The whole 'what is this model' report, as one styled document."""
    stats = dict(stats or {})

    def section(title, body):
        return (f'<div style="margin:16px 0 6px 0;color:{COL_TEXT_DIM};'
                f'font-size:{FS_SM}px;font-weight:700;letter-spacing:1px">'
                f'{title}</div>{body}')

    def row(key, value):
        return (f'<tr><td style="padding:3px 12px 3px 0;color:{COL_TEXT_DIM};'
                f'white-space:nowrap">{_html_escape(key)}</td>'
                f'<td style="padding:3px 0;color:{COL_TEXT}">'
                f'{_html_escape(value)}</td></tr>')

    parts = [f'<body style="background:{COL_BG};color:{COL_TEXT};'
             f'font-family:Segoe UI;font-size:{FS_MD}px">']

    # -- what goes in ---------------------------------------------------------
    parts.append(section("WHAT GOES IN", (
        f'<table cellspacing="0">'
        + row("Window", f"{stats.get('window_s', 0) * 1000:.0f} ms of all "
                        f"{N_CHANNELS} channels")
        + row("Resampled to", f"{stats.get('fs', predictor.fs):g} Hz")
        + row("Features", f"{stats.get('n_features', '?')} "
                          f"({stats.get('feature_set', '?')} set, "
                          f"{stats.get('representation', '?')})")
        + row("Outputs", ", ".join(predictor.labels or
                                   stats.get("labels", []) or ["?"]))
        + '</table>')))

    # -- how it decides -------------------------------------------------------
    parts.append(section("HOW IT DECIDES", estimator_html(predictor.model)))

    # -- what it beat ---------------------------------------------------------
    spec = stats.get("model_spec")
    if isinstance(spec, dict):
        clean, _ = validate_model_spec(spec)
        vote = clean.get("voting") or {}
        auto = clean.get("auto") or {}
        candidates = [m for m in clean["models"] if m.get("enabled")]
        winner = stats.get("model_name")
        lines = []
        for model in candidates:
            is_winner = model["name"] == winner
            params = ", ".join(f"{k}={v}" for k, v in
                               sorted((model.get("params") or {}).items()))
            lines.append(
                f'<tr style="background:{COL_PANEL_2 if is_winner else "transparent"}">'
                f'<td style="padding:4px 10px;'
                f'color:{COL_GOOD if is_winner else COL_TEXT}">'
                + ("&#9733; " if is_winner else "&nbsp;&nbsp;&nbsp;")
                + f'{_html_escape(model["name"])}</td>'
                f'<td style="padding:4px 10px;color:{COL_TEXT_DIM};'
                f'font-size:{FS_SM}px">{_html_escape(params)}</td></tr>')
        parts.append(section(
            f"WHAT IT BEAT &nbsp;({len(candidates)} candidates, "
            f"same cross-validation)",
            f'<table width="100%" cellspacing="0">' + "".join(lines)
            + '</table>'
            + f'<div style="color:{COL_TEXT_DIM};font-size:{FS_SM}px;'
              f'padding-top:6px">Voting: '
            + (f'{vote.get("mode")}' if vote.get("enabled") else "off")
            + f' &nbsp;&middot;&nbsp; Auto mode: '
            + (f'on, up to {auto.get("max_per_model")} per model'
               if auto.get("enabled") else "off") + '</div>'))

        if auto.get("enabled"):
            parts.append(
                f'<div style="margin-top:10px;padding:10px;'
                f'border-left:3px solid {COL_WARN};background:{COL_PANEL}">'
                f'<b style="color:{COL_WARN}">Auto mode was on.</b> '
                f'<span style="color:{COL_TEXT_DIM}">The winner was chosen '
                f'from {len(candidates)} candidates scored on the same folds, '
                f'so the headline accuracy is optimistic for this model. The '
                f'leave-one-subject-out figure below did not get to pick its '
                f'own folds.</span></div>')

    parts.append('</body>')
    return "".join(parts)


def describe_model_spec(spec: dict) -> str:
    """One line for the GUI and the log."""
    clean, _ = validate_model_spec(spec)
    on = [m for m in clean["models"] if m["enabled"]]
    voting = clean["voting"]
    text = f"{len(on)} of {len(clean['models'])} models"
    auto = clean.get("auto") or {}
    if auto.get("enabled"):
        text += f" x up to {auto.get('max_per_model')} auto combos"
    if voting["enabled"] and len(on) > 1:
        text += f" + {voting['mode']} vote"
        if voting["always"]:
            text += " (always used)"
        else:
            text += " (competing)"
    return text


class Recording:
    """
    One labeled EMG recording held in memory.

    `emg` is the untouched device signal: (n_samples, 8) integer counts exactly
    as the armband reported them. Nothing in this class mutates it -- every
    derived view and every trim returns a new array or a new Recording.
    """

    def __init__(self, emg, timestamps, label, person_id, session_id,
                 repetition=0, stage=STAGE_RAW, uid=None, created_unix=None,
                 label_start_index=0, label_end_index=None, meta=None,
                 source=None):
        self.emg = np.ascontiguousarray(
            np.asarray(emg, dtype=np.int32).reshape(-1, N_CHANNELS)
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
            self.emg, self.fs, self.meta.get("emg_mode", EMG_MODE)
        )
        return self.activity

    def run_quality_checks(self, expected_seconds: float = None,
                           had_disconnect: bool = False):
        self.issues = check_quality(
            self.emg, self.timestamps, self.label,
            expected_seconds=expected_seconds,
            mode=self.meta.get("emg_mode", EMG_MODE),
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
        if self.label in AUTO_TRIM_SKIP_LABELS:
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
            "label_index": GESTURES.index(self.label) if self.label in GESTURES else -1,
            "person_id": self.person_id,
            "session_id": self.session_id,
            "repetition": self.repetition,
            "n_samples": self.n_samples,
            "n_channels": N_CHANNELS,
            "channel_names": list(CHANNEL_NAMES),
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
        if not HAVE_PANDAS:
            raise RuntimeError("pandas is required for to_dataframe()")
        data = {
            "sample_index": np.arange(self.n_samples, dtype=np.int64),
            "t_rel_s": self.t_rel.astype(np.float64),
            "t_unix": self.timestamps,
        }
        for c, name in enumerate(CHANNEL_NAMES):
            data[name.lower()] = self.emg[:, c]
        if CSV_INCLUDE_LABEL_COLUMNS:
            idx = np.arange(self.n_samples)
            data["in_label_window"] = (
                (idx >= self.label_start_index) & (idx < self.label_end_index)
            ).astype(np.int8)
            data["gesture"] = self.label
            data["person_id"] = self.person_id
            data["session_id"] = self.session_id
            data["repetition"] = self.repetition
        return _pandas().DataFrame(data)


# ------------------------------------------------------------------------------
# Writing
# ------------------------------------------------------------------------------


def csv_header():
    cols = ["sample_index", "t_rel_s", "t_unix"] + [c.lower() for c in CHANNEL_NAMES]
    if CSV_INCLUDE_LABEL_COLUMNS:
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
        writer.writerow(csv_header())
        tail = ([rec.label, rec.person_id, rec.session_id, rec.repetition]
                if CSV_INCLUDE_LABEL_COLUMNS else [])
        for i in range(rec.n_samples):
            row = [i, f"{t_rel[i]:.6f}", f"{rec.timestamps[i]:.6f}"]
            row.extend(int(v) for v in emg[i])
            if CSV_INCLUDE_LABEL_COLUMNS:
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
        channel_names=np.array(CHANNEL_NAMES),
        metadata_json=np.array(json.dumps(rec.metadata_dict(), default=str)),
    )
    return path


def write_recording_parquet(rec: Recording, path: str):
    if not HAVE_PANDAS:
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
        return ensure_dir(os.path.join(self.session_dir, FREE_TAKE_FOLDER))

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
            "handedness": HANDEDNESS,
            "arm_worn_on": ARM_WORN_ON,
            "device": DEVICE_NAME,
            "subject_notes": SUBJECT_NOTES,
            "channel_names": list(CHANNEL_NAMES),
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

        if SAVE_CSV:
            written["csv"] = write_recording_csv(rec, base + ".csv")
        if SAVE_NPZ:
            written["npz"] = write_recording_npz(rec, base + ".npz")
        if SAVE_PARQUET:
            parquet = write_recording_parquet(rec, base + ".parquet")
            if parquet:
                written["parquet"] = parquet

        for fmt, path in written.items():
            rec.files[f"{stage}:{fmt}"] = os.path.relpath(path, self.root)

        if SAVE_META_JSON:
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

        if SAVE_CSV:
            written["csv"] = write_recording_csv(rec, base + ".csv")
        if SAVE_NPZ:
            written["npz"] = write_recording_npz(rec, base + ".npz")
        if SAVE_PARQUET:
            parquet = write_recording_parquet(rec, base + ".parquet")
            if parquet:
                written["parquet"] = parquet

        for fmt, path in written.items():
            rec.files[f"{STAGE_TAKE}:{fmt}"] = os.path.relpath(path, self.root)

        if SAVE_META_JSON:
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
        folder = os.path.join(self.session_dir, FREE_TAKE_FOLDER)
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
        folder = os.path.join(self.session_dir, FREE_TAKE_FOLDER)
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
        return os.path.join(self.session_dir, MARKERS_NAME)

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


# ------------------------------------------------------------------------------
# Markers / notes
# ------------------------------------------------------------------------------
# Timestamped free text, kept apart from the signal it describes. Written
# append-only so an annotation can never damage a recording, and so two notes
# about the same moment both survive.

MARKER_KINDS = ("note", "break")


class Marker:
    """One timestamped annotation."""

    def __init__(self, t_unix: float, text: str, kind: str = "note",
                 uid: str = None, created_unix: float = None,
                 person_id: str = "", session_id: str = ""):
        self.t_unix = float(t_unix)
        self.text = str(text)
        self.kind = kind if kind in MARKER_KINDS else "note"
        self.uid = uid or uuid.uuid4().hex[:8]
        self.created_unix = float(
            created_unix if created_unix is not None else time.time())
        self.person_id = str(person_id)
        self.session_id = str(session_id)

    # -- display --------------------------------------------------------------

    @property
    def is_break(self) -> bool:
        return self.kind == "break"

    def short_text(self, limit: int = None) -> str:
        """The text, elided for drawing on the graph."""
        limit = MARKER_LABEL_MAX_CHARS if limit is None else limit
        text = " ".join(self.text.split())
        if not text:
            return "BREAK" if self.is_break else "note"
        return text if len(text) <= limit else text[: max(1, limit - 1)] + "\u2026"

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "kind": self.kind,
            "text": self.text,
            "t_unix": round(self.t_unix, 6),
            "t_iso": iso_time(self.t_unix),
            "created_unix": round(self.created_unix, 6),
            "created_iso": iso_time(self.created_unix),
            "person_id": self.person_id,
            "session_id": self.session_id,
        }

    @staticmethod
    def from_dict(entry: dict):
        return Marker(
            t_unix=entry.get("t_unix", 0.0),
            text=entry.get("text", ""),
            kind=entry.get("kind", "note"),
            uid=entry.get("uid"),
            created_unix=entry.get("created_unix"),
            person_id=entry.get("person_id", ""),
            session_id=entry.get("session_id", ""),
        )

    def __repr__(self) -> str:                               # pragma: no cover
        return (f"Marker({self.kind}, {iso_time(self.t_unix)}, "
                f"{self.short_text(18)!r})")


def append_marker_jsonl(path: str, marker: Marker) -> str:
    """Append one marker as a JSON line. The file is never rewritten."""
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(marker.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_markers(path: str):
    """Read a marker log, oldest first. A damaged line is skipped, not fatal."""
    markers = []
    if not path or not os.path.exists(path):
        return markers
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                markers.append(Marker.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue                                     # pragma: no cover
    markers.sort(key=lambda m: m.t_unix)
    return markers


def markers_between(markers, start_unix: float, end_unix: float):
    """The markers falling inside a time span, oldest first."""
    return [m for m in markers if start_unix <= m.t_unix <= end_unix]


def notes_sidecar_path(data_path: str) -> str:
    """
    Where the notes for a recording live:

        FIST_rep001_..._raw.csv  ->  FIST_rep001_..._raw_notes.json

    The stage suffix is kept, so notes follow the file they describe rather
    than being shared between a raw take and its trimmed copy.
    """
    stem, _ = os.path.splitext(data_path)
    return stem + NOTES_SIDECAR_SUFFIX


def load_recording_notes(data_path: str):
    """Notes attached to one recording, oldest first. Missing file = []."""
    path = notes_sidecar_path(data_path)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):                  # pragma: no cover
        return []
    entries = payload.get("notes", []) if isinstance(payload, dict) else payload
    markers = [Marker.from_dict(e) for e in entries if isinstance(e, dict)]
    markers.sort(key=lambda m: m.created_unix)
    return markers


def append_recording_note(data_path: str, marker: Marker,
                          rec_uid: str = "") -> str:
    """
    Add a note to a recording's sidecar.

    Read-modify-write of the *sidecar* only. The recording's own data file is
    never opened. Writing the whole sidecar each time is fine: it holds a
    handful of short strings, and it keeps the file valid JSON that any tool
    can read, unlike an append-only JSONL sidecar per recording. The write
    goes through a temp file and os.replace so an interrupted save cannot
    leave a half-written sidecar behind.
    """
    path = notes_sidecar_path(data_path)
    existing = load_recording_notes(data_path)
    existing.append(marker)
    payload = {
        "recording_file": os.path.basename(data_path),
        "recording_uid": rec_uid,
        "updated_iso": iso_time(time.time()),
        "notes": [m.to_dict() for m in existing],
    }
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return path


# ------------------------------------------------------------------------------
# Reading / ML-ready loading
# ------------------------------------------------------------------------------


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
    cols = [c.lower() for c in CHANNEL_NAMES]
    with open(path, "r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        header = reader.fieldnames or []
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
        emg=emg or np.zeros((0, N_CHANNELS)),
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
        if not HAVE_PANDAS:
            raise RuntimeError("pandas is required to read parquet recordings")
        frame = _pandas().read_parquet(path)
        label, person, session, repetition = _parse_path_identity(path)
        return Recording(
            emg=frame[[c.lower() for c in CHANNEL_NAMES]].to_numpy(),
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


def discover_recordings(root: str = None, persons=None, sessions=None,
                        gestures=None, stage: str = STAGE_RAW,
                        prefer: str = "npz"):
    """
    Walk the dataset tree and return the recording file paths that match.

    `stage` is "raw", "trimmed", "processed", "augmented" or
    "prefer_trimmed" (use the trimmed file when one exists, else the raw one).
    `prefer` picks the format when both .npz and .csv exist for a recording.
    """
    root = os.path.abspath(root or os.path.join(PROJECT_DIR, DATA_FOLDER))
    if not os.path.isdir(root):
        return []

    wanted_stages = ([STAGE_TRIMMED, STAGE_RAW] if stage == "prefer_trimmed"
                     else [stage])
    order = [prefer] + [f for f in ("npz", "csv", "parquet") if f != prefer]

    # uid -> {stage: {ext: path}}
    found = {}
    for dirpath, dirnames, filenames in os.walk(root):
        # Underscore folders are tooling, not data: _graphs holds exported
        # images and _deleted holds quarantined recordings, which must stay
        # recoverable on disk without ever re-entering the dataset.
        dirnames[:] = [d for d in dirnames if not d.startswith("_")]
        for name in filenames:
            ext = os.path.splitext(name)[1].lower().lstrip(".")
            if ext not in ("npz", "csv", "parquet") or name.endswith("_meta.json"):
                continue
            path = os.path.join(dirpath, name)
            label, person, session, _ = _parse_path_identity(path)
            if persons and person not in persons:
                continue
            if sessions and session not in sessions:
                continue
            if gestures and label not in gestures:
                continue
            file_stage = _stage_from_filename(path)
            stem = os.path.splitext(name)[0]
            # Group the formats of one recording together. The dedupe counter
            # stays part of the key, so a re-trimmed copy is its own entry
            # rather than silently shadowing the first one.
            suffix = STAGE_SUFFIX_RE.search(stem)
            base = stem[:suffix.start()] if suffix else stem
            counter = suffix.group(0)[len(file_stage) + 1:] if suffix else ""
            key = (person, session, label, base + counter)
            found.setdefault(key, {}).setdefault(file_stage, {})[ext] = path

    paths = []
    for key in sorted(found):
        for want in wanted_stages:
            by_ext = found[key].get(want)
            if not by_ext:
                continue
            for ext in order:
                if ext in by_ext:
                    paths.append(by_ext[ext])
                    break
            break
    return paths


def load_dataset(root: str = None, persons=None, sessions=None, gestures=None,
                 stage: str = STAGE_RAW, prefer: str = "npz",
                 skip_quality_errors: bool = False, verbose: bool = False):
    """
    Load whole recordings into memory.

    Returns List[Recording]. This is the entry point for a training script:

        recs = load_dataset("datasets", stage="prefer_trimmed")
        X, y, groups, meta = make_windows(recs)
    """
    paths = discover_recordings(root, persons, sessions, gestures, stage, prefer)
    recs = []
    for path in paths:
        try:
            rec = load_recording(path)
        except Exception as exc:
            print(f"[warn] could not load {path}: {exc}")
            continue
        if skip_quality_errors and rec.worst_issue_level == "error":
            if verbose:
                print(f"[skip] {path}: flagged with a quality error")
            continue
        recs.append(rec)
    if verbose:
        print(f"loaded {len(recs)} recordings from {len(paths)} files")
    return recs


def make_windows(recordings, window_ms: float = None, overlap: float = None,
                 fs: float = None, labels=None, use_label_window: bool = True,
                 representation: str = "raw", normalize: str = "mad"):
    """
    Turn recordings into a supervised windowed dataset.

    Returns (X, y, groups, meta) where
        X       (n_windows, window_samples, 8) float32
        y       (n_windows,) int   -- index into `labels`
        groups  (n_windows,) str   -- person_id, for subject-independent CV
        meta    dict with "labels", "window_samples", "step", "fs" and a
                per-window record list.

    `groups` is the important part: pass it to leave_one_person_out() or to
    scikit-learn GroupKFold so that windows from one person never straddle the
    train/test boundary. Random splitting of windows from the same person is
    the classic way to accidentally report 99 % accuracy that does not transfer.

    `representation`:
        "raw"       -> the device signal, untouched
        "normalized"-> per-recording robust normalisation (see normalize)
        "envelope"  -> moving-RMS envelope
    """
    recordings = list(recordings)
    if not recordings:
        return (np.empty((0, 0, N_CHANNELS), np.float32), np.empty(0, int),
                np.empty(0, object), {"labels": [], "window_samples": 0})

    labels = list(labels) if labels else sorted({r.label for r in recordings})
    label_to_index = {name: i for i, name in enumerate(labels)}

    fs = float(fs or np.median([r.fs for r in recordings]))
    win, step = window_params(fs, window_ms, overlap)

    xs, ys, gs, records = [], [], [], []
    for rec in recordings:
        if rec.label not in label_to_index:
            continue

        if representation == "envelope":
            signal = rec.envelope().astype(np.float32)
        elif representation == "normalized":
            signal = rec.normalized(normalize)[0].astype(np.float32)
        else:
            signal = rec.emg.astype(np.float32)

        if use_label_window and rec.label_end_index > rec.label_start_index:
            signal = signal[rec.label_start_index:rec.label_end_index]

        windows = sliding_windows(signal, win, step)
        if windows.shape[0] == 0:
            continue

        xs.append(np.ascontiguousarray(windows))
        ys.append(np.full(windows.shape[0], label_to_index[rec.label], dtype=int))
        gs.append(np.full(windows.shape[0], rec.person_id, dtype=object))
        records.extend({
            "uid": rec.uid,
            "person_id": rec.person_id,
            "session_id": rec.session_id,
            "label": rec.label,
            "repetition": rec.repetition,
            "window_index": w,
            "source": rec.source,
        } for w in range(windows.shape[0]))

    if not xs:
        return (np.empty((0, win, N_CHANNELS), np.float32), np.empty(0, int),
                np.empty(0, object), {"labels": labels, "window_samples": win})

    meta = {
        "labels": labels,
        "window_samples": win,
        "step": step,
        "fs": fs,
        "representation": representation,
        "records": records,
    }
    return (np.concatenate(xs), np.concatenate(ys), np.concatenate(gs), meta)


def make_feature_matrix(X, fs: float, include_freq: bool = True):
    """
    (n_windows, window, 8) -> (n_windows, n_features) for the classic models
    (Random Forest / SVM / XGBoost). Returns (features, names).
    """
    X = np.asarray(X)
    if X.shape[0] == 0:
        return np.empty((0, len(feature_names(include_freq))), np.float32), \
               feature_names(include_freq)
    feats = np.stack([extract_features(X[i], fs, include_freq)
                      for i in range(X.shape[0])])
    return feats, feature_names(include_freq)


def leave_one_person_out(groups):
    """
    Yield (train_index, test_index, held_out_person) for every person.

    This is the validation protocol that actually answers "will this work on
    somebody new?". Use it instead of a random split.
    """
    groups = np.asarray(groups, dtype=object)
    for person in sorted(set(groups.tolist())):
        test = np.flatnonzero(groups == person)
        train = np.flatnonzero(groups != person)
        if train.size and test.size:
            yield train, test, person


def split_by_person(groups, test_persons):
    """Explicit split, e.g. train on persons 1-3 and test on person 4."""
    groups = np.asarray(groups, dtype=object)
    test_persons = set(test_persons if isinstance(test_persons, (list, tuple, set))
                       else [test_persons])
    mask = np.isin(groups, list(test_persons))
    return np.flatnonzero(~mask), np.flatnonzero(mask)


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


def dataset_fingerprint(root: str = None, stage: str = STAGE_RAW) -> dict:
    """
    A compact, comparable description of what is on disk right now.

    `uid_digest` is the identity: two fingerprints with the same digest
    describe the same set of recordings, whatever order they were found in.
    """
    root = os.path.abspath(root or os.path.join(PROJECT_DIR, DATA_FOLDER))
    uids, labels, subjects, sessions = [], {}, set(), set()
    newest_stamp = ""

    for path in discover_recordings(root, stage=stage, prefer="npz"):
        label, person, session, _ = _parse_path_identity(path)
        parsed = parse_recording_name(path)
        if parsed is None:
            continue
        _, stamp, uid = parsed
        uids.append(uid)
        labels[label] = labels.get(label, 0) + 1
        subjects.add(person)
        sessions.add(f"{person}/{session}")
        if stamp > newest_stamp:        # the slug sorts chronologically
            newest_stamp = stamp

    digest = hashlib.sha1(
        "\n".join(sorted(uids)).encode("utf-8")).hexdigest()[:16]
    return {
        "root": root,
        "n_recordings": len(uids),
        "labels": labels,
        "subjects": sorted(subjects),
        "sessions": sorted(sessions),
        "newest_stamp": newest_stamp,
        "newest_unix": stamp_to_unix(newest_stamp) if newest_stamp else None,
        "uid_digest": digest,
    }


def count_recordings_after(stamp: str, root: str = None,
                           stage: str = STAGE_RAW) -> int:
    """How many recordings on disk are newer than `stamp`."""
    if not stamp:
        return 0
    root = os.path.abspath(root or os.path.join(PROJECT_DIR, DATA_FOLDER))
    newer = 0
    for path in discover_recordings(root, stage=stage, prefer="npz"):
        parsed = parse_recording_name(path)
        if parsed is not None and parsed[1] > stamp:
            newer += 1
    return newer


def describe_training_data(data: dict) -> str:
    """One line naming what a model was trained on."""
    if not data:
        return "not recorded"
    bits = [f"{data.get('n_recordings', 0)} recordings"]
    labels = data.get("labels") or {}
    if labels:
        bits.append(", ".join(f"{name} {count}"
                              for name, count in sorted(labels.items())))
    subjects = data.get("subjects") or []
    if subjects:
        bits.append(f"{len(subjects)} subject(s): {', '.join(subjects)}")
    sessions = data.get("sessions") or []
    if sessions:
        bits.append(f"{len(sessions)} session(s)")
    return "  |  ".join(bits)


def model_data_status(payload: dict, fingerprint: dict = None) -> dict:
    """
    Is there a model, what was it trained on, and is that still the dataset?

    Returns `state`, a `headline`, a `detail` and a `severity` the GUI maps
    to a colour. `state` is one of:

        "untrained"  nothing loaded
        "unknown"    a model, but it does not record its training data
        "current"    trained on exactly the recordings now on disk
        "stale"      the dataset has changed since it was trained
    """
    if not payload:
        return {"state": "untrained", "n_new": 0,
                "headline": "NO MODEL TRAINED",
                "detail": "Record some gestures, then press RETRAIN ON ALL "
                          "DATA. Until then the classifier stays disabled -- "
                          "it never invents a prediction.",
                "severity": "dim"}

    stats = payload.get("stats") or {}
    # `created_iso` is the sklearn payload's key, `created` the torch
    # checkpoint's. Both are asked the same question, so both are read.
    trained_at = (payload.get("created_iso") or payload.get("created")
                  or stats.get("created_iso") or "")
    when = f" on {trained_at[:16].replace('T', ' ')}" if trained_at else ""
    data = payload.get("data") or {}

    if not data.get("uid_digest"):
        return {"state": "unknown", "n_new": 0,
                "headline": f"TRAINED{when}",
                "detail": ("This model does not record which recordings it "
                           "was trained on, so whether it is up to date "
                           "cannot be answered. Retrain to find out.\n"
                           f"It reports: {stats.get('n_recordings', '?')} "
                           f"recordings, subjects "
                           f"{', '.join(stats.get('subjects') or []) or '?'}."),
                "severity": "warn"}

    trained_on = describe_training_data(data)
    if fingerprint is None:
        return {"state": "unknown", "n_new": 0,
                "headline": f"TRAINED{when}",
                "detail": f"Trained on {trained_on}.\n"
                          "The dataset on disk has not been checked.",
                "severity": "dim"}

    if fingerprint.get("uid_digest") == data.get("uid_digest"):
        return {"state": "current", "n_new": 0,
                "headline": f"UP TO DATE  --  trained{when}",
                "detail": f"Trained on {trained_on}.\n"
                          "Every recording now on disk went into this model.",
                "severity": "good"}

    n_new = count_recordings_after(data.get("newest_stamp", ""),
                                   fingerprint.get("root"))
    on_disk = fingerprint.get("n_recordings", 0)
    trained_n = data.get("n_recordings", 0)
    if n_new > 0:
        what = (f"{n_new} recording(s) have been added since this model was "
                f"trained.")
    elif on_disk < trained_n:
        what = (f"{trained_n - on_disk} recording(s) it was trained on are "
                f"no longer on disk.")
    else:
        what = "The set of recordings on disk is not the one it was trained on."
    return {"state": "stale", "n_new": n_new,
            "headline": f"OUT OF DATE  --  trained{when}",
            "detail": f"Trained on {trained_on}.\n{what}\n"
                      f"{on_disk} recording(s) on disk now. "
                      "RETRAIN ON ALL DATA to include them.",
            "severity": "warn"}


def dataset_summary(root: str = None):
    """Count recordings per person / session / gesture across the whole tree."""
    root = os.path.abspath(root or os.path.join(PROJECT_DIR, DATA_FOLDER))
    summary = {"root": root, "persons": {}, "total_raw": 0, "total_trimmed": 0}
    if not os.path.isdir(root):
        return summary

    for path in discover_recordings(root, stage=STAGE_RAW, prefer="npz"):
        label, person, session, _ = _parse_path_identity(path)
        person_entry = summary["persons"].setdefault(person, {})
        session_entry = person_entry.setdefault(session, {})
        session_entry[label] = session_entry.get(label, 0) + 1
        summary["total_raw"] += 1

    summary["total_trimmed"] = len(
        discover_recordings(root, stage=STAGE_TRIMMED, prefer="npz")
    )
    return summary


# ==============================================================================
# SECTION 5 -- MYO ACQUISITION (background thread)
#
# This section is deliberately Qt-free so that it can be tested headless.
#
# It does NOT reimplement the Myo protocol: it drives the existing, working
# `myo_raw.MyoRaw` wrapper exactly the way test_myo.py does (construct ->
# add_emg_handler -> connect -> run() in a loop), and adds around it:
#   * a bounded, thread-safe ring buffer for the live plot
#   * zero-loss capture straight out of the EMG callback
#   * stall detection and automatic reconnection with back-off
# ==============================================================================

from myo_raw import MyoRaw                      # the working communication layer

# Attribute handles of the Myo's four unfiltered-EMG characteristics. Only used
# when EMG_MODE == "raw"; the default "filtered" path never touches these.
RAW_EMG_DATA_ATTRS = (0x2b, 0x2e, 0x31, 0x34)
RAW_EMG_NOTIFY_ATTRS = (0x2c, 0x2f, 0x32, 0x35)

# Connection states, shown verbatim in the GUI.
CONN_DISCONNECTED = "DISCONNECTED"
CONN_CONNECTING = "CONNECTING"
CONN_CONNECTED = "CONNECTED"
CONN_RECONNECTING = "RECONNECTING"
CONN_ERROR = "ERROR"
CONN_STOPPED = "STOPPED"


class SampleRing:
    """
    Bounded thread-safe ring buffer for the live display.

    Bounded on purpose: a four-hour session must not grow memory. Recording
    data does NOT come from here (see Capture) -- losing a display sample is
    cosmetic, losing a recorded sample is not, so the two paths are separate.
    """

    def __init__(self, capacity: int):
        self.capacity = int(max(64, capacity))
        self._emg = np.zeros((self.capacity, N_CHANNELS), dtype=np.int32)
        self._t = np.zeros(self.capacity, dtype=np.float64)
        self._write = 0
        self._total = 0
        self._lock = threading.Lock()

    def push(self, t: float, values) -> None:
        with self._lock:
            i = self._write
            self._emg[i] = values
            self._t[i] = t
            self._write = (i + 1) % self.capacity
            self._total += 1

    def clear(self) -> None:
        with self._lock:
            self._write = 0
            self._total = 0
            self._emg[:] = 0
            self._t[:] = 0.0

    @property
    def total(self) -> int:
        with self._lock:
            return self._total

    def snapshot(self, n: int = None):
        """Last `n` samples in chronological order, as fresh copies."""
        with self._lock:
            have = min(self._total, self.capacity)
            if have == 0:
                return (np.zeros((0, N_CHANNELS), np.int32), np.zeros(0, np.float64))
            n = have if n is None else int(min(max(1, n), have))
            start = (self._write - n) % self.capacity
            if start + n <= self.capacity:
                return (self._emg[start:start + n].copy(), self._t[start:start + n].copy())
            tail = self.capacity - start
            emg = np.concatenate([self._emg[start:], self._emg[:n - tail]])
            t = np.concatenate([self._t[start:], self._t[:n - tail]])
            return emg, t

    def latest(self):
        """Most recent sample as (values, t), or (zeros, 0.0) if empty."""
        with self._lock:
            if self._total == 0:
                return np.zeros(N_CHANNELS, np.int32), 0.0
            i = (self._write - 1) % self.capacity
            return self._emg[i].copy(), float(self._t[i])


class Capture:
    """
    An in-progress recording.

    Samples are appended from inside the Myo callback, in the acquisition
    thread, so nothing can be dropped by a slow GUI frame or a blocked event
    loop. The GUI only ever reads the finished arrays.
    """

    def __init__(self, label: str, repetition: int, max_samples: int,
                 meta: dict = None):
        self.label = label
        self.repetition = int(repetition)
        self.max_samples = int(max_samples)
        self.meta = dict(meta or {})
        self.started_unix = time.time()
        self.finished_unix = None
        self.had_disconnect = False
        self.overflowed = False
        self.label_start_index = 0
        self.label_end_index = None
        self._emg = []
        self._t = []
        self._lock = threading.Lock()

    def append(self, t: float, values) -> None:
        with self._lock:
            if len(self._emg) >= self.max_samples:
                self.overflowed = True
                return
            self._emg.append(values)
            self._t.append(t)

    def mark_label_start(self) -> None:
        with self._lock:
            self.label_start_index = len(self._emg)

    def mark_label_end(self) -> None:
        with self._lock:
            self.label_end_index = len(self._emg)

    def __len__(self) -> int:
        with self._lock:
            return len(self._emg)

    def arrays(self):
        with self._lock:
            if not self._emg:
                return (np.zeros((0, N_CHANNELS), np.int32), np.zeros(0, np.float64))
            return (np.asarray(self._emg, dtype=np.int32),
                    np.asarray(self._t, dtype=np.float64))

    def to_recording(self, person_id: str, session_id: str,
                     extra_meta: dict = None) -> Recording:
        emg, t = self.arrays()
        meta = dict(self.meta)
        meta.update(extra_meta or {})
        meta.setdefault("emg_mode", EMG_MODE)
        meta.setdefault("capture_overflowed", self.overflowed)
        meta.setdefault("capture_had_disconnect", self.had_disconnect)

        end = self.label_end_index if self.label_end_index is not None else emg.shape[0]
        return Recording(
            emg=emg, timestamps=t, label=self.label,
            person_id=person_id, session_id=session_id,
            repetition=self.repetition, stage=STAGE_RAW,
            created_unix=self.started_unix,
            label_start_index=min(self.label_start_index, max(0, emg.shape[0] - 1)),
            label_end_index=max(1, min(end, emg.shape[0])) if emg.shape[0] else 0,
            meta=meta,
        )


class MyoAcquisition(threading.Thread):
    """
    Background Myo reader.

    Lifecycle:  start()  ->  (connect -> stream -> on error reconnect)*  ->  stop()

    Callbacks (invoked from THIS thread, keep them short):
        on_state(state, message)    connection state changed
        on_sample(t, values)        optional per-sample tap (live prediction)
        on_log(text)               human-readable log line
    """

    def __init__(self, port=None, mode: str = None, on_state=None,
                 on_sample=None, on_log=None):
        super().__init__(name="MyoAcquisition", daemon=True)
        self.port = port if port is not None else MYO_PORT
        self.mode = mode or EMG_MODE

        fs = nominal_rate(self.mode)
        self.ring = SampleRing(int(RING_BUFFER_SECONDS * fs * 1.5) + 1024)

        self._on_state = on_state or (lambda state, msg: None)
        self._on_sample = on_sample
        self._on_log = on_log or (lambda text: None)

        self._myo = None
        self._myo_lock = threading.Lock()
        self._capture = None
        self._capture_lock = threading.Lock()

        self._vibrate_request = None
        self._vibrate_lock = threading.Lock()

        self._running = threading.Event()
        self._state = CONN_DISCONNECTED
        self._state_message = ""

        # Timebase: monotonic deltas pinned to one wall-clock reading, so
        # timestamps are both absolute and free of clock-adjustment jumps.
        self._t0_unix = time.time()
        self._t0_perf = time.perf_counter()

        self._known_addr = None           # cached armband address
        self._connect_started = 0.0       # non-zero while a scan is in flight
        self._error_hint = ""             # friendly reason for the last failure
        self._last_sample_perf = 0.0
        self._sample_count = 0
        self._dropped_packets = 0
        self._rate_estimate = 0.0
        self._rate_window = deque(maxlen=64)
        self._reconnects = 0
        self._backlog_events = 0
        self._worst_backlog = 0
        self._stats_lock = threading.Lock()

    # -- public API -----------------------------------------------------------

    @property
    def state(self) -> str:
        return self._state

    @property
    def state_message(self) -> str:
        return self._state_message

    @property
    def connected(self) -> bool:
        return self._state == CONN_CONNECTED

    def stats(self) -> dict:
        # Read the port depth BEFORE taking the stats lock: nesting
        # _stats_lock around _myo_lock would be the only place in this class
        # that orders those two that way.
        backlog = self._input_backlog()
        with self._stats_lock:
            return {
                "samples": self._sample_count,
                "rate": self._rate_estimate,
                "dropped": self._dropped_packets,
                "reconnects": self._reconnects,
                "ring_fill": min(1.0, self.ring.total / self.ring.capacity),
                "backlog_events": self._backlog_events,
                "worst_backlog": self._worst_backlog,
                "backlog": backlog,
                "last_sample_age": (time.perf_counter() - self._last_sample_perf
                                    if self._last_sample_perf else None),
            }

    def _input_backlog(self) -> int:
        """Bytes waiting in the serial port right now (0 when not connected)."""
        with self._myo_lock:
            myo = self._myo
        if myo is None:
            return 0
        try:
            return int(myo.bt.ser.in_waiting)
        except Exception:
            return 0

    def unix_time(self) -> float:
        return self._t0_unix + (time.perf_counter() - self._t0_perf)

    def start_capture(self, label: str, repetition: int, meta: dict = None,
                      max_seconds: float = None) -> Capture:
        """
        Begin recording. Samples land in the returned Capture immediately.

        `max_seconds` overrides the MAX_RECORDING_SECONDS safety valve, which
        is sized for a cued take of a few seconds. A FREE TAKE passes its own,
        much larger cap; the 1.5x headroom stays, because the real rate runs
        above the nominal one often enough to matter.
        """
        seconds = float(max_seconds or MAX_RECORDING_SECONDS)
        capture = Capture(
            label, repetition,
            max_samples=int(seconds * nominal_rate(self.mode) * 1.5),
            meta=meta,
        )
        capture.had_disconnect = not self.connected
        with self._capture_lock:
            self._capture = capture
        return capture

    def stop_capture(self):
        """End the current recording and hand it back."""
        with self._capture_lock:
            capture = self._capture
            self._capture = None
        if capture is not None:
            capture.finished_unix = time.time()
        return capture

    @property
    def capturing(self) -> bool:
        with self._capture_lock:
            return self._capture is not None

    def vibrate(self, length: int = 1) -> None:
        """
        Request a haptic cue. Safe to call from the GUI thread.

        The write is NOT performed here. `BT` shares one serial port and one
        packet-assembly buffer between all callers and takes no lock while
        reading, so writing from a second thread desynchronises the byte
        stream: packets get misframed and EMG notifications start arriving as
        "data with unknown attr" garbage. The request is therefore handed to
        the acquisition thread, which performs it between two packets.
        """
        if not VIBRATE_CUES:
            return
        with self._vibrate_lock:
            self._vibrate_request = int(np.clip(length, 1, 3))

    def _service_vibrate(self, myo) -> None:
        """Perform a pending haptic cue. Called only from the acquisition thread."""
        with self._vibrate_lock:
            length = self._vibrate_request
            self._vibrate_request = None
        if length is None:
            return
        try:
            payload = struct.pack("<3B", 3, 1, length)
            if MYO_BULK_READ:
                # Fire and forget. myo_raw's write_attr blocks waiting for the
                # command response, but in bulk-read mode `_pump` has already
                # consumed and discarded it, so that wait would hang forever.
                self._write_attr_nowait(myo, 0x19, payload)
            else:
                myo.write_attr(0x19, payload)
        except Exception:
            pass          # a buzz is never worth losing the connection over

    @staticmethod
    def _write_attr_nowait(myo, attr: int, value: bytes) -> None:
        """
        Write a BLE attribute without waiting for the acknowledgement.

        Same framing as myo_raw's BT.write_attr/send_command -- BGAPI command
        class 4, id 5 -- just without the blocking read of the response.
        """
        if myo.conn is None:
            return
        payload = struct.pack("<BHB", myo.conn, attr, len(value)) + value
        myo.bt.ser.write(struct.pack("<4B", 0, len(payload), 4, 5) + payload)

    def stop(self) -> None:
        """
        Ask the thread to finish.

        Closing the serial port from here is intentional: MyoRaw.connect() can
        block in a bare serial read, and closing the handle turns that hang
        into an exception the loop already knows how to handle.
        """
        self._running.clear()
        with self._myo_lock:
            myo = self._myo
        if myo is not None:
            try:
                myo.bt.ser.close()
            except Exception:
                pass

    # -- thread body ----------------------------------------------------------

    def run(self) -> None:
        self._running.set()
        delay = RECONNECT_DELAY_S
        attempts = 0

        watchdog = threading.Thread(target=self._watchdog,
                                    name="MyoWatchdog", daemon=True)
        watchdog.start()

        while self._running.is_set():
            try:
                self._error_hint = ""
                self._connect_started = time.perf_counter()
                self._set_state(CONN_CONNECTING, "Opening the Myo dongle...")
                self._connect()
                self._connect_started = 0.0
                self._set_state(CONN_CONNECTED, "Streaming EMG.")
                delay = RECONNECT_DELAY_S
                attempts = 0
                self._stream()

            except Exception as exc:
                if not self._running.is_set():
                    break

                # A hint set by the watchdog explains the cause far better
                # than the serial error that closing the port produced.
                hint = self._error_hint
                self._log(hint or f"Myo error: {exc}")
                self._mark_capture_disconnect()
                self._teardown()

                if not AUTO_RECONNECT:
                    self._set_state(CONN_ERROR, hint or str(exc))
                    break

                attempts += 1
                if RECONNECT_MAX_ATTEMPTS and attempts > RECONNECT_MAX_ATTEMPTS:
                    self._set_state(
                        CONN_ERROR,
                        f"Giving up after {attempts - 1} reconnect attempts.",
                    )
                    break

                self._set_state(
                    CONN_RECONNECTING,
                    (f"{hint}  " if hint else "")
                    + f"Attempt {attempts} in {delay:.0f} s...",
                )
                self._sleep(delay)
                delay = min(delay * 1.5, RECONNECT_MAX_DELAY_S)

        self._teardown()
        watchdog.join(timeout=1.0)
        self._set_state(CONN_STOPPED, "Acquisition stopped.")

    # -- connection -----------------------------------------------------------

    def _apply_link_settings(self) -> None:
        """
        Push the BLE connection parameters into myo_raw before it connects.

        They live as module constants there so this can tune them without
        forking the connection logic.
        """
        import myo_raw as _mr
        interval = max(1, int(round(MYO_CONN_INTERVAL_MS / 1.25)))
        _mr.CONN_INTERVAL_MIN = interval
        _mr.CONN_INTERVAL_MAX = interval
        _mr.CONN_TIMEOUT = int(np.clip(round(MYO_CONN_TIMEOUT_MS / 10.0),
                                       10, 3200))
        _mr.CONN_LATENCY = 0

    def _connect(self) -> None:
        """Construct and connect a MyoRaw exactly the way the working code does."""
        self._apply_link_settings()
        myo = MyoRaw(self.port)
        myo.add_emg_handler(self._on_emg_filtered)

        if self.mode == "raw":
            self._install_raw_emg_support(myo)

        # Publish the instance BEFORE connect(). connect() blocks in the BLE
        # scan loop -- indefinitely if the armband is asleep -- and stop()
        # needs the serial handle to break that read. Publishing afterwards
        # left the thread stuck in the scan and the port open until the
        # process exited.
        with self._myo_lock:
            self._myo = myo

        # Reconnect straight to the address we already know, when we have one.
        # If that fails the address is forgotten so the next attempt scans.
        addr = self._known_addr if MYO_FAST_RECONNECT else None
        try:
            myo.connect(addr)
        except Exception:
            if addr is not None:
                self._known_addr = None
                self._log("Direct reconnect failed; will scan on the next try.")
            raise
        self._known_addr = getattr(myo, "addr", None)

        if MYO_NEVER_SLEEP:
            try:
                myo.set_sleep_mode(True)
                self._log("Sleep mode disabled (armband will not doze off).")
            except Exception as exc:
                self._log(f"Could not disable sleep mode: {exc}")

        if self.mode == "raw":
            self._enable_raw_emg(myo)

        self._last_sample_perf = time.perf_counter()
        self._log(
            f"Myo connected on {myo.bt.ser.port} (mode: {self.mode}, "
            f"supervision timeout {MYO_CONN_TIMEOUT_MS} ms"
            + (", direct" if addr is not None else ", after scan") + ")."
        )

    def _stream(self) -> None:
        """
        Pump packets until stopped or the link breaks.

        `myo.run()` is called with NO timeout argument, exactly as the
        known-good test_myo.py does, and this matters a great deal:

        `BT.recv_packet(timeout)` re-assigns `self.ser.timeout` before every
        single byte it reads, and on Windows each assignment reconfigures the
        COM port. With `run(1)` that is thousands of port reconfigurations per
        second -- measured on this machine it throttled a 50 Hz stream down to
        4.6 Hz and then failed outright with "A device attached to the system
        is not functioning". With no timeout, `recv_packet` sets the timeout
        once per packet and the stream runs at full rate.

        The cost is that this call blocks in a bare serial read when nothing
        arrives, so it cannot notice a stall by itself. `_watchdog` does that
        instead, and `stop()` closes the port to break out of the read.
        """
        with self._myo_lock:
            myo = self._myo
        if myo is None:
            raise RuntimeError("not connected")

        if MYO_FLUSH_ON_START:
            try:
                myo.bt.ser.reset_input_buffer()
            except Exception:
                pass

        if MYO_BULK_READ:
            # Set the timeout ONCE. myo_raw re-assigns it per packet, which is
            # the expensive part on Windows.
            try:
                myo.bt.ser.timeout = None
            except Exception:
                pass
            while self._running.is_set():
                self._pump(myo)
                self._service_vibrate(myo)
            return

        while self._running.is_set():
            myo.run()
            # Between packets is the only safe moment to write to the port.
            self._service_vibrate(myo)

    def _pump(self, myo) -> None:
        """
        Read everything waiting and parse it with myo_raw's own byte parser.

        Blocks on at least one byte so the loop does not spin. Command
        responses (typ 0) are dropped: during streaming nothing waits for
        them, and `_service_vibrate` writes fire-and-forget for that reason.
        """
        ser = myo.bt.ser
        bt = myo.bt

        waiting = ser.in_waiting
        if MYO_MAX_BACKLOG_BYTES and waiting > MYO_MAX_BACKLOG_BYTES:
            self._handle_backlog(ser, waiting)
            waiting = ser.in_waiting

        data = ser.read(waiting if waiting > 0 else 1)
        if not data:
            return

        for byte in data:
            packet = bt.proc_byte(byte)
            if packet is not None and packet.typ == 0x80:
                bt.handle_event(packet)

    def _handle_backlog(self, ser, waiting: int) -> None:
        """We are behind. Either catch up by discarding, or say so."""
        approx_seconds = waiting / max(1.0, 25.0 * nominal_rate(self.mode))
        with self._stats_lock:
            self._backlog_events += 1
            self._worst_backlog = max(self._worst_backlog, waiting)

        if self.capturing:
            self._log(
                f"Running {approx_seconds:.1f} s behind ({waiting} bytes "
                "queued) -- not flushing during a recording."
            )
            return

        try:
            ser.reset_input_buffer()
        except Exception:
            return
        self._log(
            f"Dropped a {approx_seconds:.1f} s backlog ({waiting} bytes) to "
            "stay live."
        )

    def _watchdog(self) -> None:
        """
        Notice a silent link.

        An armband going to sleep does not raise a serial error, it just stops
        sending. Closing the port from here turns the blocked read in
        `_stream` into an exception, which the reconnect loop already handles.
        """
        while self._running.is_set():
            time.sleep(0.25)

            # A scan that never finds anything: MyoRaw.connect() loops on
            # recv_packet() with no bound, so without this the thread sits
            # there forever and the UI only ever says "Opening the dongle".
            started = self._connect_started
            if SCAN_TIMEOUT_S and started and self._state == CONN_CONNECTING:
                if time.perf_counter() - started > SCAN_TIMEOUT_S:
                    self._connect_started = 0.0
                    self._error_hint = (
                        f"Armband did not answer in {SCAN_TIMEOUT_S:.0f} s -- "
                        "it is probably asleep. Pick it up, flex, or move it "
                        "to wake it."
                    )
                    self._log(self._error_hint)
                    self._force_close()
                    continue

            if not STALL_TIMEOUT_S or self._state != CONN_CONNECTED:
                continue
            last = self._last_sample_perf
            if not last:
                continue
            age = time.perf_counter() - last
            if age > STALL_TIMEOUT_S:
                self._log(
                    f"no EMG packet for {age:.1f} s -- the armband went to "
                    "sleep or the link dropped; forcing a reconnect"
                )
                # Debounce so the reconnect gets a chance before we fire again.
                self._last_sample_perf = time.perf_counter()
                self._force_close()

    def _force_close(self) -> None:
        with self._myo_lock:
            myo = self._myo
        if myo is None:
            return
        try:
            myo.bt.ser.close()
        except Exception:
            pass

    def _teardown(self) -> None:
        self._connect_started = 0.0
        with self._myo_lock:
            myo = self._myo
            self._myo = None
        if myo is None:
            return
        try:
            myo.disconnect()
        except Exception:
            pass
        try:
            myo.bt.ser.close()
        except Exception:
            pass

    def _mark_capture_disconnect(self) -> None:
        """Flag the in-flight recording instead of discarding it."""
        with self._capture_lock:
            if self._capture is not None:
                self._capture.had_disconnect = True

    # -- raw (unfiltered, 200 Hz) EMG ----------------------------------------
    #
    # The default "filtered" mode needs none of this. Raw mode is additive: it
    # registers an extra BT packet handler and enables four more notification
    # characteristics. The connection logic in myo_raw.py is untouched.

    def _install_raw_emg_support(self, myo) -> None:
        def handle_raw(packet):
            if (packet.cls, packet.cmd) != (4, 5):
                return
            try:
                _, attr, _ = struct.unpack("<BHB", packet.payload[:4])
            except struct.error:
                return
            if attr not in RAW_EMG_DATA_ATTRS:
                return
            payload = packet.payload[5:]
            if len(payload) < 16:
                return
            vals = struct.unpack("<16b", payload[:16])
            # Each notification carries two consecutive 8-channel samples.
            self._on_emg_raw(vals[0:8])
            self._on_emg_raw(vals[8:16])

        myo.bt.add_handler(handle_raw)

    def _enable_raw_emg(self, myo) -> None:
        for attr in RAW_EMG_NOTIFY_ATTRS:
            myo.write_attr(attr, b"\x01\x00")
        # set_mode(emg = send_emg_raw, imu = on, classifier = off)
        myo.write_attr(0x19, b"\x01\x03\x03\x01\x00")
        self._log("Unfiltered 200 Hz EMG enabled.")

    # -- sample path ----------------------------------------------------------

    def _on_emg_filtered(self, emg, moving) -> None:
        if self.mode == "filtered":
            self._ingest(emg)

    def _on_emg_raw(self, emg) -> None:
        self._ingest(emg)

    def _ingest(self, values) -> None:
        """
        The hot path: called once per sample, in the acquisition thread.

        Order matters. The capture is fed FIRST, because that is the data that
        must not be lost; display and statistics come after.
        """
        t = self.unix_time()
        values = tuple(int(v) for v in values)

        with self._capture_lock:
            capture = self._capture
        if capture is not None:
            capture.append(t, values)

        self.ring.push(t, values)

        now = time.perf_counter()
        with self._stats_lock:
            if self._last_sample_perf:
                dt = now - self._last_sample_perf
                if dt > 0:
                    self._rate_window.append(dt)
                    self._rate_estimate = 1.0 / (sum(self._rate_window)
                                                 / len(self._rate_window))
                expected = 1.0 / max(1.0, nominal_rate(self.mode))
                if dt > expected * 3.0:
                    self._dropped_packets += int(dt / expected) - 1
            self._last_sample_perf = now
            self._sample_count += 1

        if self._on_sample is not None:
            try:
                self._on_sample(t, values)
            except Exception:
                pass

    # -- plumbing -------------------------------------------------------------

    def _set_state(self, state: str, message: str = "") -> None:
        self._state = state
        self._state_message = message
        if state == CONN_RECONNECTING:
            with self._stats_lock:
                self._reconnects += 1
        try:
            self._on_state(state, message)
        except Exception:
            traceback.print_exc()

    def _log(self, text: str) -> None:
        print(f"[myo] {text}")
        try:
            self._on_log(text)
        except Exception:
            pass

    def _sleep(self, seconds: float) -> None:
        """Interruptible sleep -- stop() must not have to wait out a back-off."""
        end = time.perf_counter() + seconds
        while self._running.is_set() and time.perf_counter() < end:
            time.sleep(0.05)


# ==============================================================================
# SECTION 6 -- AUTOMATIC RECORDING SEQUENCE (state machine)
#
# Pure logic, no Qt and no timers of its own: the GUI calls tick() and the
# machine reports what should be on screen. That makes the whole recording
# protocol testable without an armband.
# ==============================================================================

STATE_IDLE = "IDLE"
STATE_REST = "REST"
STATE_PREPARE = "PREPARE"
STATE_COUNTDOWN = "COUNTDOWN"
STATE_RECORDING = "RECORDING"
STATE_STOPPING = "STOPPING"
STATE_REVIEW = "REVIEW"
STATE_SAVED = "SAVED"
STATE_PAUSED = "PAUSED"
STATE_DONE = "DONE"
STATE_ERROR = "ERROR"
STATE_RECONNECTING = "RECONNECTING"

# Colour per state, used by the big status banner.
STATE_COLORS = {
    STATE_IDLE: "#64748b",
    STATE_REST: "#38bdf8",
    STATE_PREPARE: "#fbbf24",
    STATE_COUNTDOWN: "#fb923c",
    STATE_RECORDING: "#ef4444",
    STATE_STOPPING: "#f97316",
    STATE_REVIEW: "#06b6d4",
    STATE_SAVED: "#22c55e",
    STATE_PAUSED: "#a78bfa",
    STATE_DONE: "#22c55e",
    STATE_ERROR: "#ef4444",
    STATE_RECONNECTING: "#f59e0b",
}


def build_plan(gestures=None, repetitions: int = None, order: str = None,
               seed: int = None):
    """
    Build the ordered list of (gesture, repetition) the session will collect.

    "interleaved" walks one repetition of every gesture before starting the
    next round, so fatigue and electrode drift are spread evenly across the
    label set instead of being baked into whichever gesture came last.
    """
    gestures = list(gestures if gestures is not None else GESTURES)
    repetitions = int(REPETITIONS if repetitions is None else repetitions)
    order = (order or GESTURE_ORDER).lower()
    rng = random.Random(RANDOM_SEED if seed is None else seed)

    if not gestures or repetitions <= 0:
        return []

    if order == "blocked":
        plan = [(g, r + 1) for g in gestures for r in range(repetitions)]
    elif order == "random":
        plan = [(g, r + 1) for g in gestures for r in range(repetitions)]
        rng.shuffle(plan)
    else:  # interleaved
        plan = []
        for r in range(repetitions):
            round_gestures = list(gestures)
            rng.shuffle(round_gestures)
            plan.extend((g, r + 1) for g in round_gestures)

    if MAX_RECORDINGS and len(plan) > MAX_RECORDINGS:
        plan = plan[:MAX_RECORDINGS]
    return plan


class SequenceController:
    """
    The REST -> GET READY -> 3 -> 2 -> 1 -> RECORD -> pause loop.

    Driven by repeated tick() calls (the GUI uses a 20 ms timer). All timing is
    wall-clock based, so a dropped frame delays nothing: the machine simply
    notices it is already past the deadline.

    Callbacks:
        on_capture_start(label, repetition) -> Capture
        on_capture_stop(capture, label, repetition)
        on_state(state, previous_state)
        on_finished()
    """

    def __init__(self, on_capture_start, on_capture_stop,
                 on_state=None, on_finished=None, on_capture_discard=None):
        self._on_capture_start = on_capture_start
        self._on_capture_stop = on_capture_stop
        self._on_state = on_state or (lambda state, previous: None)
        self._on_finished = on_finished or (lambda: None)
        self._on_capture_discard = on_capture_discard or (lambda cap, label, rep: None)

        self.plan = []
        self.index = 0
        self.state = STATE_IDLE
        self.state_entered = time.perf_counter()
        self.state_duration = 0.0
        self.paused = False
        self.pause_started = 0.0
        self.active_capture = None
        self.manual = False               # True = free-running manual recording
        self.last_message = ""

    # -- queries --------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.state not in (STATE_IDLE, STATE_DONE, STATE_ERROR)

    @property
    def current(self):
        """(label, repetition) of the item being collected, or None."""
        if 0 <= self.index < len(self.plan):
            return self.plan[self.index]
        return None

    @property
    def next_item(self):
        if 0 <= self.index + 1 < len(self.plan):
            return self.plan[self.index + 1]
        return None

    @property
    def progress(self):
        """(completed, total) over the whole plan."""
        return (min(self.index, len(self.plan)), len(self.plan))

    @property
    def elapsed(self) -> float:
        if self.paused:
            return self.pause_started - self.state_entered
        return time.perf_counter() - self.state_entered

    @property
    def remaining(self) -> float:
        if not self.state_duration:
            return 0.0
        return max(0.0, self.state_duration - self.elapsed)

    def banner(self):
        """
        (big_text, small_text) for the status banner -- the thing the subject
        actually looks at. Deliberately blunt and high-contrast.
        """
        item = self.current
        label = item[0] if item else ""

        if self.state == STATE_IDLE:
            return ("READY", "Press START SESSION")
        if self.state == STATE_PAUSED:
            return ("PAUSED", self.last_message or "Press RESUME to continue")
        if self.state == STATE_DONE:
            return ("SESSION COMPLETE", f"{self.progress[1]} recordings collected")
        if self.state == STATE_ERROR:
            return ("ERROR", self.last_message)
        if self.state == STATE_REST:
            return ("RELAX", f"Next: {label}")
        if self.state == STATE_PREPARE:
            return (f"GET READY: {label}", "Hold the start position")
        if self.state == STATE_COUNTDOWN:
            ticks = max(1, int(math.ceil(self.remaining)))
            return (str(ticks), f"{label} in {ticks}")
        if self.state == STATE_RECORDING:
            return ("RECORD NOW", label)
        if self.state == STATE_STOPPING:
            return ("STOP", label)
        if self.state == STATE_REVIEW:
            done, total = self.progress
            return (f"CHECK: {label}",
                    f"{done + 1} / {total}   --   ACCEPT & NEXT (Space) "
                    f"or REDO (Ctrl+R)")
        if self.state == STATE_SAVED:
            done, total = self.progress
            return ("SAVED", f"Repetition {done} / {total}")
        return (self.state, self.last_message)

    # -- control --------------------------------------------------------------

    def start(self, plan=None, start_index: int = 0) -> None:
        self.plan = list(plan) if plan is not None else build_plan()
        self.index = int(np.clip(start_index, 0, max(0, len(self.plan))))
        self.manual = False
        self.paused = False
        if not self.plan:
            self._enter(STATE_DONE, 0.0)
            return
        self._enter(STATE_REST, REST_TIME)

    def start_manual(self, label: str, repetition: int = 0) -> None:
        """
        Free-running single recording: starts immediately and runs until
        stop_manual(). Used for the manual RECORD button.
        """
        self.plan = [(label, repetition)]
        self.index = 0
        self.manual = True
        self.paused = False
        self._begin_capture()
        if self.active_capture is not None:
            self.active_capture.mark_label_start()
        self._enter(STATE_RECORDING, 0.0)       # 0 = no automatic end

    def stop_manual(self) -> None:
        if self.manual and self.state == STATE_RECORDING:
            self._finish_capture()
            if REVIEW_EACH_RECORDING:
                self._enter(STATE_REVIEW, 0.0)
            else:
                self.index += 1
                self._enter(STATE_SAVED, 0.6)

    # -- review step ----------------------------------------------------------

    @property
    def reviewing(self) -> bool:
        return self.state == STATE_REVIEW

    def accept_review(self) -> None:
        """Keep the take and move to the next repetition."""
        if self.state != STATE_REVIEW:
            return
        self.index += 1
        if self.manual:
            self._enter(STATE_IDLE, 0.0)
        elif self.index >= len(self.plan):
            self._enter(STATE_DONE, 0.0)
            self._on_finished()
        else:
            self._enter(STATE_REST, REST_TIME)

    def redo_review(self) -> None:
        """
        Reject the take and record the SAME repetition again.

        The index is not advanced, so the plan still owes you this gesture.
        Quarantining the rejected file is the caller's job -- this class never
        touches the filesystem.
        """
        if self.state != STATE_REVIEW:
            return
        if self.manual:
            self._enter(STATE_IDLE, 0.0)
            return
        self._enter(STATE_REST, REST_TIME)

    def pause(self, discard_in_flight: bool = False) -> None:
        """
        Pause between repetitions.

        A recording in flight is normally finished and handed over -- freezing
        mid-gesture would produce a truncated sample carrying a full label, and
        the operator pressing PAUSE usually did so between repetitions anyway.

        `discard_in_flight=True` is for an involuntary pause (the Myo link
        dropped). The partial recording is thrown away and the index is NOT
        advanced, so the same repetition is retried once the link is back.
        Otherwise a dropout would silently cost one sample of that gesture.
        """
        if not self.running or self.paused:
            return
        if self.state == STATE_REVIEW:
            return          # already waiting for the operator; nothing to pause

        if self.state in (STATE_RECORDING, STATE_STOPPING):
            if discard_in_flight:
                self._discard_capture()
            else:
                self._finish_capture()
                self.index += 1

        self.paused = True
        self.pause_started = time.perf_counter()
        self._enter(STATE_PAUSED, 0.0)

    def resume(self) -> None:
        if not self.paused:
            return
        self.paused = False
        self.last_message = ""
        if self.index >= len(self.plan):
            self._enter(STATE_DONE, 0.0)
        else:
            self._enter(STATE_REST, REST_TIME)

    def stop(self) -> None:
        """Abort the session. Any in-flight recording is still handed over."""
        if self.state in (STATE_RECORDING, STATE_STOPPING):
            self._finish_capture()
        self.paused = False
        self.active_capture = None
        self._enter(STATE_IDLE, 0.0)

    def skip(self) -> None:
        """Drop the current item and move to the next one."""
        if self.state in (STATE_RECORDING, STATE_STOPPING):
            self._finish_capture()
        self.index += 1
        if self.index >= len(self.plan):
            self._enter(STATE_DONE, 0.0)
        else:
            self._enter(STATE_REST, REST_TIME)

    # -- the clock ------------------------------------------------------------

    def tick(self) -> None:
        """
        Advance the machine. Safe (and cheap) to call at any rate.

        The capture window is deliberately WIDER than the labeled window:

            |<- PRE_GESTURE_MS ->|<---- RECORD_TIME ---->|<- POST_GESTURE_MS ->|
            capture opens         label window opens      label closes   capture closes

        So a subject who reacts a fraction early, or an explosive contraction
        that overshoots the cue, is still fully inside the raw file.
        """
        if self.paused or not self.running:
            return

        # Pre-roll: open the buffer before the countdown finishes.
        if self.state == STATE_COUNTDOWN and self.active_capture is None:
            if self.remaining <= PRE_GESTURE_MS / 1000.0:
                self._begin_capture()

        if self.state_duration <= 0.0:
            return                              # manual recording: no deadline
        if self.elapsed < self.state_duration:
            return

        if self.state == STATE_REST:
            self._enter(STATE_PREPARE, max(0.2, PREPARE_TIME))
        elif self.state == STATE_PREPARE:
            self._enter(STATE_COUNTDOWN, COUNTDOWN_TIME)
        elif self.state == STATE_COUNTDOWN:
            if self.active_capture is None:
                self._begin_capture()
            if self.active_capture is not None:
                self.active_capture.mark_label_start()
            self._enter(STATE_RECORDING, RECORD_TIME)
        elif self.state == STATE_RECORDING:
            # Label window closes here, but the buffer keeps running so the
            # relaxation phase of the gesture is preserved in the raw file.
            if self.active_capture is not None:
                self.active_capture.mark_label_end()
            self._enter(STATE_STOPPING, max(0.05, POST_GESTURE_MS / 1000.0))
        elif self.state == STATE_STOPPING:
            self._finish_capture()
            if REVIEW_EACH_RECORDING:
                # Hand control over. Duration 0 = no deadline: the machine
                # sits here until accept_review() or redo_review() is called.
                self._enter(STATE_REVIEW, 0.0)
            else:
                self.index += 1
                self._enter(STATE_SAVED, max(0.3, PAUSE_TIME))
        elif self.state == STATE_SAVED:
            if self.index >= len(self.plan):
                self._enter(STATE_DONE, 0.0)
                self._on_finished()
            elif AUTO_ADVANCE:
                self._enter(STATE_REST, REST_TIME)
            else:
                self.pause()

    # -- internals ------------------------------------------------------------

    def _enter(self, state: str, duration: float) -> None:
        previous = self.state
        self.state = state
        self.state_duration = float(duration)
        self.state_entered = time.perf_counter()
        if previous != state:
            self._on_state(state, previous)

    def _begin_capture(self) -> None:
        item = self.current
        if item is None:
            return
        label, repetition = item
        self.active_capture = self._on_capture_start(label, repetition)

    def _finish_capture(self) -> None:
        """
        Close the capture and hand it over.

        If the label window was never closed (abort, pause, manual stop) it is
        closed here, so a recording can never carry an open-ended label.
        """
        item = self.current
        capture = self.active_capture
        self.active_capture = None
        if capture is None or item is None:
            return
        if capture.label_end_index is None:
            capture.mark_label_end()
        self._on_capture_stop(capture, item[0], item[1])

    def _discard_capture(self) -> None:
        """Abandon an incomplete recording without saving it."""
        item = self.current
        capture = self.active_capture
        self.active_capture = None
        if capture is None or item is None:
            return
        self._on_capture_discard(capture, item[0], item[1])


# ==============================================================================
# SECTION 7 -- GUI
#
# Qt is imported lazily-ish (at module level, but guarded) so that every
# function above can be imported by a training script on a machine without a
# display. The acquisition thread never touches a widget: it emits through a
# QObject bridge and the GUI pulls from the ring buffer on its own timer.
# ==============================================================================

try:
    from PySide6 import QtCore, QtGui, QtSvg, QtWidgets
    import pyqtgraph as pg
    import pyqtgraph.exporters                       # noqa: F401  (registers exporters)
    HAVE_QT = True
except ImportError as _qt_import_error:               # pragma: no cover
    HAVE_QT = False
    QT_IMPORT_ERROR = _qt_import_error


# ------------------------------------------------------------------------------
# Theme
# ------------------------------------------------------------------------------

# ------------------------------------------------------------------------------
# DESIGN TOKENS
# ------------------------------------------------------------------------------
# Everything visual is derived from this block. Changing a number here changes
# it everywhere, which is the whole point: before this, paddings and font sizes
# were typed in at ~200 call sites and drifted apart.
#
# Spacing is an 8px rhythm with a 4px half-step. Nothing in the UI should use a
# gap that is not on this scale.
SP_XS = 4
SP_SM = 8
SP_MD = 12
SP_LG = 16
SP_XL = 24

# Type scale. Also a ratio, not a pile of arbitrary sizes.
FS_XS = 9         # hints, captions under a control
FS_SM = 10        # secondary labels, status lines
FS_MD = 11        # body: field labels, list rows
FS_LG = 12        # group headers
FS_XL = 14        # dialog titles
FS_HUGE = 18      # the one number that matters on a screen
FS_BANNER = 34    # the banner's instruction

# Corner radii.
RAD_SM = 4
RAD_MD = 8
RAD_LG = 10

# Control heights, so buttons in a row are never a pixel apart.
H_CONTROL = 30
H_ACTION = 38

COL_BG = "#0f1115"
COL_PANEL = "#171a21"
COL_PANEL_2 = "#1e222b"
COL_BORDER = "#2a2f3a"
COL_TEXT = "#e6e9ef"
COL_TEXT_DIM = "#9aa3b2"
COL_ACCENT = "#38bdf8"
COL_GOOD = "#22c55e"
COL_WARN = "#f59e0b"
COL_BAD = "#ef4444"

STYLESHEET = f"""
QWidget {{
    background: {COL_BG};
    color: {COL_TEXT};
    font-family: "Segoe UI", "Inter", sans-serif;
    font-size: 12px;
}}
QGroupBox {{
    background: {COL_PANEL};
    border: 1px solid {COL_BORDER};
    border-radius: 8px;
    margin-top: 14px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    color: {COL_TEXT_DIM};
    font-size: 11px;
    letter-spacing: 1px;
}}
QPushButton {{
    background: {COL_PANEL_2};
    border: 1px solid {COL_BORDER};
    border-radius: 6px;
    padding: 7px 12px;
    font-weight: 600;
}}
QPushButton:hover   {{ background: #272c37; border-color: #3a4150; }}
QPushButton:pressed {{ background: #121620; }}
QPushButton:disabled {{ color: #5a6170; background: #14171e; border-color: #22262f; }}
QPushButton#primary   {{ background: #0b5f86; border-color: {COL_ACCENT}; color: #eaf7ff; }}
QPushButton#primary:hover {{ background: #0d74a3; }}
QPushButton#danger    {{ background: #7f1d1d; border-color: #b91c1c; color: #fee2e2; }}
QPushButton#danger:hover  {{ background: #991b1b; }}
QPushButton#warn      {{ background: #78350f; border-color: #b45309; color: #fef3c7; }}
QPushButton#warn:hover    {{ background: #92400e; }}
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {{
    background: #0b0e13;
    border: 1px solid {COL_BORDER};
    border-radius: 5px;
    padding: 5px 6px;
    selection-background-color: {COL_ACCENT};
}}
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COL_ACCENT};
}}
QLineEdit:disabled, QComboBox:disabled {{ color: #656d7c; background: #0d1016; }}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {COL_PANEL_2};
    border: 1px solid {COL_BORDER};
    selection-background-color: #0b5f86;
}}
QCheckBox {{ spacing: 6px; }}
QCheckBox::indicator {{
    width: 14px; height: 14px; border-radius: 3px;
    border: 1px solid #3a4150; background: #0b0e13;
}}
QCheckBox::indicator:checked {{ background: {COL_ACCENT}; border-color: {COL_ACCENT}; }}
QTableWidget {{
    background: #0b0e13;
    border: 1px solid {COL_BORDER};
    border-radius: 6px;
    gridline-color: #20242d;
}}
QHeaderView::section {{
    background: {COL_PANEL_2};
    border: none;
    border-bottom: 1px solid {COL_BORDER};
    padding: 5px;
    font-weight: 600;
    color: {COL_TEXT_DIM};
}}
QTableWidget::item:selected {{ background: #0b5f86; }}
QProgressBar {{
    background: #0b0e13;
    border: 1px solid {COL_BORDER};
    border-radius: 5px;
    height: 10px;
    text-align: center;
}}
QProgressBar::chunk {{ background: {COL_ACCENT}; border-radius: 4px; }}
QStatusBar {{ background: {COL_PANEL}; border-top: 1px solid {COL_BORDER}; }}
QStatusBar::item {{ border: none; }}
QScrollArea {{ border: none; }}
QSplitter::handle {{ background: {COL_BORDER}; }}
QToolTip {{
    background: {COL_PANEL_2}; color: {COL_TEXT};
    border: 1px solid {COL_BORDER}; padding: 4px;
}}
"""


class WheelGuard(QtCore.QObject):
    """
    Stops the mouse wheel from changing a control it is merely passing over.

    Qt's default is that a wheel event over a combo box, spin box or slider
    changes its value. In a tall scrolling side panel that means scrolling
    past a control silently edits it -- the gesture, the window length, a
    hyperparameter -- and the operator never sees it happen. Here that would
    be written into recording metadata, so it is a data-integrity bug, not a
    nuisance.

    The rule: a wheel event is only accepted when the control has keyboard
    focus, i.e. the operator deliberately clicked into it first. Otherwise the
    event is passed up to the scroll area, which is what they meant.
    """

    GUARDED = ()          # filled in below, once QtWidgets is imported

    def eventFilter(self, obj, event):
        if event.type() == QtCore.QEvent.Wheel and isinstance(obj, self.GUARDED):
            if not obj.hasFocus():
                event.ignore()
                return True
        return False


def install_wheel_guard(app) -> "WheelGuard":
    """Apply the wheel guard to every control in the application."""
    WheelGuard.GUARDED = (
        QtWidgets.QComboBox, QtWidgets.QAbstractSpinBox,
        QtWidgets.QSlider,
    )
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    # Wheel-over-without-focus is now inert, so these need a click to change:
    # make that click actually give them focus.
    app._wheel_guard = guard
    return guard


# Fusion's default selection is a saturated magenta that appears nowhere else
# in this app. Selections are frequent here -- the recordings table, the model
# roster, the take history -- so they get the accent colour instead.
SELECTION_QSS = f"""
QListWidget::item:selected, QTreeWidget::item:selected {{
    background: {COL_ACCENT}33;
    color: {COL_TEXT};
    border-left: 3px solid {COL_ACCENT};
}}
QListWidget::item:hover, QTreeWidget::item:hover {{
    background: {COL_PANEL_2};
}}
QTableWidget::item:selected {{
    background: {COL_ACCENT}33;
    color: {COL_TEXT};
}}
QListWidget, QTreeWidget {{
    outline: none;
}}
QToolButton#sectionHeader {{
    background: {COL_PANEL_2};
    color: {COL_TEXT};
    border: 1px solid {COL_BORDER};
    border-radius: {RAD_SM}px;
    padding: {SP_XS}px {SP_SM}px;
    font-size: {FS_MD}px;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-align: left;
    margin-top: {SP_SM}px;
}}
QToolButton#sectionHeader:hover {{
    background: {COL_BORDER};
    border-color: {COL_ACCENT};
}}
QToolButton#sectionHeader:checked {{
    border-left: 3px solid {COL_ACCENT};
    background: {COL_BORDER};
}}

/* An open section is a framed card joined to its header, so it is obvious
   where it starts and stops. Before this the content just floated. */
QWidget#sectionBody {{
    background: {COL_PANEL};
    border: 1px solid {COL_BORDER};
    border-top: none;
    border-radius: 0 0 {RAD_SM}px {RAD_SM}px;
}}

/* Thin, flat scrollbars. The Fusion default is a chunky 3-D affair with
   arrow buttons at both ends, and there are a dozen of them on screen. */
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {COL_BORDER};
    border-radius: 5px;
    min-height: 28px;
    min-width: 28px;
}}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{
    background: {COL_TEXT_DIM};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    height: 0; width: 0; border: none; background: none;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: none;
}}
"""


def apply_theme(app) -> None:
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET + SELECTION_QSS)
    install_wheel_guard(app)
    pg.setConfigOptions(
        antialias=PLOT_ANTIALIAS,
        useOpenGL=PLOT_USE_OPENGL,
        background=COL_BG,
        foreground=COL_TEXT_DIM,
        imageAxisOrder="row-major",
    )


def _marker_caption(marker: dict) -> str:
    """Elided caption for a marker dict, mirroring Marker.short_text()."""
    text = " ".join(str(marker.get("text", "")).split())
    if not text:
        return "BREAK" if marker.get("kind") == "break" else "note"
    limit = MARKER_LABEL_MAX_CHARS
    return text if len(text) <= limit else text[: max(1, limit - 1)] + "\u2026"


def make_label(text: str, size: int = 12, bold: bool = False,
               color: str = None, align=None):
    label = QtWidgets.QLabel(text)
    font = label.font()
    font.setPointSize(size)
    font.setBold(bold)
    label.setFont(font)
    if color:
        label.setStyleSheet(f"color: {color}; background: transparent;")
    else:
        label.setStyleSheet("background: transparent;")
    if align is not None:
        label.setAlignment(align)
    return label


# ------------------------------------------------------------------------------
# Acquisition -> GUI bridge
# ------------------------------------------------------------------------------


class AcquisitionBridge(QtCore.QObject):
    """
    Turns worker-thread callbacks into Qt signals.

    Qt queues cross-thread signal emissions, so this is the one safe way to let
    the acquisition thread talk to widgets. It carries no data payload per
    sample -- the GUI reads samples from the ring buffer on its own timer,
    which keeps the event loop free no matter how fast the armband streams.
    """
    state_changed = QtCore.Signal(str, str)
    log_line = QtCore.Signal(str)


# ------------------------------------------------------------------------------
# The real-time plot
# ------------------------------------------------------------------------------


class EmgPlotWidget(QtWidgets.QWidget):
    """
    Eight stacked channel bands with a shared, linked time axis.

    Performance notes (this is the part that has to keep up with the armband):
      * one PlotDataItem per channel, updated with setData -- no re-creation
      * peak-mode auto downsampling, so a 30 s window still shows every spike
        instead of smoothing it away
      * clipToView, so off-screen samples cost nothing
      * x axis is seconds relative to now, which avoids large float64 values
        in the renderer
    """

    # Emitted while the operator drags the review trim band (start_s, end_s).
    trim_changed = QtCore.Signal(float, float)
    # Emitted while dragging the LABEL ON GRAPH selection (absolute unix times).
    selection_changed = QtCore.Signal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.setContentsMargins(4, 2, 8, 2)
        self.glw.ci.setSpacing(0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.glw)

        lo, hi = emg_range()
        self._signed = lo < 0
        self._y_limit = [max(1.0, (hi - lo) * 0.25)] * N_CHANNELS

        self.window_s = float(PLOT_WINDOW_S)
        self.autoscale = bool(PLOT_AUTOSCALE)
        self.manual_scale = float(max(1.0, (hi - lo) * 0.5))
        self.show_envelope = bool(SHOW_ENVELOPE)
        self.frozen = False

        self._frame = 0
        self._last_stats = (np.zeros(N_CHANNELS), np.zeros(N_CHANNELS),
                            np.zeros(N_CHANNELS), np.zeros(N_CHANNELS))

        # Title row: this doubles as the caption of an exported image.
        self.title_item = self.glw.addLabel(
            APP_NAME, row=0, col=0, size="10pt", color=COL_TEXT_DIM
        )

        # Review mode: the plot shows one finished recording instead of the
        # live stream, with a draggable band marking what will be kept.
        self.review_mode = False
        self._review_t = np.zeros(0, dtype=np.float32)
        self._syncing_trim = False

        # True once the operator has zoomed/panned by hand: the autoscale
        # then stops overwriting the axes every frame.
        self._user_zoomed = False

        # LABEL ON GRAPH state. Selections and marks live in absolute unix
        # time and are re-projected every frame (see _project_selection).
        self.label_mode = False
        self.selection_abs = None
        self.marks = []
        self._syncing_select = False
        self._t_now = None

        self.plots, self.curves, self.env_curves = [], [], []
        self.value_items, self.regions, self.trim_regions = [], [], []
        self.select_regions = []
        self.mark_pool = [[] for _ in range(LABEL_MAX_VISIBLE_MARKS)]
        self.mark_labels = []
        # Markers: a timestamped note or break, drawn as a vertical separator
        # across every channel plus one caption on the top plot. Same pooling
        # discipline as the labelled bands -- no scene items are created or
        # destroyed while streaming.
        self.markers = []
        self.marker_pool = [[] for _ in range(MARKER_MAX_VISIBLE)]
        self.marker_labels = []
        self._marker_slot_kind = [None] * MARKER_MAX_VISIBLE
        self._mark_slot_style = [None] * LABEL_MAX_VISIBLE_MARKS

        for i in range(N_CHANNELS):
            plot = self.glw.addPlot(row=i + 1, col=0)
            plot.showGrid(x=True, y=True, alpha=0.12)
            plot.setMouseEnabled(x=True, y=True)
            # The x axis is "seconds relative to now", so 0 is the present and
            # there is nothing to the right of it. Without a limit pyqtgraph
            # happily pans into the future and shows a blank canvas; and
            # nothing exists further back than the ring buffer either.
            self._apply_live_limits(plot)
            # Any mouse zoom/pan hands the axes to the operator until RESET.
            plot.sigRangeChangedManually.connect(self._on_manual_zoom)
            # Markers are anchored in absolute time, so they have to be
            # re-projected whenever the view moves -- not only when a new
            # frame of samples arrives.
            plot.sigRangeChanged.connect(self._on_range_changed)
            plot.setMenuEnabled(True)
            plot.hideButtons()
            plot.setLabel("left", CHANNEL_NAMES[i])
            axis = plot.getAxis("left")
            axis.setWidth(52)
            axis.setStyle(tickFont=QtGui.QFont("Consolas", 7))
            axis.setTextPen(pg.mkPen(CHANNEL_COLORS[i]))

            bottom = plot.getAxis("bottom")
            if i < N_CHANNELS - 1:
                bottom.setStyle(showValues=False)
                bottom.setHeight(2)
            else:
                plot.setLabel("bottom", "time", units="s")
                bottom.setStyle(tickFont=QtGui.QFont("Consolas", 7))

            if i > 0:
                plot.setXLink(self.plots[0])

            curve = plot.plot(pen=pg.mkPen(CHANNEL_COLORS[i], width=PLOT_LINE_WIDTH))
            curve.setDownsampling(auto=True, method="peak")
            curve.setClipToView(True)

            env = plot.plot(pen=pg.mkPen("#ffffff", width=1, style=QtCore.Qt.DashLine))
            env.setDownsampling(auto=True, method="peak")
            env.setClipToView(True)
            env.setVisible(self.show_envelope)

            # Recording span highlight.
            region = pg.LinearRegionItem(
                values=(0, 0), movable=False,
                brush=pg.mkBrush(239, 68, 68, 38),
                pen=pg.mkPen(239, 68, 68, 120),
            )
            region.setZValue(-20)
            region.setVisible(False)
            plot.addItem(region)

            # Review trim band: what will be kept. Draggable on every channel,
            # all eight kept in sync.
            trim = pg.LinearRegionItem(
                values=(0, 0), movable=True,
                brush=pg.mkBrush(34, 197, 94, 48),
                pen=pg.mkPen(34, 197, 94, 210),
            )
            trim.setZValue(-15)
            trim.setVisible(False)
            trim.sigRegionChanged.connect(self._on_trim_region_changed)
            plot.addItem(trim)

            # Labelled bands from LABEL ON GRAPH mode. Pooled: creating and
            # destroying scene items every frame is what makes a live plot
            # stutter.
            for slot in range(LABEL_MAX_VISIBLE_MARKS):
                band = pg.LinearRegionItem(values=(0, 0), movable=False)
                band.setZValue(-25)
                band.setVisible(False)
                plot.addItem(band)
                self.mark_pool[slot].append(band)

            # Marker separators: one vertical line per channel per slot.
            for slot in range(MARKER_MAX_VISIBLE):
                line = pg.InfiniteLine(pos=0, angle=90, movable=False)
                line.setZValue(-15)
                line.setVisible(False)
                plot.addItem(line)
                self.marker_pool[slot].append(line)

            # The selection box, draggable on every channel and kept in sync.
            select = pg.LinearRegionItem(
                values=(0, 0), movable=True,
                brush=pg.mkBrush(250, 250, 250, 40),
                pen=pg.mkPen("#ffffff", width=2),
            )
            select.setZValue(-5)
            select.setVisible(False)
            select.sigRegionChanged.connect(self._on_select_region_changed)
            plot.addItem(select)
            self.select_regions.append(select)

            value = pg.TextItem("", color=CHANNEL_COLORS[i], anchor=(1, 0))
            value.setFont(QtGui.QFont("Consolas", 8))
            plot.addItem(value)

            self.plots.append(plot)
            self.curves.append(curve)
            self.env_curves.append(env)
            self.regions.append(region)
            self.trim_regions.append(trim)
            self.value_items.append(value)

        # Mark captions sit on the top plot only, so they do not repeat eight
        # times down the screen.
        for _ in range(LABEL_MAX_VISIBLE_MARKS):
            text = pg.TextItem("", anchor=(0, 1))
            text.setFont(QtGui.QFont("Segoe UI", 8, QtGui.QFont.Bold))
            text.setVisible(False)
            self.plots[0].addItem(text)
            self.mark_labels.append(text)

        # Marker captions, anchored the opposite way to the gesture captions
        # so the two do not sit on top of each other at the same instant.
        for _ in range(MARKER_MAX_VISIBLE):
            caption = pg.TextItem("", anchor=(0, 0))
            caption.setFont(QtGui.QFont("Segoe UI", 8, QtGui.QFont.Bold))
            caption.setVisible(False)
            self.plots[0].addItem(caption)
            self.marker_labels.append(caption)

        self.plots[0].setXRange(-self.window_s, 0, padding=0)

        # Big centred overlay for the countdown. A QLabel on top of the canvas
        # is far cheaper than drawing text into the scene every frame.
        self.overlay = QtWidgets.QLabel("", self.glw)
        self.overlay.setAlignment(QtCore.Qt.AlignCenter)
        self.overlay.setStyleSheet(
            "color: rgba(255,255,255,210); background: transparent;"
            "font-size: 150px; font-weight: 800;"
        )
        self.overlay.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents)
        self.overlay.hide()

        # Prediction read-out, top-LEFT so it never collides with the
        # countdown badge in the top-right.
        self.prediction_overlay = QtWidgets.QLabel("", self.glw)
        self.prediction_overlay.setAlignment(QtCore.Qt.AlignLeft
                                             | QtCore.Qt.AlignTop)
        self.prediction_overlay.setAttribute(
            QtCore.Qt.WA_TransparentForMouseEvents)
        self.prediction_overlay.hide()

    # -- geometry -------------------------------------------------------------

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.overlay.setGeometry(self.glw.rect())
        self._place_prediction_overlay()

    def _place_prediction_overlay(self) -> None:
        rect = self.glw.rect()
        self.prediction_overlay.setGeometry(
            QtCore.QRect(rect.left() + 62, rect.top() + 4,
                         max(220, rect.width() // 2), 76))

    def set_prediction_overlay(self, text: str, color: str = None,
                               sub: str = "") -> None:
        """
        Draw the live prediction over the traces.

        Left-aligned in the top-left corner and deliberately compact: the
        point of watching the graph during a prediction is to see the signal
        that produced it.
        """
        if not text or not PREDICTION_OVERLAY:
            self.prediction_overlay.hide()
            return
        colour = color or COL_ACCENT
        body = (f"<div style='color:{colour}; font-size:30px;"
                f" font-weight:800; line-height:1.0'>{text}</div>")
        if sub:
            body += (f"<div style='color:rgba(230,233,239,170);"
                     f" font-size:11px'>{sub}</div>")
        self.prediction_overlay.setText(body)
        self.prediction_overlay.setStyleSheet("background: transparent;")
        self._place_prediction_overlay()
        self.prediction_overlay.show()
        self.prediction_overlay.raise_()

    # -- controls -------------------------------------------------------------

    def set_window_seconds(self, seconds: float) -> None:
        self.window_s = float(max(0.2, seconds))
        # Choosing a window is an explicit instruction, so it overrides a
        # previous manual zoom rather than being ignored by it.
        self._user_zoomed = False
        self.plots[0].setXRange(-self.window_s, 0, padding=0)

    def set_autoscale(self, enabled: bool) -> None:
        self.autoscale = bool(enabled)

    def set_manual_scale(self, value: float) -> None:
        self.manual_scale = float(max(1.0, value))

    def set_envelope_visible(self, enabled: bool) -> None:
        self.show_envelope = bool(enabled)
        for env in self.env_curves:
            env.setVisible(enabled)
            if not enabled:
                env.setData([], [])

    def set_channel_visible(self, index: int, visible: bool) -> None:
        self.plots[index].setVisible(bool(visible))

    def set_frozen(self, frozen: bool) -> None:
        self.frozen = bool(frozen)

    @staticmethod
    def _apply_live_limits(plot) -> None:
        """
        Confine panning to the live axis: nothing exists after "now", and
        nothing before the start of the ring buffer.
        """
        plot.setLimits(xMax=PLOT_FUTURE_MARGIN_S,
                       xMin=-float(RING_BUFFER_SECONDS) * 1.05)

    def _clear_limits(self) -> None:
        """Lift the live limits -- a reviewed take has its own time axis."""
        for plot in self.plots:
            plot.setLimits(xMin=None, xMax=None)

    def _restore_limits(self) -> None:
        for plot in self.plots:
            self._apply_live_limits(plot)

    def _on_manual_zoom(self, *_args) -> None:
        self._user_zoomed = True

    @property
    def user_zoomed(self) -> bool:
        """True while the operator's own zoom is in charge of the axes."""
        return self._user_zoomed

    def reset_view(self) -> None:
        self._user_zoomed = False
        for plot in self.plots:
            plot.enableAutoRange(enable=False)
        self.plots[0].setXRange(-self.window_s, 0, padding=0)
        lo, hi = emg_range()
        self._y_limit = [max(1.0, (hi - lo) * 0.25)] * N_CHANNELS

    def set_overlay(self, text: str, color: str = None) -> None:
        """
        Draw the cue on top of the graph.

        Placement follows COUNTDOWN_OVERLAY. The default "corner" exists
        because a 150 px digit in the middle of the canvas hides exactly the
        traces you are trying to watch while you perform the gesture -- the
        big version of the cue already lives in the banner above the graph.
        """
        if not text or COUNTDOWN_OVERLAY == "off":
            self.overlay.hide()
            return

        rect = self.glw.rect()
        if COUNTDOWN_OVERLAY == "center":
            size = 150 if len(text) <= 2 else max(34, int(240 / len(text)))
            align = QtCore.Qt.AlignCenter
            geometry = rect
        else:                                   # "corner"
            size = 56 if len(text) <= 2 else max(20, int(150 / len(text)))
            align = QtCore.Qt.AlignRight | QtCore.Qt.AlignTop
            width = min(rect.width(), max(140, size * max(2, len(text))))
            geometry = QtCore.QRect(
                rect.right() - width - 60, rect.top() + 6,
                width, int(size * 1.35),
            )

        self.overlay.setAlignment(align)
        self.overlay.setStyleSheet(
            f"color: {color or 'rgba(255,255,255,200)'}; background: transparent;"
            f"font-size: {size}px; font-weight: 800;"
        )
        self.overlay.setText(text)
        self.overlay.setGeometry(geometry)
        self.overlay.show()
        self.overlay.raise_()

    # -- label on graph -------------------------------------------------------
    #
    # Everything here is anchored in ABSOLUTE unix time and re-projected onto
    # the x axis every frame. The axis is "seconds relative to now", so a
    # selection stored in axis coordinates would slide along the signal as
    # time passed and end up labelling the wrong samples.

    def set_label_mode(self, enabled: bool) -> None:
        self.label_mode = bool(enabled)
        if not enabled:
            self.selection_abs = None
        for region in self.select_regions:
            region.setVisible(bool(enabled) and self.selection_abs is not None)

    def arm_selection(self, t_now: float, width_s: float = None) -> None:
        """Drop a fresh selection box at the right-hand edge of the graph."""
        width = float(width_s or LABEL_DEFAULT_SELECTION_S)
        self.selection_abs = (t_now - width, t_now)
        for region in self.select_regions:
            region.setVisible(True)
        self._project_selection(t_now)

    def clear_selection(self) -> None:
        self.selection_abs = None
        for region in self.select_regions:
            region.setVisible(False)

    def selection_bounds(self):
        """The selection as absolute (start, end) unix times, or None."""
        return self.selection_abs

    def add_mark(self, start_abs: float, end_abs: float, label: str,
                 uid: str = "") -> None:
        self.marks.append({
            "start": float(start_abs), "end": float(end_abs),
            "label": str(label), "uid": uid,
        })
        if len(self.marks) > 400:
            del self.marks[:200]

    def remove_last_mark(self):
        return self.marks.pop() if self.marks else None

    def clear_marks(self) -> None:
        self.marks.clear()
        self._mark_slot_style = [None] * LABEL_MAX_VISIBLE_MARKS
        for pool in self.mark_pool:
            for region in pool:
                region.setVisible(False)
        for text in self.mark_labels:
            text.setVisible(False)

    def _project_selection(self, t_now: float) -> None:
        if self.selection_abs is None:
            return
        a, b = self.selection_abs
        self._syncing_select = True
        try:
            for region in self.select_regions:
                region.setRegion((a - t_now, b - t_now))
        finally:
            self._syncing_select = False

    def _on_select_region_changed(self, region) -> None:
        """A drag happened: convert straight back to absolute time."""
        if self._syncing_select or self._t_now is None:
            return
        low, high = region.getRegion()
        a, b = sorted((float(low), float(high)))
        self.selection_abs = (a + self._t_now, b + self._t_now)
        self._project_selection(self._t_now)
        self.selection_changed.emit(*self.selection_abs)

    def _draw_marks(self, t_now: float) -> None:        # noqa: D401
        """
        Re-project the labelled bands, newest first, hiding the ones that have
        scrolled off the left edge.
        """
        window = self._visible_span()
        shown = 0
        for mark in reversed(self.marks):
            if shown >= LABEL_MAX_VISIBLE_MARKS:
                break
            start = mark["start"] - t_now
            end = mark["end"] - t_now
            if end < -window * 1.02:            # fully off-screen to the left
                continue
            colour = LABEL_COLORS.get(
                mark["label"], CHANNEL_COLORS[hash(mark["label"]) % N_CHANNELS])

            # Restyle only when this slot changes gesture. Re-applying a brush
            # and pen to 8 items every frame is pure waste, and
            # LinearRegionItem has no setPen -- the pen lives on its boundary
            # lines.
            if self._mark_slot_style[shown] != mark["label"]:
                self._mark_slot_style[shown] = mark["label"]
                rgb = QtGui.QColor(colour)
                brush = pg.mkBrush(rgb.red(), rgb.green(), rgb.blue(), 60)
                pen = pg.mkPen(colour, width=1)
                for region in self.mark_pool[shown]:
                    region.setBrush(brush)
                    for line in region.lines:
                        line.setPen(pen)
                    region.update()

            for i, region in enumerate(self.mark_pool[shown]):
                region.setRegion((start, end))
                region.setVisible(self.plots[i].isVisible())
            text = self.mark_labels[shown]
            text.setText(mark["label"])
            text.setColor(colour)
            text.setPos(start, 0)
            text.setVisible(True)
            shown += 1

        for index in range(shown, LABEL_MAX_VISIBLE_MARKS):
            for region in self.mark_pool[index]:
                region.setVisible(False)
            self.mark_labels[index].setVisible(False)

    # -- markers --------------------------------------------------------------

    def add_marker(self, t_abs: float, text: str, kind: str = "note",
                   uid: str = "") -> None:
        """Remember a marker for drawing. Does not write anything to disk."""
        self.markers.append({
            "t": float(t_abs), "text": str(text),
            "kind": kind if kind in MARKER_KINDS else "note", "uid": uid,
        })
        if len(self.markers) > 400:
            del self.markers[:200]

    def remove_last_marker(self):
        return self.markers.pop() if self.markers else None

    def clear_markers(self) -> None:
        self.markers.clear()
        self._marker_slot_kind = [None] * MARKER_MAX_VISIBLE
        for pool in self.marker_pool:
            for line in pool:
                line.setVisible(False)
        for caption in self.marker_labels:
            caption.setVisible(False)

    def _visible_span(self) -> float:
        """
        How far back the view currently reaches, in seconds.

        This is NOT self.window_s once the operator has zoomed: using the
        nominal window meant anything older than it was culled, so markers and
        bands vanished the moment you zoomed out to look for them.
        """
        span = self.window_s
        try:
            (x_lo, _x_hi), _ = self.plots[0].viewRange()
            span = max(span, abs(float(x_lo)))
        except Exception:                                    # pragma: no cover
            pass
        return span

    def _on_range_changed(self, *_args) -> None:
        """Keep absolute-time items glued to the signal while the view moves."""
        if self._t_now:
            self._draw_markers(float(self._t_now))
            self._draw_marks(float(self._t_now))

    def _draw_markers(self, t_now: float) -> None:
        """
        Re-project the marker separators, newest first, hiding the ones that
        have scrolled off the left edge.
        """
        window = self._visible_span()
        shown = 0
        for marker in reversed(self.markers):
            if shown >= MARKER_MAX_VISIBLE:
                break
            x = marker["t"] - t_now
            if x < -window * 1.02:              # scrolled off to the left
                continue

            kind = marker["kind"]
            colour = MARKER_COLORS.get(kind, MARKER_COLORS["note"])

            # Restyle a slot only when its kind changes, for the same reason
            # the labelled bands do: re-penning 8 items every frame is waste.
            if self._marker_slot_kind[shown] != kind:
                self._marker_slot_kind[shown] = kind
                pen = pg.mkPen(
                    colour,
                    width=2 if kind == "break" else 1,
                    style=QtCore.Qt.DashLine if kind == "break"
                    else QtCore.Qt.SolidLine,
                )
                for line in self.marker_pool[shown]:
                    line.setPen(pen)

            for i, line in enumerate(self.marker_pool[shown]):
                line.setPos(x)
                line.setVisible(self.plots[i].isVisible())

            caption = self.marker_labels[shown]
            prefix = "| " if kind == "break" else ""
            caption.setText(prefix + _marker_caption(marker))
            caption.setColor(colour)
            caption.setPos(x, 0)
            caption.setVisible(True)
            shown += 1

        for index in range(shown, MARKER_MAX_VISIBLE):
            for line in self.marker_pool[index]:
                line.setVisible(False)
            self.marker_labels[index].setVisible(False)

    # -- review mode ----------------------------------------------------------

    def review(self, rec, activity=None) -> Tuple[float, float]:
        """
        Freeze the graph on a finished recording and draw the proposed cut.

        Returns the (start_s, end_s) the band was initialised to. The live
        stream keeps filling the ring buffer underneath; nothing is lost.
        """
        self.review_mode = True
        self.overlay.hide()
        # A take is shown on its own 0..duration axis, which the live limits
        # would forbid.
        self._clear_limits()

        t = rec.t_rel.astype(np.float32)
        signal = rec.emg.astype(np.float32)
        self._review_t = t

        if t.size == 0:
            return (0.0, 0.0)

        env = envelope(signal, max(1.0, rec.fs)) if self.show_envelope else None

        for i in range(N_CHANNELS):
            self.curves[i].setData(t, signal[:, i])
            if env is not None:
                self.env_curves[i].setData(t, env[:, i])
            self.regions[i].setVisible(False)

            span = max(float(np.abs(signal[:, i]).max()), 8.0) * PLOT_Y_PADDING
            self.plots[i].setYRange(-span if self._signed else 0, span, padding=0)

        self.plots[0].setXRange(float(t[0]), float(t[-1]), padding=0.02)

        if activity is not None and getattr(activity, "found", False):
            start = int(np.clip(activity.start, 0, t.size - 1))
            end = int(np.clip(activity.end - 1, start, t.size - 1))
            bounds = (float(t[start]), float(t[end]))
        else:
            bounds = (float(t[0]), float(t[-1]))

        self.set_trim_bounds(*bounds)
        for trim in self.trim_regions:
            trim.setVisible(bool(REVIEW_SHOW_TRIM))
        return bounds

    def exit_review(self) -> None:
        """Back to the live stream."""
        self.review_mode = False
        self._review_t = np.zeros(0, dtype=np.float32)
        for trim in self.trim_regions:
            trim.setVisible(False)
        # Back on the live axis, so the live pan limits apply again. Restore
        # them BEFORE setting the range, or the range would be clamped against
        # limits that are about to change.
        self._restore_limits()
        self.plots[0].setXRange(-self.window_s, 0, padding=0)

    def set_trim_bounds(self, start_s: float, end_s: float) -> None:
        self._syncing_trim = True
        try:
            for trim in self.trim_regions:
                trim.setRegion((float(start_s), float(end_s)))
        finally:
            self._syncing_trim = False

    def trim_bounds(self) -> Tuple[float, float]:
        low, high = self.trim_regions[0].getRegion()
        return (float(min(low, high)), float(max(low, high)))

    def trim_indices(self, n_samples: int) -> Tuple[int, int]:
        """Current trim band as [start, end) sample indices."""
        t = self._review_t
        if t.size == 0:
            return (0, n_samples)
        low, high = self.trim_bounds()
        start = int(np.clip(np.searchsorted(t, low), 0, max(0, t.size - 1)))
        end = int(np.clip(np.searchsorted(t, high) + 1, start + 1, t.size))
        return (start, min(end, n_samples))

    def _on_trim_region_changed(self, region) -> None:
        if self._syncing_trim:
            return
        low, high = region.getRegion()
        self.set_trim_bounds(low, high)
        self.trim_changed.emit(float(min(low, high)), float(max(low, high)))

    def set_title(self, text: str) -> None:
        self.title_item.setText(text, size="10pt", color=COL_TEXT_DIM)

    def set_recording_span(self, start_rel: float, end_rel: float,
                           visible: bool = True) -> None:
        for region in self.regions:
            region.setVisible(visible)
            if visible:
                region.setRegion((start_rel, end_rel))

    # -- the frame ------------------------------------------------------------

    def update_data(self, emg: np.ndarray, t: np.ndarray) -> None:
        """Push one frame of samples. Called from the GUI timer only."""
        if self.frozen or self.review_mode or emg.shape[0] == 0:
            return

        self._frame += 1
        now = t[-1]
        self._t_now = float(now)
        x = (t - now).astype(np.float32)

        # How far back the curves must reach. Using self.window_s alone meant
        # that zooming OUT past the window showed empty space -- the data was
        # in the ring buffer, it was just never handed to the curve. Follow
        # the visible range instead, so zooming out reveals real signal.
        span = self.window_s
        try:
            (x_lo, _x_hi), _ = self.plots[0].viewRange()
            span = max(span, abs(float(x_lo)))
        except Exception:                                    # pragma: no cover
            pass
        keep = x >= -span * 1.05
        if not keep.all():
            x = x[keep]
            emg = emg[keep]
        if x.size == 0:
            return

        signal = emg.astype(np.float32)
        env = (envelope(signal, max(1.0, measured_rate(t)))
               if self.show_envelope else None)

        mins = signal.min(axis=0)
        maxs = signal.max(axis=0)
        last = signal[-1]
        rms = np.sqrt(np.mean((signal - signal.mean(axis=0)) ** 2, axis=0))
        self._last_stats = (last, mins, maxs, rms)

        label_every = max(1, GUI_FPS // 8)

        for i in range(N_CHANNELS):
            plot = self.plots[i]
            if not plot.isVisible():
                continue

            self.curves[i].setData(x, signal[:, i])
            if env is not None:
                self.env_curves[i].setData(x, env[:, i])

            # Vertical range.
            if self.autoscale:
                span = max(abs(float(maxs[i])), abs(float(mins[i])))
                target = max(span * PLOT_Y_PADDING, 8.0)
                current = self._y_limit[i]
                # Grow instantly (never clip a peak), shrink slowly (no jitter).
                self._y_limit[i] = target if target > current else \
                    current * 0.93 + target * 0.07
                limit = self._y_limit[i]
            else:
                limit = self.manual_scale

            # Once the operator has zoomed by hand, leave the axes alone --
            # otherwise every frame snaps the view back and zooming is
            # impossible. RESET VIEW gives control back to the autoscale.
            if not self._user_zoomed:
                if self._signed:
                    plot.setYRange(-limit, limit, padding=0)
                else:
                    plot.setYRange(0, limit, padding=0)

            if i == 0 and self.mark_labels:
                # Keep the mark captions pinned near the top of the band.
                for text in self.mark_labels:
                    if text.isVisible():
                        text.setPos(text.pos().x(), limit * 0.95)

            if self._frame % label_every == 0:
                top = limit * 0.92 if not self._signed else limit * 0.8
                self.value_items[i].setPos(0, top)
                self.value_items[i].setText(
                    f"{int(last[i]):>5d}   min {int(mins[i]):>5d}  "
                    f"max {int(maxs[i]):>5d}  rms {rms[i]:6.1f}"
                )

        # Re-project the absolute-time overlays onto the moving axis.
        if self.marks:
            self._draw_marks(float(now))
            self._draw_markers(float(now))
        if self.selection_abs is not None:
            self._project_selection(float(now))

    def channel_stats(self):
        """(last, min, max, rms) arrays from the most recent frame."""
        return self._last_stats

    # -- export ---------------------------------------------------------------

    def export_image(self, path: str, width: int = 2400,
                     description: str = "") -> str:
        """
        Save the current graph as PNG (raster) or SVG (vector).

        SVG goes through Qt's own QSvgGenerator rather than PyQtGraph's
        SVGExporter: the latter crashes on these plots in pyqtgraph 0.14
        (ValueError in correctCoordinates while re-parsing path data).
        Rendering the scene with QPainter gives a true vector file and also
        lets the caption and metadata be embedded in the SVG header.
        """
        path = unique_path(path)
        ensure_dir(os.path.dirname(path))
        scene = self.glw.scene()
        rect = scene.sceneRect()

        if path.lower().endswith(".svg"):
            generator = QtSvg.QSvgGenerator()
            generator.setFileName(path)
            generator.setSize(QtCore.QSize(max(1, int(rect.width())),
                                           max(1, int(rect.height()))))
            generator.setViewBox(QtCore.QRectF(0, 0, rect.width(), rect.height()))
            generator.setTitle(str(self.title_item.text))
            generator.setDescription(description or str(self.title_item.text))
            painter = QtGui.QPainter(generator)
            try:
                scene.render(
                    painter,
                    QtCore.QRectF(0, 0, rect.width(), rect.height()),
                    rect,
                )
            finally:
                painter.end()
            return path

        exporter = pg.exporters.ImageExporter(scene)
        try:
            exporter.parameters()["width"] = int(width)
        except Exception:
            pass
        exporter.export(path)
        return path


# ------------------------------------------------------------------------------
# Status banner
# ------------------------------------------------------------------------------


def available_screen_size(widget=None):
    """
    The usable area of the screen a widget will appear on, in the same logical
    pixels Qt lays out in.

    Falls back to the primary screen, and then to a conservative 1280x720,
    because a headless/offscreen run has no screen at all and must not crash
    here.
    """
    screen = None
    if widget is not None:
        handle = widget.window().windowHandle() if widget.window() else None
        screen = handle.screen() if handle is not None else None
        if screen is None and widget.parent() is not None:
            screen = widget.parent().screen()
    if screen is None:
        screen = QtGui.QGuiApplication.primaryScreen()
    if screen is None:                                       # pragma: no cover
        return 1280, 720
    rect = screen.availableGeometry()
    return max(640, rect.width()), max(480, rect.height())


def fit_to_screen(widget, want_width: int, want_height: int,
                  min_width: int = None, min_height: int = None,
                  centre: bool = True):
    """
    Size a window to `want_*`, but never beyond the screen it opens on.

    Also clamps the MINIMUM size, which is the part that actually matters: a
    minimum larger than the screen cannot be resized away by the operator, so
    the window would stay off the edge of the display no matter what they did.

    Returns (width, height) actually applied.
    """
    avail_w, avail_h = available_screen_size(widget)
    cap_w = int(avail_w * WINDOW_FIT_FRACTION)
    cap_h = int(avail_h * WINDOW_FIT_FRACTION)

    width = max(320, min(int(want_width), cap_w))
    height = max(240, min(int(want_height), cap_h))

    if min_width is not None or min_height is not None:
        widget.setMinimumSize(
            max(320, min(int(min_width or 0), cap_w)),
            max(240, min(int(min_height or 0), cap_h)),
        )

    widget.resize(width, height)

    if centre:
        # Qt centres a dialog on its parent by itself, but a window that was
        # just clamped can still be positioned partly off-screen, so place it
        # explicitly inside the available area.
        screen = (widget.window().windowHandle().screen()
                  if widget.window() and widget.window().windowHandle()
                  else QtGui.QGuiApplication.primaryScreen())
        if screen is not None:
            rect = screen.availableGeometry()
            widget.move(rect.left() + max(0, (rect.width() - width) // 2),
                        rect.top() + max(0, (rect.height() - height) // 2))

    return width, height


class CollapsibleSection(QtWidgets.QWidget):
    """
    A panel with a header you can click to fold it away.

    The side panel stacks nine groups; on anything smaller than a big monitor
    they do not all fit, and scrolling past the one you want is the single
    most common irritation in this UI. Folding the ones you are not using is
    the cheapest fix that does not move anything around.

    The content widget is reparented, not rebuilt, so every `self.*` reference
    inside it keeps working and no behaviour changes.
    """

    toggled = QtCore.Signal(bool)

    def __init__(self, title: str, content: QtWidgets.QWidget,
                 expanded: bool = True, parent=None):
        super().__init__(parent)
        self._content = content

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header = QtWidgets.QToolButton()
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(bool(expanded))
        self.header.setToolButtonStyle(QtCore.Qt.ToolButtonTextBesideIcon)
        self.header.setArrowType(QtCore.Qt.DownArrow if expanded
                                 else QtCore.Qt.RightArrow)
        self.header.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                                  QtWidgets.QSizePolicy.Fixed)
        self.header.setCursor(QtCore.Qt.PointingHandCursor)
        self.header.setObjectName("sectionHeader")
        self.header.setMinimumHeight(26)
        self.header.toggled.connect(self._on_toggled)
        layout.addWidget(self.header)

        # The content sits in a named frame so the stylesheet can draw the
        # card that joins it to its header.
        body = QtWidgets.QWidget()
        body.setObjectName("sectionBody")
        body_layout = QtWidgets.QVBoxLayout(body)
        body_layout.setContentsMargins(SP_SM, SP_SM, SP_SM, SP_SM)
        body_layout.setSpacing(0)
        body_layout.addWidget(content)
        layout.addWidget(body)
        self._body = body

        body.setVisible(bool(expanded))

    def _on_toggled(self, expanded: bool) -> None:
        self.header.setArrowType(QtCore.Qt.DownArrow if expanded
                                 else QtCore.Qt.RightArrow)
        self._body.setVisible(expanded)
        self.toggled.emit(expanded)

    def set_expanded(self, expanded: bool) -> None:
        self.header.setChecked(bool(expanded))

    @property
    def expanded(self) -> bool:
        return self.header.isChecked()


def collapsible(group: QtWidgets.QGroupBox, expanded: bool = True):
    """
    Wrap an existing QGroupBox in a CollapsibleSection.

    The box's own title moves to the header, so there is one title rather than
    two, and the frame is dropped because the header already delimits the
    section.
    """
    title = group.title()
    group.setTitle("")
    group.setFlat(True)
    return CollapsibleSection(title, group, expanded)


class StatusBanner(QtWidgets.QFrame):
    """The big, unmissable "what should I do right now" panel."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(104)
        self.setFrameShape(QtWidgets.QFrame.NoFrame)

        self._color = STATE_COLORS[STATE_IDLE]

        self.big = make_label("READY", 34, True, "#ffffff",
                              QtCore.Qt.AlignCenter)
        self.small = make_label("Press START SESSION", 12, False,
                                "rgba(255,255,255,190)", QtCore.Qt.AlignCenter)

        self.bar = QtWidgets.QProgressBar()
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        self.bar.setRange(0, 1000)
        self.bar.setValue(0)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(16, 10, 16, 10)
        layout.setSpacing(2)
        layout.addWidget(self.big)
        layout.addWidget(self.small)
        layout.addWidget(self.bar)

        self._apply_color()

    def _apply_color(self) -> None:
        self.setStyleSheet(
            f"StatusBanner {{ background: {self._color};"
            f" border-radius: 10px; border: 1px solid rgba(255,255,255,40); }}"
            f"QProgressBar {{ background: rgba(0,0,0,70); border: none;"
            f" border-radius: 4px; }}"
            f"QProgressBar::chunk {{ background: rgba(255,255,255,220);"
            f" border-radius: 4px; }}"
        )

    def update_state(self, state: str, big: str, small: str,
                     fraction: float = None) -> None:
        color = STATE_COLORS.get(state, STATE_COLORS[STATE_IDLE])
        if color != self._color:
            self._color = color
            self._apply_color()
        if self.big.text() != big:
            self.big.setText(big)
        if self.small.text() != small:
            self.small.setText(small)
        self.bar.setVisible(fraction is not None)
        if fraction is not None:
            self.bar.setValue(int(np.clip(fraction, 0.0, 1.0) * 1000))


# ------------------------------------------------------------------------------
# Quality / trim dialogs
# ------------------------------------------------------------------------------


class QualityDialog(QtWidgets.QDialog):
    """
    Shown when a recording fails a check.

    There is no "delete" button on purpose: the raw file is already on disk by
    the time this appears, and the operator decides later whether to exclude it.
    """

    def __init__(self, rec: Recording, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Recording quality warning")
        fit_to_screen(self, 560, 420, min_width=420, min_height=240)

        layout = QtWidgets.QVBoxLayout(self)
        layout.addWidget(make_label(
            f"{rec.label}  rep {rec.repetition}  --  {rec.n_samples} samples, "
            f"{rec.duration:.2f} s", 13, True
        ))

        box = QtWidgets.QPlainTextEdit()
        box.setReadOnly(True)
        box.setPlainText("\n".join(
            f"{'!!' if i.level == 'error' else ' !' if i.level == 'warning' else '  '} "
            f"[{i.code}] {i.message}"
            for i in rec.issues
        ) or "No issues.")
        box.setMinimumHeight(140)
        layout.addWidget(box)

        layout.addWidget(make_label(
            "The raw recording has already been saved and is untouched. "
            "Use the recording list to review, trim or exclude it.",
            11, False, COL_TEXT_DIM
        ))

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        self.redo = QtWidgets.QPushButton("Repeat this gesture")
        self.redo.setObjectName("warn")
        keep = QtWidgets.QPushButton("Keep and continue")
        keep.setObjectName("primary")
        keep.setDefault(True)
        buttons.addWidget(self.redo)
        buttons.addWidget(keep)
        layout.addLayout(buttons)

        self._repeat = False
        self.redo.clicked.connect(self._on_repeat)
        keep.clicked.connect(self.accept)

    def _on_repeat(self) -> None:
        self._repeat = True
        self.accept()

    @property
    def repeat_requested(self) -> bool:
        return self._repeat


#  Per-class colours for the trail. Deliberately not CHANNEL_COLORS: those
#  mean "electrode 3", and reusing them for "the class FIST" would make two
#  unrelated things look related on the same screen.
CLASS_COLORS = ["#fb923c", "#38bdf8", "#a78bfa", "#34d399",
                "#f472b6", "#facc15", "#60a5fa", "#fb7185"]


def class_color(index: int) -> str:
    return CLASS_COLORS[index % len(CLASS_COLORS)]


class DecisionTrailWidget(QtWidgets.QWidget):
    """
    Every candidate's decision on one window, and how they fuse into one.

    Draws four columns -- the input, the candidates, the wires, the verdict --
    and nothing it was not given. With no step set it paints the reason why
    instead of an empty frame or a plausible-looking placeholder.

    It is a painter, not a source: `set_step()` is the only way anything gets
    on screen, and the caller is the one that decides whether the numbers are
    live or recorded. That keeps the "no fake predictions" rule checkable in
    one place.
    """

    ROW_H = 34
    PAD = 12
    IN_W = 148
    WIRE_W = 64
    OUT_W = 196

    def __init__(self, parent=None):
        super().__init__(parent)
        self.step: Optional[TrailStep] = None
        self.channels: Optional[np.ndarray] = None
        self.channel_peak = 1.0
        self.caption = ""
        self.empty_reason = "no decision panel loaded"
        self.cv: dict = {}
        self.setMinimumHeight(300)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.MinimumExpanding)

    def set_empty(self, reason: str) -> None:
        self.step = None
        self.empty_reason = reason
        self.update()

    def set_step(self, step: Optional[TrailStep], channels=None,
                 caption: str = "", cv: dict = None) -> None:
        self.step = step
        self.channels = None if channels is None else np.asarray(channels,
                                                                 dtype=float)
        if self.channels is not None and self.channels.size:
            self.channel_peak = max(1.0, float(np.max(self.channels)))
        self.caption = caption
        if cv is not None:
            self.cv = cv
        self.update()

    def sizeHint(self):
        rows = len(self.step.names) if self.step else 7
        return QtCore.QSize(720, 86 + rows * self.ROW_H + 60)

    # -- painting -------------------------------------------------------------

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        painter.fillRect(self.rect(), QtGui.QColor(COL_BG))

        if self.step is None or not self.step.names:
            self._paint_empty(painter)
            painter.end()
            return

        width, height = self.width(), self.height()
        #  Shed columns rather than squeezing them when the dialog is narrow:
        #  a 40 px wide probability bar communicates nothing.
        show_input = width >= 640 and self.channels is not None
        show_wires = width >= 520
        left = self.PAD + (self.IN_W + self.PAD if show_input else 0)
        right_w = min(self.OUT_W, max(150, int(width * 0.3)))
        wire_w = self.WIRE_W if show_wires else 0
        models_w = width - left - wire_w - right_w - self.PAD * 2
        if models_w < 120:
            models_w = max(100, width - left - right_w - self.PAD * 2)
            wire_w, show_wires = 0, False

        top = self.PAD + 22
        rows = len(self.step.names)

        self._paint_header(painter, width)
        if show_input:
            self._paint_input(painter, self.PAD, top, self.IN_W,
                              rows * self.ROW_H)
        centres = self._paint_models(painter, left, top, models_w)
        out_x = left + models_w + wire_w + self.PAD * 2
        fuse_top = top + max(0, (rows * self.ROW_H) // 2 - 58)
        if show_wires:
            self._paint_wires(painter, left + models_w + self.PAD, centres,
                              wire_w, out_x, fuse_top + 46)
        self._paint_fusion(painter, out_x, fuse_top, right_w)

        if self.caption:
            painter.setPen(QtGui.QColor(COL_TEXT_DIM))
            font = painter.font()
            font.setPointSize(8)
            font.setBold(False)
            painter.setFont(font)
            painter.drawText(
                QtCore.QRect(self.PAD, top + rows * self.ROW_H + 6,
                             max(60, width - self.PAD * 2), 34),
                QtCore.Qt.TextWordWrap | QtCore.Qt.AlignTop, self.caption)
        painter.end()

    def _paint_empty(self, painter):
        painter.setPen(QtGui.QColor(COL_TEXT_DIM))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(self.rect().adjusted(self.PAD, self.PAD,
                                              -self.PAD, -self.PAD),
                         QtCore.Qt.AlignCenter | QtCore.Qt.TextWordWrap,
                         self.empty_reason)

    def _paint_header(self, painter, width):
        step = self.step
        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        tag = "LIVE WINDOW" if step.source == "live" else "RECORDED WINDOW"
        colour = COL_GOOD if step.source == "live" else COL_ACCENT
        painter.setPen(QtGui.QColor(colour))
        painter.drawText(self.PAD, self.PAD + 10, tag)
        if step.note:
            painter.setPen(QtGui.QColor(COL_TEXT_DIM))
            font.setBold(False)
            painter.setFont(font)
            metrics = painter.fontMetrics()
            room = max(40, width - self.PAD * 2 - metrics.horizontalAdvance(tag) - 14)
            painter.drawText(
                self.PAD + metrics.horizontalAdvance(tag) + 14, self.PAD + 10,
                metrics.elidedText(step.note, QtCore.Qt.ElideRight, room))

    def _paint_input(self, painter, x, y, w, h):
        painter.setPen(QtGui.QPen(QtGui.QColor(COL_BORDER), 1))
        painter.setBrush(QtGui.QColor(COL_PANEL))
        painter.drawRoundedRect(QtCore.QRectF(x, y, w, h), RAD_SM, RAD_SM)

        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(COL_TEXT_DIM))
        painter.drawText(int(x + 8), int(y + 15), "INPUT  CHANNEL LEVEL")

        values = self.channels
        n = int(values.size)
        if not n:
            return
        bar_h = max(5, int((h - 32) / n) - 3)
        for i in range(n):
            row_y = y + 24 + i * (bar_h + 3)
            painter.setPen(QtGui.QColor(COL_TEXT_DIM))
            font.setBold(False)
            font.setPointSize(7)
            painter.setFont(font)
            painter.drawText(int(x + 8), int(row_y + bar_h - 1),
                             CHANNEL_NAMES[i] if i < len(CHANNEL_NAMES)
                             else f"CH{i + 1}")
            track = QtCore.QRectF(x + 34, row_y, w - 76, bar_h)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(COL_PANEL_2))
            painter.drawRect(track)
            frac = max(0.0, min(1.0, float(values[i]) / self.channel_peak))
            if frac > 0:
                painter.setBrush(QtGui.QColor(CHANNEL_COLORS[
                    i % len(CHANNEL_COLORS)]))
                painter.drawRect(QtCore.QRectF(track.x(), track.y(),
                                               track.width() * frac, bar_h))
            painter.setPen(QtGui.QColor(COL_TEXT))
            painter.drawText(
                QtCore.QRectF(x + w - 40, row_y - 1, 34, bar_h + 2),
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                f"{float(values[i]):.0f}")

    def _paint_models(self, painter, x, y, w):
        step = self.step
        centres = []
        target = step.decision(step.fused_from_mean)
        font = painter.font()

        for i, name in enumerate(step.names):
            probabilities = step.per_model[i]
            pick = step.decision(probabilities)
            confidence = float(probabilities[pick])
            row_y = y + i * self.ROW_H
            centres.append((row_y + self.ROW_H / 2 - 4, pick, confidence))

            #  A row that disagrees with the vote is the only thing worth
            #  highlighting here, so it is the only thing highlighted.
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(COL_PANEL if pick == target
                                          else COL_PANEL_2))
            painter.drawRoundedRect(
                QtCore.QRectF(x, row_y, w, self.ROW_H - 4), 3, 3)
            painter.setBrush(QtGui.QColor(class_color(pick)))
            painter.drawRect(QtCore.QRectF(x, row_y, 3, self.ROW_H - 4))

            font.setPointSize(9)
            font.setBold(name == step.deployed_name)
            painter.setFont(font)
            painter.setPen(QtGui.QColor(COL_TEXT))
            metrics = painter.fontMetrics()
            name_w = max(60, int(w * 0.34))
            shown = name + ("  *" if name == step.deployed_name else "")
            painter.drawText(
                QtCore.QRectF(x + 9, row_y, name_w, self.ROW_H - 4),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                metrics.elidedText(shown, QtCore.Qt.ElideRight, name_w))

            #  The probability split: one segment per class, to scale.
            bar_x = x + 9 + name_w + 6
            bar_w = max(40.0, w - (name_w + 150))
            bar = QtCore.QRectF(bar_x, row_y + self.ROW_H / 2 - 5, bar_w, 8)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(COL_BG))
            painter.drawRect(bar)
            offset = 0.0
            for k, p in enumerate(probabilities):
                seg = max(0.0, float(p)) * bar_w
                if seg <= 0.2:
                    offset += seg
                    continue
                painter.setBrush(QtGui.QColor(class_color(k)))
                painter.drawRect(QtCore.QRectF(bar.x() + offset, bar.y(),
                                               seg, bar.height()))
                offset += seg

            font.setPointSize(8)
            font.setBold(False)
            painter.setFont(font)
            painter.setPen(QtGui.QColor(class_color(pick)))
            painter.drawText(
                QtCore.QRectF(bar_x + bar_w + 6, row_y, 78, self.ROW_H - 4),
                QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
                f"{step.labels[pick] if pick < len(step.labels) else pick}")
            painter.setPen(QtGui.QColor(COL_TEXT_DIM))
            painter.drawText(
                QtCore.QRectF(x + w - 62, row_y, 56, self.ROW_H - 4),
                QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter,
                f"{confidence:.3f}")
        return centres

    def _paint_wires(self, painter, x, centres, w, out_x, out_y):
        """
        One curve per candidate into the vote.

        Width and opacity carry that candidate's confidence, and the colour
        carries its decision -- so a split vote is visible as two colours
        arriving at the same point, which is the thing worth seeing.
        """
        end_x = out_x - 2
        for y, pick, confidence in centres:
            path = QtGui.QPainterPath(QtCore.QPointF(x, y))
            mid = x + (end_x - x) * 0.55
            path.cubicTo(QtCore.QPointF(mid, y),
                         QtCore.QPointF(mid, out_y),
                         QtCore.QPointF(end_x, out_y))
            colour = QtGui.QColor(class_color(pick))
            colour.setAlpha(int(70 + 160 * max(0.0, min(1.0, confidence))))
            painter.setPen(QtGui.QPen(colour, 0.9 + 2.0 * confidence))
            painter.setBrush(QtCore.Qt.NoBrush)
            painter.drawPath(path)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(COL_TEXT))
        painter.drawEllipse(QtCore.QPointF(end_x, out_y), 3.0, 3.0)

    def _paint_fusion(self, painter, x, y, w):
        step = self.step
        fused = step.fused if step.fused is not None else step.fused_from_mean
        pick = step.decision(fused)
        confidence = float(fused[pick])
        height = 118

        painter.setPen(QtGui.QPen(QtGui.QColor(COL_TEXT), 1))
        painter.setBrush(QtGui.QColor(COL_PANEL))
        painter.drawRoundedRect(QtCore.QRectF(x, y, w, height),
                                RAD_SM, RAD_SM)

        font = painter.font()
        font.setPointSize(8)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(COL_TEXT_DIM))
        painter.drawText(int(x + 10), int(y + 15), "SOFT VOTE")

        font.setBold(False)
        font.setPointSize(7)
        painter.setFont(font)
        mean_pick = step.decision(step.fused_from_mean)
        painter.drawText(
            QtCore.QRectF(x + 10, y + 19, w - 20, 26),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop | QtCore.Qt.TextWordWrap,
            f"mean of {len(step.names)} = "
            f"{step.fused_from_mean[mean_pick]:.4f}"
            + ("" if step.fused is None else
               f"   ensemble {fused[pick]:.4f}"))

        font.setPointSize(17)
        font.setBold(True)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(class_color(pick)))
        painter.drawText(
            QtCore.QRectF(x + 10, y + 42, w - 20, 30),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            step.labels[pick] if pick < len(step.labels) else str(pick))

        font.setPointSize(8)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(COL_TEXT_DIM))
        painter.drawText(
            QtCore.QRectF(x + 10, y + 70, w - 20, 16),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignVCenter,
            f"{confidence:.3f}   {step.agreement} of {len(step.names)} agree")

        #  What actually runs on the robot, stated every time: a diagram of
        #  seven models fusing into one would otherwise imply the ensemble
        #  ships, and it does not.
        deployed_text = f"DEPLOYED  {step.deployed_name}"
        if step.deployed is not None:
            d_pick = step.decision(step.deployed)
            deployed_text += (f"  ->  "
                              f"{step.labels[d_pick] if d_pick < len(step.labels) else d_pick}"
                              f"  {step.deployed[d_pick]:.3f}")
        painter.setPen(QtGui.QColor(COL_WARN))
        font.setPointSize(7)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            QtCore.QRectF(x + 10, y + 88, w - 20, 26),
            QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop | QtCore.Qt.TextWordWrap,
            deployed_text)


class PanelWarmUp(QtCore.QRunnable):
    """
    Runs `DecisionPanel.warm_up()` on a worker thread.

    One call per app run, and it only touches fitted estimators read-only,
    so there is nothing to synchronise. The view polls `panel.warming` on
    the timer it already has rather than waiting on a signal -- one less
    cross-thread connection for a job whose only output is a flag.
    """

    def __init__(self, panel: "DecisionPanel"):
        super().__init__()
        self.panel = panel
        self.setAutoDelete(True)

    def run(self):                                         # pragma: no cover
        try:
            self.panel.warm_up()
        except Exception:
            self.panel.warming = False


class FeatureSpaceWidget(QtWidgets.QWidget):
    """
    Every training window as one point, in two dimensions, plus the live one.

    The projection is an unsupervised PCA computed by the trainer (see
    `feature_space_projection`). It is NOT an LDA, and that matters: an LDA
    picks its axes to separate the labels, so its picture looks clean even
    when the separation comes from electrode drift rather than from the
    gesture. What is drawn here is the structure that is really in the
    features, which on this dataset means the class-take confound is visible
    rather than hidden.

    Three colourings, because each answers a different question:

      * BY GESTURE  -- do the labels form their own clusters?
      * BY TAKE     -- or do the clusters actually follow the recording
                       session? If this looks tidier than BY GESTURE, the
                       model is reading the block (docs/EDA.md section 2).
      * BY LEVEL    -- where does amplitude sit? This is the picture of
                       docs/EDA.md R4: the gap between resting and the
                       effortful holds that a relaxed gesture falls into.

    A painter, not a source. With no projection it paints the reason.
    """

    MARGIN_L, MARGIN_R = 46, 12
    MARGIN_T, MARGIN_B = 10, 34

    def __init__(self, panel: "DecisionPanel", parent=None):
        super().__init__(parent)
        self.panel = panel
        self.mode = "label"
        self.live_point: Optional[Tuple[float, float]] = None
        self.live_label: Optional[str] = None
        self.trail: List[Tuple[float, float]] = []
        self.setMinimumHeight(300)
        self.setSizePolicy(QtWidgets.QSizePolicy.Expanding,
                           QtWidgets.QSizePolicy.Expanding)
        self._bounds = None

    # -- inputs -----------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        self.mode = mode
        self.update()

    def set_live(self, point, label: str = None) -> None:
        """
        Place the live window. `None` clears it rather than leaving the last
        one on screen pretending to be current.
        """
        self.live_point = point
        self.live_label = label
        if point is not None:
            #  A short tail, so a gesture decaying out of its cluster is
            #  visible as a direction and not just a jumping dot.
            self.trail.append(point)
            del self.trail[:-40]
        self.update()

    def clear_trail(self) -> None:
        self.trail = []
        self.update()

    # -- geometry ---------------------------------------------------------
    def _plot_rect(self) -> QtCore.QRectF:
        return QtCore.QRectF(
            self.MARGIN_L, self.MARGIN_T,
            max(10, self.width() - self.MARGIN_L - self.MARGIN_R),
            max(10, self.height() - self.MARGIN_T - self.MARGIN_B))

    def _data_bounds(self):
        """Axis limits from the training cloud, padded, cached per load."""
        if self._bounds is not None:
            return self._bounds
        coords = np.asarray(self.panel.space["coords"], dtype=np.float64)
        lo = coords.min(axis=0)
        hi = coords.max(axis=0)
        span = np.maximum(hi - lo, 1e-6)
        #  Padded so the live point can sit outside the training cloud and
        #  still be drawn -- that it CAN is the interesting case.
        lo = lo - span * 0.12
        hi = hi + span * 0.12
        self._bounds = (lo, hi)
        return self._bounds

    def _to_px(self, rect, lo, hi, x, y):
        fx = (x - lo[0]) / max(1e-9, hi[0] - lo[0])
        fy = (y - lo[1]) / max(1e-9, hi[1] - lo[1])
        return (rect.left() + fx * rect.width(),
                rect.bottom() - fy * rect.height())

    # -- painting ---------------------------------------------------------
    def paintEvent(self, event) -> None:
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, False)
        painter.fillRect(self.rect(), QtGui.QColor(COL_BG))

        if not self.panel.has_space:
            self._paint_missing(painter)
            painter.end()
            return

        space = self.panel.space
        rect = self._plot_rect()
        lo, hi = self._data_bounds()

        painter.setPen(QtGui.QPen(QtGui.QColor(COL_BORDER), 1))
        painter.setBrush(QtGui.QColor(COL_PANEL))
        painter.drawRect(rect)

        coords = np.asarray(space["coords"], dtype=np.float64)
        truth = np.asarray(space["truth"], dtype=int)
        labels = list(self.panel.labels)

        #  Grouped into one polygon per colour and drawn with drawPoints:
        #  one call per colour instead of one per point, which is what keeps
        #  several thousand points off the GUI thread's budget.
        groups = self._colour_groups(space, truth, labels)
        for colour, idx in groups:
            if len(idx) == 0:
                continue
            poly = QtGui.QPolygonF()
            for x, y in coords[idx]:
                px, py = self._to_px(rect, lo, hi, x, y)
                poly.append(QtCore.QPointF(px, py))
            pen = QtGui.QPen(QtGui.QColor(colour), 3)
            pen.setCapStyle(QtCore.Qt.RoundCap)
            painter.setPen(pen)
            painter.drawPoints(poly)

        self._paint_live(painter, rect, lo, hi)
        self._paint_axes(painter, rect, space)
        self._paint_legend(painter, rect, groups, space, labels)
        painter.end()

    def _colour_groups(self, space, truth, labels):
        """(colour, row indices) per colour, for the current mode."""
        if self.mode == "take" and space.get("take") is not None:
            take = np.asarray(space["take"], dtype=int)
            out = []
            for t in range(int(take.max()) + 1 if take.size else 0):
                out.append((class_color(t), np.flatnonzero(take == t)))
            return out
        if self.mode == "level" and space.get("level") is not None:
            level = np.asarray(space["level"], dtype=float)
            #  Fixed bands rather than quantiles, so the colours mean an
            #  amplitude and not a rank -- the point is to read the gap.
            edges = [0, 45, 90, 140, 200, 280, 1e9]
            ramp = ["#1e3a5f", "#2563eb", "#22c55e", "#f59e0b",
                    "#f97316", "#ef4444"]
            out = []
            for i in range(len(edges) - 1):
                out.append((ramp[i], np.flatnonzero(
                    (level >= edges[i]) & (level < edges[i + 1]))))
            return out
        return [(LABEL_COLORS.get(labels[k], class_color(k)),
                 np.flatnonzero(truth == k)) for k in range(len(labels))]

    def _paint_live(self, painter, rect, lo, hi) -> None:
        if not self.trail:
            return
        #  The tail, oldest faintest.
        for i, (x, y) in enumerate(self.trail[:-1]):
            px, py = self._to_px(rect, lo, hi, x, y)
            alpha = int(30 + 120 * (i / max(1, len(self.trail) - 1)))
            colour = QtGui.QColor(COL_TEXT)
            colour.setAlpha(alpha)
            painter.setPen(QtGui.QPen(colour, 2))
            painter.drawPoint(QtCore.QPointF(px, py))
        if self.live_point is None:
            return
        px, py = self._to_px(rect, lo, hi, *self.live_point)
        colour = QtGui.QColor(LABEL_COLORS.get(self.live_label or "", COL_TEXT))
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(QtGui.QPen(QtGui.QColor(COL_BG), 4))
        painter.drawEllipse(QtCore.QPointF(px, py), 7, 7)
        painter.setPen(QtGui.QPen(colour, 2))
        painter.drawEllipse(QtCore.QPointF(px, py), 7, 7)
        painter.drawLine(QtCore.QPointF(px - 11, py), QtCore.QPointF(px - 4, py))
        painter.drawLine(QtCore.QPointF(px + 4, py), QtCore.QPointF(px + 11, py))
        painter.drawLine(QtCore.QPointF(px, py - 11), QtCore.QPointF(px, py - 4))
        painter.drawLine(QtCore.QPointF(px, py + 4), QtCore.QPointF(px, py + 11))

    def _paint_axes(self, painter, rect, space) -> None:
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(COL_TEXT_DIM))
        explained = space.get("explained") or [0.0, 0.0]
        painter.drawText(
            QtCore.QRectF(rect.left(), rect.bottom() + 4, rect.width(), 14),
            QtCore.Qt.AlignHCenter,
            f"PC1 -- {explained[0] * 100:.1f} % of the variance")
        painter.save()
        painter.translate(12, rect.center().y())
        painter.rotate(-90)
        painter.drawText(QtCore.QRectF(-rect.height() / 2, -10,
                                       rect.height(), 14),
                         QtCore.Qt.AlignHCenter,
                         f"PC2 -- {explained[1] * 100:.1f} %")
        painter.restore()

    def _paint_legend(self, painter, rect, groups, space, labels) -> None:
        font = painter.font()
        font.setPointSize(8)
        painter.setFont(font)
        if self.mode == "take":
            names = [f"take {i + 1}" for i in range(len(groups))]
            names = names[:10]
        elif self.mode == "level":
            names = ["<45", "45-90", "90-140", "140-200", "200-280", "280+"]
        else:
            names = labels
        x = rect.left() + 8
        y = rect.top() + 8
        for i, name in enumerate(names):
            if i >= len(groups):
                break
            colour, idx = groups[i]
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(QtGui.QColor(colour))
            painter.drawRect(QtCore.QRectF(x, y + 3, 7, 7))
            painter.setPen(QtGui.QColor(COL_TEXT))
            text = f"{name}  {len(idx)}"
            painter.drawText(QtCore.QRectF(x + 11, y, 150, 12),
                             QtCore.Qt.AlignLeft, text)
            y += 13
            if y > rect.bottom() - 16:
                break

    def _paint_missing(self, painter) -> None:
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        painter.setPen(QtGui.QColor(COL_TEXT_DIM))
        if not self.panel.ready:
            text = ("No decision panel loaded, so there is no feature space "
                    "to draw.\nRetrain, or run:  python explain_votes.py")
        else:
            text = ("This panel has no feature-space projection.\nIt was "
                    "built before the view existed -- rerun "
                    "explain_votes.py to add one.")
        painter.drawText(self.rect().adjusted(14, 14, -14, -14),
                         QtCore.Qt.AlignLeft | QtCore.Qt.TextWordWrap, text)


class FeatureSpaceView(QtWidgets.QWidget):
    """The FEATURE SPACE section: the colouring controls plus the scatter."""

    def __init__(self, panel: "DecisionPanel", predictor: "LivePredictor",
                 parent=None):
        super().__init__(parent)
        self.panel = panel
        self.predictor = predictor
        self._seen = 0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SP_SM)

        bar = QtWidgets.QHBoxLayout()
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(SP_SM)
        bar.addWidget(make_label("COLOUR BY", 9, False, COL_TEXT_DIM))
        self.mode_select = QtWidgets.QComboBox()
        space = panel.space or {}
        self.mode_select.addItem("gesture", "label")
        if space.get("take") is not None:
            self.mode_select.addItem("take it was cut from", "take")
        if space.get("level") is not None:
            self.mode_select.addItem("amplitude level", "level")
        self.mode_select.setToolTip(
            "gesture: do the labels cluster?\n"
            "take: or do the clusters follow the recording session instead? "
            "If this looks tidier than 'gesture', the model is reading the "
            "block, not the gesture.\n"
            "amplitude level: where the effort sits, and the gap a relaxed "
            "hold falls into.")
        self.mode_select.currentIndexChanged.connect(self._mode_changed)
        bar.addWidget(self.mode_select)

        self.clear_button = QtWidgets.QPushButton("CLEAR TRAIL")
        self.clear_button.setToolTip("Forget the live window's recent path.")
        self.clear_button.clicked.connect(lambda: self.plot.clear_trail())
        bar.addWidget(self.clear_button)
        bar.addStretch(1)
        self.readout = make_label("", 9, False, COL_TEXT_DIM)
        bar.addWidget(self.readout)
        layout.addLayout(bar)

        self.plot = FeatureSpaceWidget(panel, self)
        layout.addWidget(self.plot, 1)

        self.caption = make_label("", 9, False, COL_TEXT_DIM)
        self.caption.setWordWrap(True)
        layout.addWidget(self.caption)
        self._update_caption()
        self.controls_enabled(panel.has_space)

    def controls_enabled(self, on: bool) -> None:
        self.mode_select.setEnabled(on)
        self.clear_button.setEnabled(on)

    def _mode_changed(self) -> None:
        self.plot.set_mode(self.mode_select.currentData() or "label")
        self._update_caption()

    def _update_caption(self) -> None:
        space = self.panel.space or {}
        if not self.panel.has_space:
            self.caption.setText("")
            return
        head = (f"{space.get('n_points', 0)} of "
                f"{space.get('n_windows', 0)} training windows, projected "
                f"from {space.get('n_features', 0)} features by an "
                f"unsupervised PCA -- the axes were chosen without seeing "
                f"the labels, so any separation here is really in the data. ")
        mode = self.mode_select.currentData() or "label"
        if mode == "take":
            tail = ("Clusters that follow the take rather than the gesture "
                    "are the confound in docs/EDA.md section 2: the model "
                    "can score well by recognising which recording block it "
                    "is looking at.")
        elif mode == "level":
            tail = ("The amplitude bands. A gesture recorded only at full "
                    "effort leaves a gap above rest that a relaxed hold "
                    "falls into, which is docs/EDA.md R4.")
        else:
            tail = ("The ring is the window the classifier just used -- the "
                    "same feature vector, placed by the same transform. "
                    "Watch it leave its cluster when a hold relaxes.")
        self.caption.setText(head + tail)

    def refresh_live(self) -> None:
        """Place the newest predicted window. Called on the dialog's timer."""
        if not self.panel.has_space:
            return
        predictor = self.predictor
        features = predictor.last_features
        if features is None:
            self.readout.setText("no live window yet")
            return
        #  Keyed on the prediction's own timestamp, exactly as the trail
        #  does: `id(features)` would be recycled by the allocator and could
        #  make a new window look like the previous one.
        last = predictor.last
        stamp = getattr(last, "unix_time", 0.0) if last else 0.0
        if stamp and stamp == self._seen:
            return
        self._seen = stamp
        point = self.panel.project(features)
        if point is None:
            self.readout.setText("live vector does not fit this projection")
            return
        label = getattr(last, "raw_label", None) if last else None
        self.plot.set_live(point, label)
        self.readout.setText(
            f"live: PC1 {point[0]:+.2f}  PC2 {point[1]:+.2f}"
            + (f"  -> {label}" if label else ""))


class DecisionTrailView(QtWidgets.QWidget):
    """
    The DECISION TRAIL section of MODEL STATS: controls plus the drawing.

    Two sources, and which one is on screen is never ambiguous:

      * LIVE follows the classifier. While PREDICT is running every new
        window re-runs all the candidates on the very feature vector the
        deployed model just used. With nothing streaming it holds the last
        window and says so.
      * RECORDED scrubs the out-of-fold trail stored at training time, which
        is where the honest numbers are.

    With no panel file the whole section reduces to one sentence explaining
    how to get one. Nothing here can show a number the panel did not supply.
    """

    def __init__(self, panel: "DecisionPanel", predictor: "LivePredictor",
                 parent=None):
        super().__init__(parent)
        self.panel = panel
        self.predictor = predictor
        self.row = 0
        self.live = True
        self._rows: List[int] = []
        self._splits: List[int] = []
        self._live_seen = 0

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SP_SM)

        self.controls = QtWidgets.QWidget()
        bar = QtWidgets.QHBoxLayout(self.controls)
        bar.setContentsMargins(0, 0, 0, 0)
        bar.setSpacing(SP_SM)

        self.live_button = QtWidgets.QPushButton("LIVE")
        self.live_button.setCheckable(True)
        self.live_button.setChecked(True)
        self.live_button.setToolTip(
            "Follow the classifier: every predicted window is re-run through "
            "all the candidates.")
        self.live_button.clicked.connect(lambda: self._set_live(True))
        self.recorded_button = QtWidgets.QPushButton("RECORDED")
        self.recorded_button.setCheckable(True)
        self.recorded_button.setToolTip(
            "Scrub the out-of-fold decisions captured during training.")
        self.recorded_button.clicked.connect(lambda: self._set_live(False))
        bar.addWidget(self.live_button)
        bar.addWidget(self.recorded_button)

        self.rec_select = QtWidgets.QComboBox()
        self.rec_select.setMinimumWidth(190)
        self.rec_select.currentIndexChanged.connect(self._pick_recording)
        bar.addWidget(self.rec_select, 1)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.valueChanged.connect(self._pick_window)
        bar.addWidget(self.slider, 2)

        self.position = make_label("--", 9, False, COL_TEXT_DIM)
        bar.addWidget(self.position)

        self.split_button = QtWidgets.QPushButton("NEXT DISAGREEMENT")
        self.split_button.setToolTip(
            "Jump to the next recorded window where the candidates did not "
            "all pick the same gesture.")
        self.split_button.clicked.connect(self._next_split)
        bar.addWidget(self.split_button)
        layout.addWidget(self.controls)

        self.trail = DecisionTrailWidget()
        layout.addWidget(self.trail, 1)

        self.footer = make_label("", 9, False, COL_TEXT_DIM)
        self.footer.setWordWrap(True)
        layout.addWidget(self.footer)

        self._build()

    # -- setup ----------------------------------------------------------------

    def _build(self) -> None:
        panel = self.panel
        if not panel.ready:
            self.controls.setVisible(False)
            self.trail.set_empty(
                "No decision panel.\n\n" + panel.status +
                "\n\nRetrain (RETRAIN, or python train_model.py) and this "
                "view will show every candidate's vote on each window.")
            self.footer.setText("")
            return

        stale = panel.stale_reason(self.predictor)
        if stale:
            self.controls.setVisible(False)
            self.trail.set_empty(
                "The decision panel does not match the loaded model.\n\n"
                + stale)
            self.footer.setText("")
            return

        #  Pay the candidates' first-call cost on a worker thread now, so the
        #  first live window does not stall the GUI for ~2.5 s.
        if not panel.warm and not panel.warming:
            QtCore.QThreadPool.globalInstance().start(PanelWarmUp(panel))

        self._splits = panel.split_windows()
        self.rec_select.blockSignals(True)
        self.rec_select.clear()
        for index, entry in enumerate(panel.recordings):
            wrong = [n for n in panel.names
                     if not entry["models"].get(n, {}).get("correct", True)]
            text = (f"{entry['label']}  {entry['uid']}  "
                    f"{entry['n_windows']}w  fold {entry['fold'] + 1}")
            if wrong:
                text += f"  [{len(wrong)} wrong]"
            self.rec_select.addItem(text, index)
        self.rec_select.blockSignals(False)
        self.split_button.setEnabled(bool(self._splits))

        counts = ", ".join(
            f"{name} {sum(1 for r in panel.recordings if r['label'] == name)}"
            for name in panel.labels)
        verified = panel.fusion.get("verified")
        difference = panel.fusion.get("max_abs_diff_vs_mean")
        self.footer.setText(
            f"Panel built {panel.created_iso[:16].replace('T', ' ')} from "
            f"{panel.n_windows} windows of {panel.n_recordings} recordings "
            f"({counts}).  Fusion: {panel.fusion.get('rule', '?')}"
            + (f"; checked against the stored ensemble, max difference "
               f"{difference:.1e}." if isinstance(difference, float) else ".")
            + ("" if verified else
               "  The check did not pass -- read the ensemble's own output, "
               "not the mean.")
            + f"  Cross-validation: {panel.protocol.get('cv', '?')}, "
              f"{panel.protocol.get('n_splits', '?')} folds."
        )
        self._set_live(True)

    def _set_live(self, live: bool) -> None:
        self.live = bool(live)
        self.live_button.setChecked(self.live)
        self.recorded_button.setChecked(not self.live)
        for widget in (self.rec_select, self.slider, self.split_button):
            widget.setEnabled(not self.live)
        self.split_button.setEnabled(not self.live and bool(self._splits))
        if self.live:
            self.refresh_live(force=True)
        else:
            if self.rec_select.count() and not self._rows:
                self._pick_recording(self.rec_select.currentIndex())
            self._show_recorded()

    # -- recorded -------------------------------------------------------------

    def _pick_recording(self, index: int) -> None:
        if index < 0 or index >= len(self.panel.recordings):
            return
        rows = self.panel.windows_of(index)
        self._rows = rows
        self.slider.blockSignals(True)
        self.slider.setMaximum(max(0, len(rows) - 1))
        self.slider.setValue(0)
        self.slider.blockSignals(False)
        self.row = rows[0] if rows else 0
        if not self.live:
            self._show_recorded()

    def _pick_window(self, value: int) -> None:
        rows = self._rows
        if 0 <= value < len(rows):
            self.row = rows[value]
            if not self.live:
                self._show_recorded()

    def _next_split(self) -> None:
        if not self._splits:
            return
        later = [r for r in self._splits if r > self.row]
        target = later[0] if later else self._splits[0]
        index = self.panel.recording_index_of(target)
        if index >= 0:
            if index != self.rec_select.currentIndex():
                self.rec_select.setCurrentIndex(index)
            if target in self._rows:
                self.slider.setValue(self._rows.index(target))
        self.row = target
        self._show_recorded()

    def _show_recorded(self) -> None:
        step = self.panel.recorded_step(self.row)
        if step is None:
            self.trail.set_empty(
                "This panel carries no per-window trail; only the live view "
                "and the scores are available.")
            return
        rows = self._rows
        position = rows.index(self.row) + 1 if self.row in rows else 1
        self.position.setText(f"{position} / {max(1, len(rows))}")
        truth = self.panel.recorded_truth(self.row)
        fused = step.fused if step.fused is not None else step.fused_from_mean
        correct = step.decision(fused) == truth
        self.trail.set_step(
            step, channels=self.panel.recorded_channels(self.row),
            caption=("The vote was " + ("right" if correct else "WRONG")
                     + " on this window.  These are out-of-fold numbers: the "
                     "recording was held out of the fold that produced them, "
                     "so they are the honest ones."),
            cv=self.panel.cv)

    # -- live -----------------------------------------------------------------

    def refresh_live(self, force: bool = False) -> None:
        """Re-run the candidates on the newest predicted window."""
        if not self.live or not self.panel.ready:
            return
        if self.panel.stale_reason(self.predictor):
            return
        if self.panel.warming:
            self.trail.set_empty(
                "Preparing the candidates...\n\nEach one's first prediction "
                "builds its kernel; this happens once. Live windows appear as "
                "soon as it finishes.")
            self.position.setText("...")
            return

        features = self.predictor.last_features
        if features is None:
            self.trail.set_empty(
                "Waiting for a live window.\n\nPress PREDICT in the LIVE "
                "CLASSIFIER panel, or switch to RECORDED to scrub the "
                "windows captured during training.")
            self.position.setText("--")
            return

        last = self.predictor.last
        stamp = getattr(last, "unix_time", 0.0) if last else 0.0
        if not force and stamp == self._live_seen:
            return
        self._live_seen = stamp

        step = self.panel.step_for_features(features, self.predictor)
        if step is None:
            self.trail.set_empty("The decision panel could not run on this "
                                 "window: " + self.panel.status)
            return
        window = self.predictor.last_window
        channels = None
        if window is not None and np.asarray(window).ndim == 2:
            channels = np.mean(np.asarray(window, dtype=float), axis=0)
        age = time.time() - stamp if stamp else None
        self.position.setText("live" if (age is not None and age < 2.0)
                              else "held")
        self.trail.set_step(
            step, channels=channels,
            caption=("Every candidate ran on the same feature vector the "
                     "deployed model just used. They were fitted on the whole "
                     "dataset, so agreement here shows how the vote forms -- "
                     "it is not an accuracy estimate. For that, use RECORDED "
                     "or the scores below."),
            cv=self.panel.cv)


class ModelStatsDialog(QtWidgets.QDialog):
    """
    What the trained model actually is and how well it really does.

    Everything here comes from the cross-validation that train_model.py ran;
    nothing is recomputed or rounded up. The validation protocol and its
    limitations are shown as prominently as the accuracy, because an accuracy
    without its protocol is not a result.
    """

    def __init__(self, predictor: "LivePredictor", parent=None,
                 panel: "DecisionPanel" = None):
        super().__init__(parent)
        self.setWindowTitle("Model statistics")
        fit_to_screen(self, 940, 820, min_width=480, min_height=360)
        self.predictor = predictor
        #  Self-sufficient when nobody hands one in: the dialog is opened
        #  from the tests and from --summary paths too, and a missing panel
        #  is a normal state with its own message, not an error.
        if panel is None:
            panel = DecisionPanel(model_path=predictor.path)
            panel.load()
        self.panel = panel
        self.trail_view = None
        self.space_view = None
        self.trail_timer = None

        stats = dict(predictor.stats or {})
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        if not predictor.ready and not stats:
            layout.addWidget(make_label("No model loaded.", 14, True))
            layout.addWidget(make_label(predictor.status, 11, False, COL_TEXT_DIM))
            layout.addWidget(make_label(
                "Train one with:    python train_model.py", 11, False, COL_ACCENT))
            close = QtWidgets.QPushButton("Close")
            close.clicked.connect(self.accept)
            layout.addStretch(1)
            layout.addWidget(close)
            return

        layout.addWidget(make_label(
            stats.get("model_name", "model"), 18, True, COL_ACCENT))
        layout.addWidget(make_label(
            f"trained {stats.get('created_iso', '?')}", 10, False, COL_TEXT_DIM))

        # ---- headline numbers ------------------------------------------------
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        body = QtWidgets.QVBoxLayout(inner)
        body.setSpacing(10)

        window_acc = stats.get("window_accuracy")
        vote_acc = stats.get("recording_vote_accuracy")
        tiles = QtWidgets.QHBoxLayout()
        for caption, value, note in (
            ("PER GESTURE", vote_acc,
             f"majority vote, {stats.get('n_recordings_voted', '?')} recordings"),
            ("PER WINDOW", window_acc,
             f"single {stats.get('window_s', 0) * 1000:.0f} ms slice, no smoothing"),
            ("MACRO F1", stats.get("window_macro_f1"), "per window, class-balanced"),
        ):
            box = QtWidgets.QFrame()
            box.setStyleSheet(
                f"QFrame {{ background: {COL_PANEL}; border: 1px solid "
                f"{COL_BORDER}; border-radius: 8px; }}")
            col = QtWidgets.QVBoxLayout(box)
            col.setSpacing(1)
            col.addWidget(make_label(caption, 9, False, COL_TEXT_DIM))
            text = "--" if value is None else f"{value * 100:.1f} %"
            colour = COL_TEXT_DIM
            if value is not None:
                colour = (COL_GOOD if value >= 0.85
                          else COL_WARN if value >= 0.6 else COL_BAD)
            col.addWidget(make_label(text, 22, True, colour))
            note_label = make_label(note, 9, False, COL_TEXT_DIM)
            note_label.setWordWrap(True)
            col.addWidget(note_label)
            tiles.addWidget(box)
        body.addLayout(tiles)

        chance = 1.0 / max(1, len(predictor.labels or stats.get("labels", []) or [1]))
        body.addWidget(make_label(
            f"Chance level for {len(predictor.labels)} gestures is "
            f"{chance * 100:.1f} %.", 11, False, COL_TEXT_DIM))

        # ---- decision trail --------------------------------------------------
        #  How every candidate judged this window and how the seven become
        #  one. Placed above the rest because it is the only part of this
        #  dialog that moves while you use the armband.
        body.addWidget(make_label("DECISION TRAIL", 11, True, COL_TEXT_DIM))
        note = make_label(
            "Every classifier in the spec, on one window, fusing into a "
            "single soft vote. The row marked * is the one that actually "
            "predicts; the vote is shown for comparison and is not what the "
            "app loads.", 10, False, COL_TEXT_DIM)
        note.setWordWrap(True)
        body.addWidget(note)

        self.trail_view = DecisionTrailView(self.panel, predictor, self)
        trail_frame = QtWidgets.QFrame()
        trail_frame.setStyleSheet(
            f"QFrame {{ background: {COL_PANEL}; border: 1px solid "
            f"{COL_BORDER}; border-radius: {RAD_SM}px; }}")
        trail_box = QtWidgets.QVBoxLayout(trail_frame)
        trail_box.setContentsMargins(SP_SM, SP_SM, SP_SM, SP_SM)
        trail_box.addWidget(self.trail_view)
        body.addWidget(trail_frame)

        # ---- feature space ---------------------------------------------------
        #  The same windows the numbers above come from, as a picture. Placed
        #  under the trail because it answers a different question: not "what
        #  did the models say about this window" but "where does this window
        #  sit among everything they were trained on".
        body.addWidget(make_label("FEATURE SPACE", 11, True, COL_TEXT_DIM))
        self.space_view = FeatureSpaceView(self.panel, predictor, self)
        space_frame = QtWidgets.QFrame()
        space_frame.setStyleSheet(
            f"QFrame {{ background: {COL_PANEL}; border: 1px solid "
            f"{COL_BORDER}; border-radius: {RAD_SM}px; }}")
        space_box = QtWidgets.QVBoxLayout(space_frame)
        space_box.setContentsMargins(SP_SM, SP_SM, SP_SM, SP_SM)
        space_box.addWidget(self.space_view)
        body.addWidget(space_frame)

        if self.panel is not None and self.panel.ready \
                and not self.panel.stale_reason(predictor):
            self._add_candidate_table(body, predictor)

        #  Repaint on the classifier's own cadence rather than faster: a
        #  prediction arrives a few times a second and there is nothing new
        #  to draw in between.
        self.trail_timer = QtCore.QTimer(self)
        self.trail_timer.setInterval(120)
        self.trail_timer.timeout.connect(self._tick_trail)
        self.trail_timer.start()

        # ---- what the model actually is --------------------------------------
        # Read off the loaded estimator rather than off the spec: this is the
        # object that will run on the next prediction.
        report = QtWidgets.QTextBrowser()
        report.setOpenExternalLinks(False)
        report.setMinimumHeight(360)
        report.setStyleSheet(
            f"QTextBrowser {{ background: {COL_BG}; border: 1px solid "
            f"{COL_BORDER}; border-radius: {RAD_SM}px; padding: {SP_SM}px; }}")
        report.setHtml(model_report_html(predictor, stats))
        report.setToolTip(
            "Read off the fitted estimator on disk, not off the spec that "
            "was meant to produce it. If the two ever disagreed, this is the "
            "one that predicts.")
        body.addWidget(report, 1)

        # ---- how it was validated -------------------------------------------
        body.addWidget(make_label("HOW THIS WAS MEASURED", 11, True, COL_TEXT_DIM))
        protocol = QtWidgets.QPlainTextEdit()
        protocol.setReadOnly(True)
        protocol.setMaximumHeight(130)
        folds = stats.get("fold_accuracies") or []
        subject_independent = stats.get("subject_independent")
        lines = [
            f"Protocol      : {stats.get('validation_protocol', '?')}",
            f"Folds         : {stats.get('n_splits', '?')}"
            + (f"   ({', '.join(f'{a * 100:.0f}%' for a in folds)})" if folds else ""),
            f"Windows       : {stats.get('n_windows', '?')} "
            f"x {stats.get('n_features', '?')} features",
            f"Recordings    : {stats.get('n_recordings', '?')} used "
            f"of {stats.get('n_recordings_total', '?')} "
            f"(filter: {stats.get('excluded', '?')})",
            f"Subjects      : {', '.join(stats.get('subjects', []) or ['?'])}"
            f"   sessions: {stats.get('n_sessions', '?')}",
            "",
            "A whole recording is either in train or in test, never both --"
            " windows overlap, so a random split would leak and inflate this.",
        ]
        if subject_independent:
            lines.append(
                f"Subject-independent: {subject_independent['mean'] * 100:.1f} % "
                f"+- {subject_independent['std'] * 100:.1f}")
        else:
            lines.append(
                "Subject-independent: NOT MEASURABLE -- one subject only. This "
                "model is specific to the person who recorded it.")
        protocol.setPlainText("\n".join(lines))
        body.addWidget(protocol)

        # ---- per class -------------------------------------------------------
        cm = stats.get("confusion_matrix")
        labels = stats.get("labels") or predictor.labels
        if cm and labels:
            matrix = np.array(cm, dtype=float)
            body.addWidget(make_label(
                "PER GESTURE (per window, summed over folds)", 11, True,
                COL_TEXT_DIM))
            table = QtWidgets.QTableWidget(len(labels), 4)
            table.setHorizontalHeaderLabels(
                ["gesture", "recall", "precision", "windows"])
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
            table.horizontalHeader().setSectionResizeMode(
                0, QtWidgets.QHeaderView.Stretch)
            for i, name in enumerate(labels):
                support = matrix[i].sum()
                recall = matrix[i, i] / support if support else 0.0
                column = matrix[:, i].sum()
                precision = matrix[i, i] / column if column else 0.0
                for col, text in enumerate(
                        [name, f"{recall * 100:.1f} %",
                         f"{precision * 100:.1f} %", f"{int(support)}"]):
                    item = QtWidgets.QTableWidgetItem(text)
                    if col == 1:
                        item.setForeground(QtGui.QColor(
                            COL_GOOD if recall >= 0.8 else
                            COL_WARN if recall >= 0.5 else COL_BAD))
                    table.setItem(i, col, item)
            table.setMinimumHeight(min(260, 30 + 28 * len(labels)))
            body.addWidget(table)

            body.addWidget(make_label(
                "CONFUSION MATRIX  (rows = truth, columns = predicted)",
                11, True, COL_TEXT_DIM))
            grid = QtWidgets.QTableWidget(len(labels), len(labels))
            grid.setHorizontalHeaderLabels([l[:7] for l in labels])
            grid.setVerticalHeaderLabels([l[:9] for l in labels])
            grid.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
            peak = max(1.0, matrix.max())
            for i in range(len(labels)):
                row_total = matrix[i].sum()
                for j in range(len(labels)):
                    share = matrix[i, j] / row_total if row_total else 0.0
                    item = QtWidgets.QTableWidgetItem(f"{int(matrix[i, j])}")
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    item.setToolTip(f"{labels[i]} predicted as {labels[j]}: "
                                    f"{share * 100:.1f} %")
                    if matrix[i, j]:
                        strength = matrix[i, j] / peak
                        if i == j:
                            item.setBackground(QtGui.QColor(
                                34, 197, 94, int(40 + 150 * strength)))
                        else:
                            item.setBackground(QtGui.QColor(
                                239, 68, 68, int(30 + 140 * strength)))
                    grid.setItem(i, j, item)
            grid.resizeColumnsToContents()
            grid.setMinimumHeight(min(300, 40 + 30 * len(labels)))
            body.addWidget(grid)

        # ---- setup -----------------------------------------------------------
        body.addWidget(make_label("PREPROCESSING", 11, True, COL_TEXT_DIM))
        setup = QtWidgets.QPlainTextEdit()
        setup.setReadOnly(True)
        setup.setMaximumHeight(120)
        setup.setPlainText("\n".join([
            f"window         : {stats.get('window_s', predictor.window_s)} s "
            f"({predictor.window_samples} samples at {predictor.fs:.0f} Hz)",
            f"overlap        : {stats.get('overlap', '?')}",
            f"representation : {predictor.representation}",
            f"feature set    : {predictor.feature_set} "
            f"({stats.get('n_features', '?')} features)",
            f"EMG mode       : {predictor.payload_meta.get('emg_mode', '?')}",
            f"report         : {stats.get('report_file', '-')}",
        ]))
        body.addWidget(setup)

        fit = stats.get("train_fit_accuracy")
        if fit is not None:
            body.addWidget(make_label(
                f"Training-set fit: {fit * 100:.1f} %. This is NOT a "
                "performance estimate -- it only shows the model can represent "
                "the data. Judge it by the cross-validated numbers above.",
                10, False, COL_WARN))

        # ---- provenance ------------------------------------------------------
        #  Which recordings this model saw, and whether that is still the
        #  dataset. Asked constantly, and unanswerable from the scores alone.
        body.addWidget(make_label("TRAINING DATA", 11, True, COL_TEXT_DIM))
        data = predictor.payload_meta.get("data") or {}
        status = predictor.data_status(dataset_fingerprint(
            os.path.join(PROJECT_DIR, DATA_FOLDER)))
        provenance = QtWidgets.QPlainTextEdit()
        provenance.setReadOnly(True)
        provenance.setMaximumHeight(170)
        lines = [
            f"status         : {status['headline']}",
            f"trained        : {predictor.payload_meta.get('created_iso', '?')}",
            f"recordings     : {data.get('n_recordings', stats.get('n_recordings', '?'))}"
            + (f" of {stats['n_recordings_total']} on disk at the time"
               if stats.get("n_recordings_total") else ""),
            f"per gesture    : " + (", ".join(
                f"{name} {count}"
                for name, count in sorted((data.get('labels') or {}).items()))
                or ", ".join(stats.get("labels") or predictor.labels)),
            f"subjects       : " + (", ".join(
                data.get("subjects") or stats.get("subjects") or []) or "?"),
            f"sessions       : " + (", ".join(data.get("sessions") or [])
                                    or str(stats.get("n_sessions", "?"))),
            f"newest in set  : " + (data.get("newest_stamp") or "not recorded"),
            f"dataset id     : " + (data.get("uid_digest") or "not recorded"),
            f"excluded       : {stats.get('excluded', '?')}",
        ]
        provenance.setPlainText("\n".join(lines))
        body.addWidget(provenance)
        body.addWidget(make_label(
            status["detail"], 10, False,
            {"good": COL_GOOD, "warn": COL_WARN}.get(status["severity"],
                                                     COL_TEXT_DIM)))

        body.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        reload_button = QtWidgets.QPushButton("Reload model")
        reload_button.clicked.connect(lambda: self._reload(predictor))
        close = QtWidgets.QPushButton("Close")
        close.setObjectName("primary")
        close.setDefault(True)
        close.clicked.connect(self.accept)
        buttons.addWidget(reload_button)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    def _tick_trail(self) -> None:
        if not self.isVisible():
            return
        if self.trail_view is not None:
            self.trail_view.refresh_live()
        #  Shares the trail's timer rather than adding a second one: both
        #  draw the same live window, and it only changes when a prediction
        #  arrives.
        if self.space_view is not None:
            self.space_view.refresh_live()

    def _add_candidate_table(self, body, predictor) -> None:
        """
        Every candidate's out-of-fold score on the winner's setup.

        Not the same thing as `best_per_model` in the sweep report: there,
        each model got to pick its own preprocessing. Here they all run on
        one setup, which is what makes the trail above a fair comparison and
        usually makes these numbers lower. Said in the dialog rather than
        left for someone to discover.
        """
        panel = self.panel
        body.addWidget(make_label(
            "EVERY CANDIDATE, SAME SETUP (out-of-fold)", 11, True,
            COL_TEXT_DIM))
        explain = make_label(
            f"All of them on the winning setup -- "
            f"{panel.setup.get('window_s', '?')} s windows, overlap "
            f"{panel.setup.get('overlap', '?')}, "
            f"{panel.setup.get('representation', '?')}/"
            f"{panel.setup.get('feature_set', '?')}, "
            f"{panel.setup.get('n_features', '?')} features. The sweep report "
            f"lets each model choose its own preprocessing instead, so those "
            f"numbers are higher and are not comparable with these.",
            10, False, COL_TEXT_DIM)
        explain.setWordWrap(True)
        body.addWidget(explain)

        order = list(panel.names) + ([panel.vote_name] if panel.vote_name
                                     and panel.vote_name in panel.cv else [])
        table = QtWidgets.QTableWidget(len(order), 5)
        table.setHorizontalHeaderLabels(
            ["candidate", "per window", "per recording", "mean conf.",
             "windows wrong"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch)
        for row, name in enumerate(order):
            entry = panel.cv.get(name, {})
            window = entry.get("window_accuracy")
            recording = entry.get("recording_accuracy")
            shown = name
            if name == panel.winner:
                shown += "   * deployed"
            elif name == panel.vote_name:
                shown += "   (ensemble)"
            cells = [
                shown,
                "--" if window is None else f"{window * 100:.2f} %",
                "--" if recording is None else f"{recording * 100:.2f} %",
                "--" if entry.get("mean_confidence") is None
                else f"{entry['mean_confidence']:.3f}",
                str(entry.get("n_windows_wrong", "--")),
            ]
            for column, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if column in (1, 2):
                    item.setTextAlignment(QtCore.Qt.AlignRight
                                          | QtCore.Qt.AlignVCenter)
                    value = window if column == 1 else recording
                    if value is not None:
                        item.setForeground(QtGui.QColor(
                            COL_GOOD if value >= 0.85 else
                            COL_WARN if value >= 0.6 else COL_BAD))
                if name == panel.winner:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                table.setItem(row, column, item)
        table.setMinimumHeight(min(300, 34 + 28 * len(order)))
        body.addWidget(table)

    def closeEvent(self, event):
        if self.trail_timer is not None:
            self.trail_timer.stop()
        super().closeEvent(event)

    def _reload(self, predictor) -> None:
        ok = predictor.load()
        QtWidgets.QMessageBox.information(
            self, "Reload model",
            "Model reloaded." if ok else f"Could not load: {predictor.status}")
        self.accept()


class ModelSpecDialog(QtWidgets.QDialog):
    """
    Which classifiers the sweep trains, with what hyperparameters, and whether
    they also vote as an ensemble.

    The editor is generated from `MODEL_KINDS`, so adding a model kind or a
    hyperparameter to that schema is all it takes to make it appear here --
    there is no second list of widgets to keep in step.

    This dialog changes what gets TRAINED, never what gets reported. Every
    score still comes from the same cross-validation in train_model.py, so
    switching a model on cannot flatter it.
    """

    def __init__(self, spec: dict, spec_path: str, parent=None):
        super().__init__(parent)
        self.spec_path = spec_path
        self.spec, problems = validate_model_spec(spec)
        self.saved = False
        self._loading = False
        self._param_widgets = {}

        self.setWindowTitle("Model specification")
        fit_to_screen(self, 1180, 860, min_width=760, min_height=520)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        root.addWidget(make_label(
            "Which classifiers the sweep trains, and how they vote",
            FS_XL, True, COL_ACCENT))
        intro = make_label(
            "Every enabled model is trained and scored with the same honest "
            "protocol, and the best one is saved. Turning a model on does not "
            "make it good -- the cross-validation decides that.",
            11, False, COL_TEXT_DIM)
        intro.setWordWrap(True)
        root.addWidget(intro)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body, 1)

        # ---- left: the roster ------------------------------------------------
        left = QtWidgets.QVBoxLayout()
        left.addWidget(make_label("MODELS", 10, True, COL_TEXT_DIM))

        self.model_list = QtWidgets.QListWidget()
        self.model_list.setMinimumWidth(320)
        # Deliberately no minimum height: a minimum taller than the space the
        # column actually has makes the list paint straight over the buttons
        # underneath it. It takes the stretch instead, so it is as tall as the
        # dialog allows and no taller.
        self.model_list.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                      QtWidgets.QSizePolicy.Expanding)
        # A row per model, not four rows and a scrollbar: choosing what to
        # train means comparing the list, which means seeing the list.
        self.model_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff)
        self.model_list.setTextElideMode(QtCore.Qt.ElideRight)
        self.model_list.currentRowChanged.connect(self._on_row_changed)
        self.model_list.itemChanged.connect(self._on_item_changed)
        left.addWidget(self.model_list, 1)

        self.roster_info = make_label("", 10, False, COL_TEXT_DIM)
        self.roster_info.setWordWrap(True)
        left.addWidget(self.roster_info)

        row = QtWidgets.QHBoxLayout()
        all_on = QtWidgets.QPushButton("ALL ON")
        all_on.setMinimumHeight(H_CONTROL)
        all_on.setToolTip("Train every model. Slower, and the sweep still "
                          "picks one winner.")
        all_on.clicked.connect(lambda: self._set_all(True))
        row.addWidget(all_on)
        all_off = QtWidgets.QPushButton("ALL OFF")
        all_off.setMinimumHeight(H_CONTROL)
        all_off.clicked.connect(lambda: self._set_all(False))
        row.addWidget(all_off)
        left.addLayout(row)

        row_b = QtWidgets.QHBoxLayout()
        self.add_combo = QtWidgets.QComboBox()
        for kind, schema in MODEL_KINDS.items():
            self.add_combo.addItem(schema["label"], kind)
        self.add_combo.setToolTip("Add another candidate of this kind. You can "
                                  "have several of the same kind with "
                                  "different hyperparameters.")
        row_b.addWidget(self.add_combo, 1)
        add = QtWidgets.QPushButton("ADD")
        add.clicked.connect(self.add_model)
        row_b.addWidget(add)
        left.addLayout(row_b)

        remove = QtWidgets.QPushButton("REMOVE SELECTED")
        remove.setObjectName("danger")
        remove.setMinimumHeight(H_CONTROL)
        remove.clicked.connect(self.remove_model)
        left.addWidget(remove)

        body.addLayout(left, 0)

        # ---- right: the selected model's settings ----------------------------
        right = QtWidgets.QVBoxLayout()
        self.detail_title = make_label("", FS_LG, True, COL_ACCENT)
        self.detail_title.setWordWrap(True)
        right.addWidget(self.detail_title)

        self.detail_note = make_label("", 10, False, COL_TEXT_DIM)
        self.detail_note.setWordWrap(True)
        right.addWidget(self.detail_note)

        form_box = QtWidgets.QGroupBox("HYPERPARAMETERS")
        self.form_host = QtWidgets.QVBoxLayout(form_box)
        self.form_host.setSpacing(6)
        self.form_host.setContentsMargins(10, 14, 10, 10)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(form_box)
        right.addWidget(scroll, 1)

        body.addLayout(right, 2)

        # ---- voting ----------------------------------------------------------
        vote_box = QtWidgets.QGroupBox("VOTING ENSEMBLE")
        vote_box.setMaximumHeight(150)
        vote = QtWidgets.QVBoxLayout(vote_box)
        vote.setSpacing(5)

        self.vote_check = QtWidgets.QCheckBox(
            "also build an ensemble that votes over the enabled models")
        self.vote_check.setToolTip(
            "The ensemble is added as one more candidate, so the sweep scores "
            "it against its own members.")
        self.vote_check.toggled.connect(self._on_voting_changed)
        vote.addWidget(self.vote_check)

        vrow = QtWidgets.QHBoxLayout()
        vrow.addWidget(make_label("Vote by", 11, False, COL_TEXT_DIM))
        self.vote_mode = QtWidgets.QComboBox()
        self.vote_mode.addItem("soft  -- average the predicted probabilities",
                               "soft")
        self.vote_mode.addItem("hard  -- count the votes", "hard")
        self.vote_mode.setToolTip(
            "Soft voting usually wins, but it needs predict_proba from every "
            "member; any member that cannot is left out of the ensemble.")
        self.vote_mode.currentIndexChanged.connect(self._on_voting_changed)
        vrow.addWidget(self.vote_mode, 1)
        vote.addLayout(vrow)

        self.vote_always = QtWidgets.QCheckBox(
            "always save the ensemble, even if a single model scored higher")
        self.vote_always.setToolTip(
            "Off (recommended): the ensemble competes and the best score "
            "wins. On: the ensemble is used regardless -- a deliberate "
            "choice, not an improvement.")
        self.vote_always.toggled.connect(self._on_voting_changed)
        vote.addWidget(self.vote_always)

        self.vote_info = make_label("", 10, False, COL_TEXT_DIM)
        self.vote_info.setWordWrap(True)
        vote.addWidget(self.vote_info)
        root.addWidget(vote_box)

        # ---- auto mode -------------------------------------------------------
        auto_box = QtWidgets.QGroupBox("AUTO MODE  (try combinations)")
        auto_box.setMaximumHeight(160)
        auto = QtWidgets.QVBoxLayout(auto_box)
        auto.setSpacing(5)

        self.auto_check = QtWidgets.QCheckBox(
            "try hyperparameter combinations for each model and keep the best")
        self.auto_check.setToolTip(
            "Every combination becomes its own candidate and is scored by the "
            "same cross-validation, so they all appear in the results table. "
            "This multiplies the training time."
        )
        self.auto_check.toggled.connect(self._on_auto_changed)
        auto.addWidget(self.auto_check)

        arow = QtWidgets.QHBoxLayout()
        arow.addWidget(make_label("Max combinations per model", 11, False,
                                  COL_TEXT_DIM))
        self.auto_spin = QtWidgets.QSpinBox()
        self.auto_spin.setRange(1, 64)
        self.auto_spin.setToolTip(
            "The full grid is sampled down to this many, seeded so a rerun "
            "gives the same candidates. Your configured values are always "
            "combination #1, so AUTO can never lose the setting you had."
        )
        self.auto_spin.valueChanged.connect(self._on_auto_changed)
        arow.addWidget(self.auto_spin, 1)
        auto.addLayout(arow)

        self.auto_info = make_label("", 10, False, COL_TEXT_DIM)
        self.auto_info.setWordWrap(True)
        auto.addWidget(self.auto_info)

        self.auto_warning = make_label("", 10, True, COL_WARN)
        self.auto_warning.setWordWrap(True)
        auto.addWidget(self.auto_warning)
        root.addWidget(auto_box)

        # ---- footer ----------------------------------------------------------
        self.problem_label = make_label("", 10, False, COL_WARN)
        self.problem_label.setWordWrap(True)
        root.addWidget(self.problem_label)

        where = make_label(
            f"Saved to {spec_path}  --  read by train_model.py on the next "
            f"retrain, and copied into the model it produces.",
            10, False, COL_TEXT_DIM)
        where.setWordWrap(True)
        root.addWidget(where)

        buttons = QtWidgets.QHBoxLayout()
        reset = QtWidgets.QPushButton("RESET TO DEFAULTS")
        reset.clicked.connect(self.reset_defaults)
        buttons.addWidget(reset)
        buttons.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        save = QtWidgets.QPushButton("SAVE SPEC")
        save.setObjectName("primary")
        save.setDefault(True)
        save.clicked.connect(self.save_spec)
        buttons.addWidget(save)
        root.addLayout(buttons)

        self._reload_all(problems)

    # -- population -----------------------------------------------------------

    def _reload_all(self, problems=None) -> None:
        self._loading = True
        try:
            self.model_list.clear()
            for entry in self.spec["models"]:
                item = QtWidgets.QListWidgetItem(self._row_text(entry))
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked if entry["enabled"]
                                   else QtCore.Qt.Unchecked)
                item.setForeground(QtGui.QColor(
                    COL_TEXT if entry["enabled"] else COL_TEXT_DIM))
                self.model_list.addItem(item)

            vote = self.spec["voting"]
            self.vote_check.setChecked(bool(vote["enabled"]))
            index = self.vote_mode.findData(vote["mode"])
            self.vote_mode.setCurrentIndex(max(0, index))
            self.vote_always.setChecked(bool(vote["always"]))

            auto = self.spec.get("auto") or {}
            self.auto_check.setChecked(bool(auto.get("enabled")))
            self.auto_spin.setValue(int(auto.get(
                "max_per_model", MODEL_AUTO_MAX_PER_MODEL)))
        finally:
            self._loading = False

        if self.model_list.count():
            self.model_list.setCurrentRow(0)
        self._refresh_summary(problems)

    @staticmethod
    def _row_text(entry: dict) -> str:
        kind = MODEL_KINDS.get(entry["kind"], {})
        scaler = entry.get("scaler", "none")
        return (f"{entry['name']}      [{kind.get('label', entry['kind'])}"
                + (f", {scaler} scaler]" if scaler != "none" else "]"))

    def _refresh_summary(self, problems=None) -> None:
        on = [m for m in self.spec["models"] if m["enabled"]]
        self.roster_info.setText(
            f"{len(on)} of {len(self.spec['models'])} will be trained"
            + ("   --   nothing enabled!" if not on else ""))
        self.roster_info.setStyleSheet(
            f"color: {COL_BAD if not on else COL_GOOD}; background: transparent;")

        vote = self.spec["voting"]
        if not vote["enabled"]:
            self.vote_info.setText("no ensemble -- each model stands alone")
        elif len(on) < 2:
            self.vote_info.setText(
                "at least two enabled models are needed for a vote; "
                "the ensemble will be skipped")
        else:
            self.vote_info.setText(
                f"Vote-{vote['mode']}-{len(on)} over "
                + ", ".join(m["name"] for m in on))

        self._refresh_auto_info()

        if problems:
            self.problem_label.setText("spec was corrected: "
                                       + "; ".join(problems[:4])
                                       + (" ..." if len(problems) > 4 else ""))
        else:
            self.problem_label.setText("")

    def _refresh_auto_info(self) -> None:
        """
        What AUTO mode will actually produce, and what that costs.

        Shown as a count rather than a promise: the number of candidates is
        the number of full cross-validations the next retrain will run, which
        is the thing that decides whether it takes a minute or an hour.
        """
        auto = self.spec.get("auto") or {}
        if not auto.get("enabled"):
            self.auto_info.setText(
                "off -- each model is trained once, with the values set above")
            self.auto_warning.setText("")
            return

        expanded, note = expand_auto_spec(self.spec, seed=0)
        candidates = [m for m in expanded["models"] if m["enabled"]]
        plain = [m for m in self.spec["models"] if m["enabled"]]
        tuned = sorted({m.get("auto_from") for m in candidates
                        if m.get("auto_from")})

        self.auto_info.setText(
            f"{len(candidates)} candidates from {len(plain)} enabled model(s)"
            + (f"; tuning {', '.join(tuned)}" if tuned else "")
            + f"   --   roughly {len(candidates) / max(1, len(plain)):.1f}x "
              "the training time of one pass"
        )
        self.auto_warning.setText(
            "Honest-result warning: picking the best of "
            f"{len(candidates)} candidates on the same folds makes the "
            "WINNER's score optimistic -- the more combinations, the more "
            "optimistic. Trust the leave-one-subject-out number in MODEL "
            "STATS, not the sweep's best row."
        )

    # -- the per-model form ---------------------------------------------------

    def _on_row_changed(self, row: int) -> None:
        while self.form_host.count():
            item = self.form_host.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self._param_widgets = {}

        if not (0 <= row < len(self.spec["models"])):
            self.detail_title.setText("")
            self.detail_note.setText("")
            return

        entry = self.spec["models"][row]
        schema = MODEL_KINDS.get(entry["kind"], {})
        self.detail_title.setText(
            f"{entry['name']}   --   {schema.get('label', entry['kind'])}")
        self.detail_note.setText(schema.get("note", ""))

        grid = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(grid)
        form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(7)
        form.setContentsMargins(4, 4, 4, 4)
        # Let a long hint wrap instead of forcing the dialog wider.
        form.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)

        name_edit = QtWidgets.QLineEdit(entry["name"])
        name_edit.setToolTip("How this candidate is named in the sweep "
                             "results and the saved report.")
        name_edit.editingFinished.connect(
            lambda: self._set_name(row, name_edit.text()))
        form.addRow(self._form_label("name"), name_edit)

        scaler_combo = QtWidgets.QComboBox()
        scaler_combo.addItems(list(MODEL_SCALERS))
        scaler_combo.setCurrentText(entry.get("scaler", "none"))
        scaler_combo.setToolTip(
            "Feature scaling before the classifier. Tree models do not need "
            "it; SVM, kNN, logistic regression and MLP very much do.")
        scaler_combo.currentTextChanged.connect(
            lambda text: self._set_scaler(row, text))
        form.addRow(self._form_label("scaler"), scaler_combo)

        space = model_search_space(entry["kind"])
        auto_box = QtWidgets.QCheckBox(
            f"include in AUTO mode ({len(space)} parameter(s) to try)"
            if space else "no search space for this kind")
        auto_box.setChecked(bool(entry.get("auto", True)) and bool(space))
        auto_box.setEnabled(bool(space))
        auto_box.setToolTip(
            "When AUTO MODE is on, generate combinations for this model. "
            "Switch it off to pin this one to the values below while the "
            "others are tuned."
            if space else
            "AUTO mode has nothing to vary for this kind.")
        auto_box.toggled.connect(
            lambda on: self._set_auto_for(row, on))
        form.addRow(self._form_label("auto-tune"), auto_box)
        if space:
            tried = make_label(
                "tries " + ";  ".join(
                    f"{param} = {', '.join(str(v) for v in values)}"
                    for param, values in sorted(space.items())),
                9, False, COL_TEXT_DIM)
            tried.setWordWrap(True)
            form.addRow("", tried)

        for param, rule in schema.get("params", {}).items():
            widget = self._build_param_widget(row, param, rule,
                                              entry["params"].get(param))
            form.addRow(self._form_label(param), widget)
            # The explanation goes on its own line. Putting it in the label --
            # which is what this did at first -- makes the label column as
            # wide as the longest sentence, and every row then collides with
            # the next one.
            if rule.get("note"):
                hint = make_label(rule["note"], 9, False, COL_TEXT_DIM)
                hint.setWordWrap(True)
                form.addRow("", hint)
            self._param_widgets[param] = widget

        self.form_host.addWidget(grid)
        self.form_host.addStretch(1)

    @staticmethod
    def _form_label(text: str):
        """A fixed-width field label, so every row lines up."""
        label = make_label(text, 11, False, COL_TEXT)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(130)
        return label

    def _build_param_widget(self, row: int, param: str, rule: dict, value):
        """One editor, chosen by the schema's type."""
        kind = rule.get("type")

        if kind == "choice":
            widget = QtWidgets.QComboBox()
            widget.addItems([str(v) for v in rule.get("values", [])])
            widget.setCurrentText(str(value))
            widget.currentTextChanged.connect(
                lambda text: self._set_param(row, param, text))
            return widget

        if kind == "int":
            widget = QtWidgets.QSpinBox()
            widget.setRange(int(rule.get("min", 0)), int(rule.get("max", 10**6)))
            widget.setSingleStep(int(rule.get("step", 1)))
            widget.setValue(int(value if value is not None else 0))
            widget.valueChanged.connect(
                lambda number: self._set_param(row, param, int(number)))
            return widget

        if kind == "float":
            widget = QtWidgets.QDoubleSpinBox()
            widget.setDecimals(int(rule.get("decimals", 3)))
            widget.setRange(float(rule.get("min", 0.0)),
                            float(rule.get("max", 10**6)))
            widget.setSingleStep(float(rule.get("step", 0.1)))
            widget.setValue(float(value if value is not None else 0.0))
            widget.valueChanged.connect(
                lambda number: self._set_param(row, param, float(number)))
            return widget

        if kind == "ints":
            widget = QtWidgets.QLineEdit(
                ", ".join(str(int(n)) for n in (value or [])))
            widget.setPlaceholderText("e.g. 256, 128")
            widget.setToolTip(
                f"One number per layer, comma separated. At most "
                f"{rule.get('max_items', 4)} layers, each "
                f"{rule.get('min', 1)}-{rule.get('max', 1024)} units.")
            widget.editingFinished.connect(
                lambda: self._set_param(row, param, widget.text()))
            return widget

        widget = QtWidgets.QLineEdit(str(value))             # pragma: no cover
        widget.editingFinished.connect(
            lambda: self._set_param(row, param, widget.text()))
        return widget

    # -- edits ----------------------------------------------------------------

    def _revalidate(self, refresh_form: bool = False) -> None:
        """
        Re-validate after every edit, so the dialog can never hold a spec the
        trainer would refuse. The current row is kept.
        """
        row = self.model_list.currentRow()
        self.spec, problems = validate_model_spec(self.spec)
        self._loading = True
        try:
            for index, entry in enumerate(self.spec["models"]):
                if index < self.model_list.count():
                    item = self.model_list.item(index)
                    item.setText(self._row_text(entry))
                    item.setCheckState(QtCore.Qt.Checked if entry["enabled"]
                                       else QtCore.Qt.Unchecked)
                    item.setForeground(QtGui.QColor(
                        COL_TEXT if entry["enabled"] else COL_TEXT_DIM))
        finally:
            self._loading = False
        self._refresh_summary(problems)
        if refresh_form:
            self._on_row_changed(row)

    def _set_param(self, row: int, param: str, value) -> None:
        if self._loading or not (0 <= row < len(self.spec["models"])):
            return
        self.spec["models"][row]["params"][param] = value
        self._revalidate()

    def _set_name(self, row: int, text: str) -> None:
        if self._loading or not (0 <= row < len(self.spec["models"])):
            return
        text = text.strip()
        if not text or text == self.spec["models"][row]["name"]:
            return
        self.spec["models"][row]["name"] = text
        self._revalidate(refresh_form=True)

    def _set_scaler(self, row: int, text: str) -> None:
        if self._loading or not (0 <= row < len(self.spec["models"])):
            return
        self.spec["models"][row]["scaler"] = text
        self._revalidate()

    def _on_item_changed(self, item) -> None:
        if self._loading:
            return
        row = self.model_list.row(item)
        if 0 <= row < len(self.spec["models"]):
            self.spec["models"][row]["enabled"] = \
                item.checkState() == QtCore.Qt.Checked
            self._revalidate()

    def _on_auto_changed(self, *_args) -> None:
        if self._loading:
            return
        self.spec["auto"] = {
            "enabled": self.auto_check.isChecked(),
            "max_per_model": int(self.auto_spin.value()),
        }
        self._revalidate()

    def _set_auto_for(self, row: int, enabled: bool) -> None:
        if self._loading or not (0 <= row < len(self.spec["models"])):
            return
        self.spec["models"][row]["auto"] = bool(enabled)
        self._revalidate()

    def _on_voting_changed(self, *_args) -> None:
        if self._loading:
            return
        self.spec["voting"] = {
            "enabled": self.vote_check.isChecked(),
            "mode": self.vote_mode.currentData() or "soft",
            "always": self.vote_always.isChecked(),
        }
        self._revalidate()

    def _set_all(self, enabled: bool) -> None:
        for entry in self.spec["models"]:
            entry["enabled"] = bool(enabled)
        self._reload_all()

    def add_model(self) -> None:
        kind = self.add_combo.currentData()
        schema = MODEL_KINDS[kind]
        existing = {m["name"] for m in self.spec["models"]}
        name = kind
        suffix = 2
        while name in existing:
            name = f"{kind}-{suffix}"
            suffix += 1
        self.spec["models"].append({
            "name": name, "kind": kind, "enabled": True,
            "scaler": schema.get("scaler", "none"),
            "params": {p: model_param_default(kind, p)
                       for p in schema["params"]},
        })
        self._reload_all()
        self.model_list.setCurrentRow(self.model_list.count() - 1)

    def remove_model(self) -> None:
        row = self.model_list.currentRow()
        if not (0 <= row < len(self.spec["models"])):
            return
        if len(self.spec["models"]) == 1:
            self.problem_label.setText(
                "the last model cannot be removed -- there would be nothing "
                "to train")
            return
        del self.spec["models"][row]
        self._reload_all()

    def reset_defaults(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self, "Reset model spec",
            "Go back to the built-in roster?\n\nThis discards the models and "
            "hyperparameters you have set here.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.spec = default_model_spec()
        self._reload_all()

    def save_spec(self) -> None:
        try:
            save_model_spec(self.spec_path, self.spec)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, "Model spec", f"Could not write the spec:\n{exc}")
            return
        self.saved = True
        self.accept()


class TrimDialog(QtWidgets.QDialog):
    """
    Inspect a recording and cut the useful part out of it.

        |  useless  |  gesture  |  useless  |
                    ^           ^
                  start        end

    The original Recording object is never modified. Accepting produces a NEW
    Recording via rec.slice(), which is saved as a separate *_trimmed.* file
    next to the raw one.

    With `extract_mode=True` it does a second, different job: carving MANY
    labelled clips out of one long free take. Each EXTRACT CLIP writes an
    ordinary stage-"raw" recording into its gesture folder and leaves the
    dialog open, so a five-minute take becomes twenty training recordings
    without reopening anything. The take itself is never modified either.
    """

    def __init__(self, rec: Recording, writer: DatasetWriter, parent=None,
                 extract_mode: bool = False):
        super().__init__(parent)
        self.rec = rec
        self.writer = writer
        self.result_recording = None
        self.notes = []                  # notes read from the sidecar
        self.inside_markers = []         # session markers inside this take
        self.notes_added = 0             # how many the operator added here
        self._note_lines = []            # [(plot, item)] drawn for markers

        # -- extract mode state
        self.extract_mode = bool(extract_mode)
        self.extracted = []              # clips written out of this take
        self._clip_bands = []            # [(plot, item)] drawn for clips

        if self.extract_mode:
            self.setWindowTitle(
                f"Cut clips out of -- {rec.label} take "
                f"({rec.duration:.0f} s)")
        else:
            self.setWindowTitle(f"Trim -- {rec.label} rep {rec.repetition}")
        # The eight channel rows plus the envelope need height; on a laptop
        # this comes out smaller and the plots simply get shorter rather than
        # the dialog running off the bottom of the screen.
        fit_to_screen(self, 1180, 820, min_width=700, min_height=480)

        self.fs = rec.fs
        self.t = rec.t_rel.astype(np.float64)
        self.signal = rec.emg.astype(np.float32)
        self.env = rec.envelope(TRIM_ENVELOPE_MS)
        self.env_mean = self.env.mean(axis=1)
        self.activity = rec.activity or rec.detect_activity()

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        root.addWidget(make_label(
            f"{rec.person_id} / {rec.session_id} / {rec.label}  "
            + (f"take" if self.extract_mode else f"rep {rec.repetition}")
            + f"  --  {rec.n_samples} samples, "
              f"{rec.duration:.2f} s @ {self.fs:.0f} Hz", 12, True
        ))

        self.glw = pg.GraphicsLayoutWidget()
        self.glw.ci.setSpacing(0)
        root.addWidget(self.glw, 1)

        lo, _ = emg_range(rec.meta.get("emg_mode", EMG_MODE))
        signed = lo < 0

        self.regions = []
        self.plots = []
        for i in range(N_CHANNELS):
            plot = self.glw.addPlot(row=i, col=0)
            plot.showGrid(x=True, y=True, alpha=0.12)
            plot.setLabel("left", CHANNEL_NAMES[i])
            plot.getAxis("left").setWidth(50)
            plot.getAxis("left").setTextPen(pg.mkPen(CHANNEL_COLORS[i]))
            plot.getAxis("bottom").setStyle(showValues=False)
            plot.getAxis("bottom").setHeight(2)
            if i:
                plot.setXLink(self.plots[0])
            curve = plot.plot(pen=pg.mkPen(CHANNEL_COLORS[i], width=1))
            curve.setDownsampling(auto=True, method="peak")
            curve.setData(self.t, self.signal[:, i])
            span = float(max(abs(self.signal[:, i]).max(), 1.0)) * 1.1
            plot.setYRange(-span if signed else 0, span, padding=0)

            region = pg.LinearRegionItem(brush=pg.mkBrush(56, 189, 248, 42),
                                         pen=pg.mkPen(56, 189, 248, 160))
            region.setZValue(-10)
            plot.addItem(region)
            self.regions.append(region)
            self.plots.append(plot)

        # Envelope row with the detection thresholds drawn in.
        self.env_plot = self.glw.addPlot(row=N_CHANNELS, col=0)
        self.env_plot.showGrid(x=True, y=True, alpha=0.12)
        self.env_plot.setLabel("left", "RMS")
        self.env_plot.setLabel("bottom", "time", units="s")
        self.env_plot.getAxis("left").setWidth(50)
        self.env_plot.setXLink(self.plots[0])
        self.env_plot.setMaximumHeight(130)
        self.env_plot.plot(self.t, self.env_mean,
                           pen=pg.mkPen("#e6e9ef", width=1))
        for value, color, name in (
            (self.activity.threshold_high, "#ef4444", "high"),
            (self.activity.threshold_low, "#f59e0b", "low"),
            (self.activity.baseline, "#64748b", "baseline"),
        ):
            line = pg.InfiniteLine(pos=value, angle=0,
                                   pen=pg.mkPen(color, style=QtCore.Qt.DashLine))
            line.setToolTip(f"{name} = {value:.1f}")
            self.env_plot.addItem(line)
        self.env_region = pg.LinearRegionItem(brush=pg.mkBrush(56, 189, 248, 42),
                                              pen=pg.mkPen(56, 189, 248, 160))
        self.env_plot.addItem(self.env_region)
        self.regions.append(self.env_region)

        # Label window (what the operator originally cued) as a reference.
        if rec.label_end_index > rec.label_start_index:
            for plot in self.plots[:1]:
                cue = pg.LinearRegionItem(
                    values=(float(self.t[rec.label_start_index]),
                            float(self.t[min(rec.label_end_index,
                                             rec.n_samples - 1)])),
                    movable=False,
                    brush=pg.mkBrush(34, 197, 94, 26),
                    pen=pg.mkPen(34, 197, 94, 90),
                )
                cue.setZValue(-30)
                plot.addItem(cue)

        # ---- controls --------------------------------------------------------
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(make_label("Start (s)", 11, False, COL_TEXT_DIM))
        self.start_spin = QtWidgets.QDoubleSpinBox()
        self.start_spin.setDecimals(3)
        self.start_spin.setSingleStep(0.01)
        self.start_spin.setRange(0.0, float(self.t[-1]) if self.t.size else 0.0)
        controls.addWidget(self.start_spin)

        controls.addWidget(make_label("End (s)", 11, False, COL_TEXT_DIM))
        self.end_spin = QtWidgets.QDoubleSpinBox()
        self.end_spin.setDecimals(3)
        self.end_spin.setSingleStep(0.01)
        self.end_spin.setRange(0.0, float(self.t[-1]) if self.t.size else 0.0)
        controls.addWidget(self.end_spin)

        self.info = make_label("", 11, False, COL_TEXT_DIM)
        controls.addWidget(self.info, 1)

        auto = QtWidgets.QPushButton("Auto-detect")
        auto.setToolTip("Run the configured EMG activity detector")
        auto.clicked.connect(self.apply_auto)
        controls.addWidget(auto)

        full = QtWidgets.QPushButton("Full range")
        full.clicked.connect(self.apply_full)
        controls.addWidget(full)

        preview = QtWidgets.QPushButton("Preview selection")
        preview.clicked.connect(self.preview_selection)
        controls.addWidget(preview)

        root.addLayout(controls)

        # ---- extract clips ---------------------------------------------------
        if self.extract_mode:
            root.addWidget(self._build_extract_panel())

        # ---- notes -----------------------------------------------------------
        # Annotating a take that is already on disk. The signal file is
        # immutable, so a note goes to a *_notes.json sidecar beside it.
        if MARKERS_ENABLED:
            root.addWidget(self._build_notes_panel())

        if self.extract_mode:
            note = make_label(
                "The take is never modified. Each EXTRACT CLIP writes a new "
                "recording into its gesture folder -- that is the training "
                "data. Cut as many clips as you like, then press DONE.",
                11, False, COL_WARN
            )
        else:
            note = make_label(
                "The raw recording is never modified. Saving writes a separate "
                "*_trimmed.csv / *_trimmed.npz next to it. Notes go to a separate "
                "*_notes.json and never touch the signal either.", 11, False, COL_WARN
            )
        note.setWordWrap(True)
        root.addWidget(note)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        if self.extract_mode:
            # No "save trimmed copy" here: a take is not something you trim
            # down to one useful part, it is something you harvest many parts
            # from, and each of those is already written by EXTRACT CLIP.
            done = QtWidgets.QPushButton("DONE")
            done.setObjectName("primary")
            done.setDefault(True)
            done.clicked.connect(self.accept)
            buttons.addWidget(done)
        else:
            cancel = QtWidgets.QPushButton("Cancel")
            cancel.clicked.connect(self.reject)
            save = QtWidgets.QPushButton("Save trimmed copy")
            save.setObjectName("primary")
            save.setDefault(True)
            save.clicked.connect(self.save_trimmed)
            buttons.addWidget(cancel)
            buttons.addWidget(save)
        root.addLayout(buttons)

        # ---- wiring ----------------------------------------------------------
        self._syncing = False
        for region in self.regions:
            region.sigRegionChanged.connect(self._on_region_changed)
        self.start_spin.valueChanged.connect(self._on_spin_changed)
        self.end_spin.valueChanged.connect(self._on_spin_changed)

        if self.extract_mode:
            # Start at the beginning with one clip-length window: carving a
            # long take is a sequential job, and auto-detect would jump to
            # whichever burst happens to be strongest.
            self.select_first_clip()
        elif self.activity.found:
            self.apply_auto()
        else:
            self.apply_full()

    # -- extract mode ---------------------------------------------------------

    def _build_extract_panel(self):
        """Gesture, clip length, stepping, and the list of clips written."""
        group = QtWidgets.QGroupBox("EXTRACT TRAINING CLIPS")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(SP_SM)
        group.setMaximumHeight(300)

        # Where you are in the take, in words. Carving a five-minute take is a
        # sequential job and the one thing the old panel never said was "how
        # far along am I".
        self.position_label = make_label("", FS_MD, True, COL_ACCENT)
        self.position_label.setWordWrap(True)
        layout.addWidget(self.position_label)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(make_label("Gesture", 11, False, COL_TEXT_DIM))
        self.extract_gesture = QtWidgets.QComboBox()
        self.extract_gesture.addItems(GESTURES)
        if self.rec.label in GESTURES:
            self.extract_gesture.setCurrentText(self.rec.label)
        self.extract_gesture.setToolTip(
            "What the next clip will be labelled as. Change it as often as "
            "you like -- one take can yield clips of several gestures.")
        row.addWidget(self.extract_gesture)

        row.addWidget(make_label("Clip length (s)", 11, False, COL_TEXT_DIM))
        self.clip_length_spin = QtWidgets.QDoubleSpinBox()
        self.clip_length_spin.setDecimals(2)
        self.clip_length_spin.setSingleStep(0.25)
        self.clip_length_spin.setRange(LABEL_MIN_SELECTION_S,
                                       max(LABEL_MIN_SELECTION_S,
                                           float(self.t[-1]) if self.t.size else 1.0))
        self.clip_length_spin.setValue(
            min(float(LABEL_DEFAULT_SELECTION_S),
                self.clip_length_spin.maximum()))
        self.clip_length_spin.setToolTip(
            "The default cut length, in seconds. Mirrored into "
            "LABEL_DEFAULT_SELECTION_S, so it is also what SELECT LAST n s "
            "uses on the live graph and what gets written into every "
            "*_meta.json.")
        self.clip_length_spin.valueChanged.connect(self._on_clip_length_changed)
        row.addWidget(self.clip_length_spin)

        fit = QtWidgets.QPushButton("FIT LENGTH")
        fit.setToolTip("Resize the selection to exactly the clip length, "
                       "keeping its left edge.")
        fit.clicked.connect(self.fit_selection_to_length)
        row.addWidget(fit)
        row.addStretch(1)
        layout.addLayout(row)

        row2 = QtWidgets.QHBoxLayout()
        first = QtWidgets.QPushButton("|<  FIRST")
        first.setMinimumHeight(H_CONTROL)
        first.clicked.connect(self.select_first_clip)
        row2.addWidget(first)

        prev_btn = QtWidgets.QPushButton("<<  PREV")
        prev_btn.setMinimumHeight(H_CONTROL)
        prev_btn.setToolTip(
            "Step the selection one clip length backwards.  Ctrl+Left")
        prev_btn.clicked.connect(lambda: self.step_selection(-1))
        row2.addWidget(prev_btn)

        next_btn = QtWidgets.QPushButton("NEXT  >>")
        next_btn.setMinimumHeight(H_CONTROL)
        next_btn.setToolTip(
            "Step the selection one clip length forwards. This is how you "
            "walk through a long take.  Ctrl+Right")
        next_btn.clicked.connect(lambda: self.step_selection(+1))
        row2.addWidget(next_btn)

        row2.addStretch(1)
        layout.addLayout(row2)

        # The one action that writes data gets its own row and twice the
        # height: in a dialog of eight buttons, the one you press twenty times
        # should not look like the one you press once.
        self.extract_button = QtWidgets.QPushButton(
            "EXTRACT CLIP      Ctrl+Space")
        self.extract_button.setObjectName("primary")
        self.extract_button.setMinimumHeight(H_ACTION)
        self.extract_button.setToolTip(
            "Write the selected span as a labelled training recording in its "
            "gesture folder, then step forward to the next span.")
        self.extract_button.clicked.connect(self.extract_clip)
        layout.addWidget(self.extract_button)

        self.clip_list = QtWidgets.QListWidget()
        self.clip_list.setMaximumHeight(110)
        self.clip_list.setToolTip(
            "Clips written out of this take, in the order you cut them. Each "
            "is its own recording -- they are never joined together.")
        layout.addWidget(self.clip_list)

        self.extract_status = make_label(
            "nothing extracted yet", 10, False, COL_TEXT_DIM)
        self.extract_status.setWordWrap(True)
        layout.addWidget(self.extract_status)

        self._extract_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Space"), self)
        self._extract_shortcut.activated.connect(self.extract_clip)
        self._next_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Right"), self)
        self._next_shortcut.activated.connect(lambda: self.step_selection(+1))
        self._prev_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Left"), self)
        self._prev_shortcut.activated.connect(lambda: self.step_selection(-1))

        self._refresh_clip_list()
        return group

    def _on_clip_length_changed(self, value: float) -> None:
        """The spin box is the authority for the default cut length."""
        globals()["LABEL_DEFAULT_SELECTION_S"] = float(value)

    @property
    def clip_length(self) -> float:
        spin = getattr(self, "clip_length_spin", None)
        return float(spin.value()) if spin is not None else float(LABEL_DEFAULT_SELECTION_S)

    def select_first_clip(self) -> None:
        if not self.t.size:
            return
        t0 = float(self.t[0])
        self._set_selection(t0, min(float(self.t[-1]), t0 + self.clip_length))

    def fit_selection_to_length(self) -> None:
        if not self.t.size:
            return
        t0 = float(self.start_spin.value())
        self._set_selection(t0, min(float(self.t[-1]), t0 + self.clip_length))

    def step_selection(self, direction: int) -> None:
        """
        Slide the selection one clip length along, keeping its width.

        Walking a long take is the common case, so this is bound to
        Ctrl+Left / Ctrl+Right as well as the buttons.
        """
        if not self.t.size:
            return
        width = max(self.clip_length, float(self.end_spin.value()
                                            - self.start_spin.value()))
        shift = float(direction) * width
        t0 = float(self.start_spin.value()) + shift
        t0 = float(np.clip(t0, self.t[0], max(self.t[0], self.t[-1] - width)))
        self._set_selection(t0, min(float(self.t[-1]), t0 + width))
        self.preview_selection()

    def _refresh_clip_list(self) -> None:
        self.clip_list.clear()
        if not self.extracted:
            item = QtWidgets.QListWidgetItem(
                "no clips extracted from this take yet")
            item.setForeground(QtGui.QColor(COL_TEXT_DIM))
            self.clip_list.addItem(item)
            return
        for clip in self.extracted:
            start = clip.meta.get("take_start_s", 0.0)
            end = clip.meta.get("take_end_s", 0.0)
            item = QtWidgets.QListWidgetItem(
                f"{clip.label}  rep {clip.repetition:03d}   "
                f"{start:.2f}-{end:.2f} s   {clip.n_samples} samples"
                + (f"   [{clip.worst_issue_level}]"
                   if clip.worst_issue_level else ""))
            item.setForeground(QtGui.QColor(
                LABEL_COLORS.get(clip.label, COL_GOOD)))
            self.clip_list.addItem(item)
        self.clip_list.scrollToBottom()

    def _draw_clip_bands(self) -> None:
        """
        Show which parts of the take have already been harvested, so you can
        see at a glance what is left.
        """
        for plot, item in self._clip_bands:
            plot.removeItem(item)
        self._clip_bands = []

        for clip in self.extracted:
            start = float(clip.meta.get("take_start_s", 0.0))
            end = float(clip.meta.get("take_end_s", 0.0))
            colour = QtGui.QColor(LABEL_COLORS.get(clip.label, COL_GOOD))
            brush = pg.mkBrush(colour.red(), colour.green(), colour.blue(), 40)
            pen = pg.mkPen(colour.name(), width=1)
            for plot in (self.plots[0], self.env_plot):
                band = pg.LinearRegionItem(values=(start, end), movable=False,
                                           brush=brush, pen=pen)
                band.setZValue(-25)
                plot.addItem(band)
                self._clip_bands.append((plot, band))

            caption = pg.TextItem(clip.label, color=colour.name(), anchor=(0, 1))
            caption.setFont(QtGui.QFont("Segoe UI", 8, QtGui.QFont.Bold))
            caption.setPos(start, 0)
            self.plots[0].addItem(caption)
            self._clip_bands.append((self.plots[0], caption))

    def extract_clip(self):
        """
        Write the selected span as a labelled training recording.

        The clip gets a NEW uid rather than inheriting the take's: uid is what
        pairs a raw file with its trimmed copies, and twenty clips sharing the
        take's uid would look like twenty versions of one recording. The link
        back to the take is kept in the metadata instead.
        """
        if not self.extract_mode:
            return None

        start, end = self.selection_indices()
        n = end - start
        if not self.t.size or n < 4:
            self._set_extract_status(
                f"selection is too short ({n} samples)", COL_BAD)
            return None

        t_start = float(self.t[start])
        t_end = float(self.t[min(end - 1, self.t.size - 1)])
        duration = t_end - t_start
        if duration < LABEL_MIN_SELECTION_S:
            self._set_extract_status(
                f"selection is only {duration:.2f} s -- the minimum is "
                f"{LABEL_MIN_SELECTION_S:.2f} s", COL_BAD)
            return None

        label = self.extract_gesture.currentText()
        parent = self.rec

        # Subject metadata comes from the TAKE, not from today's config: the
        # handedness and electrode placement that matter are the ones in force
        # when the signal was recorded, which may have been weeks ago.
        parent_meta = dict(parent.meta or {})
        meta = {
            "emg_mode": parent_meta.get("emg_mode", EMG_MODE),
            "handedness": parent_meta.get("handedness", HANDEDNESS),
            "arm_worn_on": parent_meta.get("arm_worn_on", ARM_WORN_ON),
            "device": parent_meta.get("device", DEVICE_NAME),
            "notes": parent_meta.get("notes", ""),
            "capture_style": "cut_from_take",
            "selected_by_operator": True,
            "selection_seconds": round(duration, 4),
            "parent_take_uid": parent.uid,
            "parent_take_label": parent.label,
            "parent_take_stage": parent.stage,
            "parent_take_file": (parent.files.get(f"{STAGE_TAKE}:npz")
                                 or parent.files.get(f"{STAGE_TAKE}:csv")
                                 or parent.source),
            "parent_take_n_samples": parent.n_samples,
            "take_start_index": int(start),
            "take_end_index": int(end),
            "take_start_s": round(t_start, 4),
            "take_end_s": round(t_end, 4),
        }

        # Markers inside the clip travel with it, exactly as they do for a
        # span cut from the live ring buffer.
        abs_start = float(parent.timestamps[start])
        abs_end = float(parent.timestamps[min(end - 1, parent.n_samples - 1)])
        inside = markers_between(self.inside_markers, abs_start, abs_end)
        if inside:
            meta["markers"] = [m.to_dict() for m in inside]

        repetition = self.writer.next_repetition(label)
        clip = Recording(
            emg=parent.emg[start:end].copy(),
            timestamps=parent.timestamps[start:end].copy(),
            label=label,
            person_id=parent.person_id,
            session_id=parent.session_id,
            repetition=repetition,
            stage=STAGE_RAW,
            # The operator drew the boundaries, so the whole clip is the
            # gesture: there is no pre/post roll to exclude.
            label_start_index=0,
            label_end_index=int(n),
            meta=meta,
        )
        clip.run_quality_checks(expected_seconds=duration)
        if AUTO_TRIM:
            clip.detect_activity()

        try:
            written = self.writer.save(clip, STAGE_RAW)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, "Extract clip",
                f"Could not write the clip:\n{exc}")
            return None

        self.extracted.append(clip)
        self._refresh_clip_list()
        self._refresh_position()
        self._draw_clip_bands()
        self._set_extract_status(
            f"wrote {os.path.basename(written.get('csv') or written.get('npz'))}"
            f"  --  {len(self.extracted)} clip(s) from this take",
            COL_GOOD)

        # Step forward so the next press of NEXT/EXTRACT continues where this
        # clip ended rather than re-cutting the same span.
        self.step_selection(+1)
        return clip

    def _set_extract_status(self, text: str, color: str) -> None:
        self.extract_status.setText(text)
        self.extract_status.setStyleSheet(
            f"color: {color}; background: transparent;")

    # -- notes ----------------------------------------------------------------

    def _build_notes_panel(self):
        """Existing notes for this take, plus a field to add another."""
        group = QtWidgets.QGroupBox("NOTES ON THIS RECORDING")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)
        group.setMaximumHeight(190)

        self.notes_list = QtWidgets.QListWidget()
        self.notes_list.setMaximumHeight(86)
        layout.addWidget(self.notes_list)

        row = QtWidgets.QHBoxLayout()
        self.note_input = QtWidgets.QLineEdit()
        self.note_input.setPlaceholderText(
            "a note about this recording, e.g. subject moved the wrist")
        self.note_input.returnPressed.connect(self.add_note)
        row.addWidget(self.note_input, 1)

        add = QtWidgets.QPushButton("ADD NOTE")
        add.setObjectName("primary")
        add.setToolTip(
            "Saved immediately to the *_notes.json sidecar. You do not have "
            "to press Save for a note to persist."
        )
        add.clicked.connect(self.add_note)
        row.addWidget(add)
        layout.addLayout(row)

        self.notes_status = make_label("", 10, False, COL_TEXT_DIM)
        self.notes_status.setWordWrap(True)
        layout.addWidget(self.notes_status)

        self._reload_notes()
        return group

    def _reload_notes(self) -> None:
        """Re-read the sidecar and redraw both the list and the graph lines."""
        path = self.rec.primary_path(self.writer.root)
        self.notes = load_recording_notes(path) if path else []

        # Session markers that fall inside this take are part of its story
        # too, so they are shown here even though they live in the session
        # log rather than in this recording's sidecar.
        try:
            session = self.writer.load_markers()
        except OSError:                                      # pragma: no cover
            session = []
        t0 = float(self.rec.timestamps[0]) if self.rec.n_samples else 0.0
        t1 = float(self.rec.timestamps[-1]) if self.rec.n_samples else 0.0

        # Prefer the markers recorded in the metadata at cut time. They were
        # computed from the span's real boundaries; recomputing from the first
        # and last SAMPLE drops a marker that sits exactly on the boundary,
        # which is precisely the case for CUT SINCE LAST MARKER -- the marker
        # it cut from would vanish from its own recording.
        stored = self.rec.meta.get("markers")
        recomputed = markers_between(session, t0, t1)
        if isinstance(stored, list) and stored:
            by_uid = {}
            for entry in stored:
                if isinstance(entry, dict):
                    try:
                        marker = Marker.from_dict(entry)
                    except (TypeError, ValueError):          # pragma: no cover
                        continue
                    by_uid[marker.uid] = marker
            for marker in recomputed:
                by_uid.setdefault(marker.uid, marker)
            self.inside_markers = sorted(by_uid.values(),
                                         key=lambda m: m.t_unix)
        else:
            self.inside_markers = recomputed

        self.notes_list.clear()
        if not self.notes and not self.inside_markers:
            item = QtWidgets.QListWidgetItem("no notes on this recording yet")
            item.setForeground(QtGui.QColor(COL_TEXT_DIM))
            self.notes_list.addItem(item)
        for marker in self.inside_markers:
            stamp = f"{marker.t_unix - t0:+.2f} s"
            item = QtWidgets.QListWidgetItem(
                f"| {stamp}  (session marker)  {marker.short_text(44)}")
            item.setForeground(QtGui.QColor(
                MARKER_COLORS.get(marker.kind, MARKER_COLORS["note"])))
            item.setToolTip(marker.text or "(no text)")
            self.notes_list.addItem(item)
        for marker in self.notes:
            stamp = datetime.fromtimestamp(marker.created_unix).strftime("%Y-%m-%d %H:%M")
            item = QtWidgets.QListWidgetItem(f"- {stamp}  {marker.short_text(52)}")
            item.setToolTip(marker.text or "(no text)")
            self.notes_list.addItem(item)
        self.notes_list.scrollToBottom()
        self._draw_note_lines(t0)

    def _draw_note_lines(self, t0: float) -> None:
        """
        Vertical separators where a marker was dropped inside this take.

        Created once per reload rather than pooled: this dialog is modal and
        static, so there is no frame budget to protect here.
        """
        # Each item is remembered together with the plot that owns it, so
        # removal is exact rather than a guess about which scene it is in.
        for plot, item in getattr(self, "_note_lines", []):
            plot.removeItem(item)
        self._note_lines = []

        for marker in self.inside_markers:
            x = float(marker.t_unix - t0)
            colour = MARKER_COLORS.get(marker.kind, MARKER_COLORS["note"])
            for plot in self.plots:
                line = pg.InfiniteLine(
                    pos=x, angle=90, movable=False,
                    pen=pg.mkPen(colour, width=2 if marker.is_break else 1,
                                 style=QtCore.Qt.DashLine if marker.is_break
                                 else QtCore.Qt.SolidLine),
                )
                line.setZValue(-8)
                line.setToolTip(marker.text or marker.kind)
                plot.addItem(line)
                self._note_lines.append((plot, line))

            caption = pg.TextItem(marker.short_text(24), color=colour,
                                  anchor=(0, 0))
            caption.setFont(QtGui.QFont("Segoe UI", 8, QtGui.QFont.Bold))
            caption.setPos(x, 0)
            self.plots[0].addItem(caption)
            self._note_lines.append((self.plots[0], caption))

    def add_note(self) -> None:
        text = self.note_input.text().strip()
        if not text:
            self.notes_status.setText("type the note first")
            self.notes_status.setStyleSheet(
                f"color: {COL_BAD}; background: transparent;")
            self.note_input.setFocus()
            return
        try:
            path, _marker = self.writer.add_note_to_recording(self.rec, text)
        except (OSError, ValueError) as exc:
            QtWidgets.QMessageBox.critical(
                self, "Add note", f"Could not write the note:\n{exc}")
            return

        self.notes_added = getattr(self, "notes_added", 0) + 1
        self.note_input.clear()
        self.notes_status.setText(f"saved to {os.path.basename(path)}")
        self.notes_status.setStyleSheet(
            f"color: {COL_GOOD}; background: transparent;")
        self._reload_notes()

    # -- selection ------------------------------------------------------------

    def _index_of(self, seconds: float) -> int:
        if self.t.size == 0:
            return 0
        return int(np.clip(np.searchsorted(self.t, seconds), 0, self.t.size - 1))

    def _set_selection(self, t0: float, t1: float) -> None:
        if self.t.size == 0:
            return
        t0, t1 = float(min(t0, t1)), float(max(t0, t1))
        t0 = float(np.clip(t0, self.t[0], self.t[-1]))
        t1 = float(np.clip(t1, self.t[0], self.t[-1]))
        if t1 - t0 < 1.0 / max(self.fs, 1.0):
            t1 = min(self.t[-1], t0 + 1.0 / max(self.fs, 1.0))

        self._syncing = True
        try:
            for region in self.regions:
                region.setRegion((t0, t1))
            self.start_spin.setValue(t0)
            self.end_spin.setValue(t1)
        finally:
            self._syncing = False
        self._update_info()

    def selection_indices(self):
        start = self._index_of(self.start_spin.value())
        end = min(self.rec.n_samples, self._index_of(self.end_spin.value()) + 1)
        return start, max(start + 1, end)

    def _on_region_changed(self, region) -> None:
        if self._syncing:
            return
        t0, t1 = region.getRegion()
        self._set_selection(t0, t1)

    def _on_spin_changed(self, _value) -> None:
        if self._syncing:
            return
        self._set_selection(self.start_spin.value(), self.end_spin.value())

    def _refresh_position(self) -> None:
        """Where the selection sits in the take, and how much is left."""
        if not getattr(self, "extract_mode", False):
            return
        label = getattr(self, "position_label", None)
        if label is None or not self.t.size:
            return
        t0 = float(self.start_spin.value())
        t1 = float(self.end_spin.value())
        total = float(self.t[-1])
        done = len(getattr(self, "extracted", []))
        label.setText(
            f"at {t0:.1f}-{t1:.1f} s of {total:.1f} s "
            f"({t0 / max(total, 0.001) * 100:.0f} % through the take)"
            + (f"   --   {done} clip(s) cut so far" if done else
               "   --   nothing cut yet"))

    def _update_info(self) -> None:
        self._refresh_position()
        start, end = self.selection_indices()
        kept = end - start
        self.info.setText(
            f"keeping {kept} / {self.rec.n_samples} samples "
            f"({kept / max(1, self.rec.n_samples) * 100:.0f} %), "
            f"{kept / max(self.fs, 1.0):.2f} s  --  "
            f"removing {self.rec.n_samples - kept} samples"
        )

    # -- actions --------------------------------------------------------------

    def apply_auto(self) -> None:
        result = self.rec.detect_activity()
        self.activity = result
        if not result.found:
            QtWidgets.QMessageBox.information(
                self, "Auto-detect",
                "No clear EMG activity was found.\n\n"
                f"Reason: {result.reason}\n\n"
                "Adjust TRIM_THRESHOLD_HIGH / TRIM_ABS_FLOOR in the config, "
                "or select the region by hand.",
            )
            return
        self._method = "auto"
        self._set_selection(float(self.t[result.start]),
                            float(self.t[min(result.end, self.t.size - 1)]))

    def apply_full(self) -> None:
        self._method = "manual"
        if self.t.size:
            self._set_selection(float(self.t[0]), float(self.t[-1]))

    def preview_selection(self) -> None:
        t0, t1 = self.start_spin.value(), self.end_spin.value()
        pad = max(0.05, (t1 - t0) * 0.15)
        self.plots[0].setXRange(t0 - pad, t1 + pad, padding=0)

    def save_trimmed(self) -> None:
        start, end = self.selection_indices()
        if end - start < 2:
            QtWidgets.QMessageBox.warning(self, "Trim",
                                          "The selection is too short.")
            return

        trimmed = self.rec.slice(start, end, STAGE_TRIMMED, {
            "trim_method": getattr(self, "_method", "manual"),
            "trim_dialog": True,
        })
        try:
            written = self.writer.save(trimmed, STAGE_TRIMMED)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Trim",
                                           f"Could not write the trimmed file:\n{exc}")
            return

        self.result_recording = trimmed
        QtWidgets.QMessageBox.information(
            self, "Trim saved",
            "Trimmed copy written:\n\n"
            + "\n".join(os.path.basename(p) for p in written.values())
            + "\n\nThe raw recording is unchanged.",
        )
        self.accept()


# ------------------------------------------------------------------------------
# Background workers
# ------------------------------------------------------------------------------


class DatasetExportWorker(QtCore.QThread):
    """
    Builds a combined, ML-ready bundle from everything on disk.

    Runs off the GUI thread because it walks the whole dataset tree. It only
    READS the dataset -- export can never damage a recording.
    """
    progress = QtCore.Signal(str)
    done = QtCore.Signal(str)
    failed = QtCore.Signal(str)

    def __init__(self, root: str, out_dir: str, stage: str = "prefer_trimmed",
                 parent=None):
        super().__init__(parent)
        self.root = root
        self.out_dir = out_dir
        self.stage = stage

    def run(self) -> None:
        try:
            self.progress.emit("Scanning the dataset...")
            recs = load_dataset(self.root, stage=self.stage, prefer="npz")
            if not recs:
                self.failed.emit("No recordings found to export.")
                return

            self.progress.emit(f"Windowing {len(recs)} recordings...")
            X, y, groups, meta = make_windows(recs)
            if X.shape[0] == 0:
                self.failed.emit(
                    "No windows could be built. WINDOW_MS is probably longer "
                    "than the recordings."
                )
                return

            self.progress.emit(f"Extracting features for {X.shape[0]} windows...")
            features, names = make_feature_matrix(X, meta["fs"])

            ensure_dir(self.out_dir)
            stamp = timestamp_slug()
            bundle = unique_path(os.path.join(self.out_dir, f"dataset_{stamp}.npz"))

            self.progress.emit("Writing the bundle...")
            np.savez_compressed(
                bundle,
                X=X.astype(np.float32),
                y=y.astype(np.int16),
                groups=np.asarray(groups, dtype=str),
                label_names=np.asarray(meta["labels"], dtype=str),
                features=features,
                feature_names=np.asarray(names, dtype=str),
                window_samples=np.array(meta["window_samples"]),
                step=np.array(meta["step"]),
                fs=np.array(meta["fs"]),
                channel_names=np.asarray(CHANNEL_NAMES, dtype=str),
            )

            index_path = os.path.splitext(bundle)[0] + "_index.csv"
            with open(index_path, "w", newline="", encoding="utf-8") as fh:
                writer = csv.DictWriter(
                    fh,
                    fieldnames=["uid", "person_id", "session_id", "label",
                                "repetition", "window_index", "source"],
                )
                writer.writeheader()
                writer.writerows(meta["records"])

            summary_path = os.path.splitext(bundle)[0] + "_summary.json"
            persons = sorted(set(np.asarray(groups, dtype=str).tolist()))
            with open(summary_path, "w", encoding="utf-8") as fh:
                json.dump({
                    "created_iso": iso_time(time.time()),
                    "stage": self.stage,
                    "n_recordings": len(recs),
                    "n_windows": int(X.shape[0]),
                    "window_samples": int(meta["window_samples"]),
                    "step": int(meta["step"]),
                    "fs": float(meta["fs"]),
                    "labels": meta["labels"],
                    "persons": persons,
                    "windows_per_label": {
                        name: int((y == i).sum())
                        for i, name in enumerate(meta["labels"])
                    },
                    "windows_per_person": {
                        p: int((np.asarray(groups, dtype=str) == p).sum())
                        for p in persons
                    },
                    "validation_note": (
                        "Use the groups array for leave-one-person-out or "
                        "GroupKFold. Do NOT split windows randomly: windows "
                        "from one recording overlap, so a random split leaks "
                        "the test set into training."
                    ),
                    "config": recording_config_snapshot(),
                }, fh, indent=2, default=str)

            self.done.emit(
                f"Exported {X.shape[0]} windows from {len(recs)} recordings "
                f"({len(persons)} person(s)).\n\n{bundle}"
            )
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}")


@dataclass
class TrailStep:
    """
    What every candidate said about ONE feature vector, and how it fused.

    `per_model` is aligned with `names`; each entry is that candidate's full
    probability vector over `labels`. `fused` is the soft vote actually
    produced by the stored ensemble, and `fused_from_mean` is the same thing
    recomputed as the plain mean of the members -- the view shows both so the
    fusion rule is demonstrated rather than asserted.
    """
    names: List[str]
    per_model: List[List[float]]
    labels: List[str]
    fused: Optional[List[float]]          # the ensemble's own output
    fused_from_mean: List[float]          # mean of the members, recomputed
    deployed_name: str                    # the model that actually predicts
    deployed: Optional[List[float]]       # its probabilities, or None
    source: str                           # "live" or "recorded"
    note: str = ""

    def decision(self, probabilities) -> int:
        return int(np.argmax(np.asarray(probabilities, dtype=float)))

    def label_of(self, probabilities) -> str:
        index = self.decision(probabilities)
        return (self.labels[index] if 0 <= index < len(self.labels)
                else str(index))

    @property
    def agreement(self) -> int:
        """How many candidates picked the same class as the soft vote."""
        if not self.per_model:
            return 0
        target = self.decision(self.fused_from_mean)
        return sum(1 for p in self.per_model if self.decision(p) == target)


class DecisionPanel:
    """
    The side file that lets MODEL STATS show every candidate's decision.

    Loaded lazily and entirely optional. With no panel file the view says so
    and shows nothing else -- it never falls back to a guess, and it never
    stands in for the deployed model, which keeps predicting from
    `gesture_model.joblib` exactly as before.

    Two sources of truth, kept apart on purpose:

      * `step_for_features()` runs the stored members on a LIVE feature
        vector. Those members were fitted on the whole dataset, so what they
        agree about live is a demonstration, not a score. Every caller gets
        that in `note`.
      * `recorded_step()` reads the out-of-fold trail captured during
        training. Those ARE honest numbers, and they come with the fold the
        window was held out in.

    The live path takes the feature vector LivePredictor already computed, so
    there is exactly one feature function in play. Recomputing features here
    would create a second path that could silently drift -- the same reason
    `model_feature_vector()` is shared with the trainer.
    """

    def __init__(self, path: str = None, model_path: str = None):
        self.path = path or panel_path(model_path)
        self.loaded = False
        self.status = "not loaded"
        self.names: List[str] = []
        self.vote_name: Optional[str] = None
        self.labels: List[str] = []
        self.members: dict = {}
        self.cv: dict = {}
        self.setup: dict = {}
        self.fusion: dict = {}
        self.protocol: dict = {}
        self.recordings: List[dict] = []
        self.space: dict = {}
        self.created_iso = ""
        self.winner = ""
        self.n_windows = 0
        self.n_recordings = 0
        # First-call warm-up state. See warm_up().
        self.warming = False
        self.warm = False
        self.warm_ms = 0.0
        self._trail: dict = {}
        self._mtime = 0.0

    # -- loading --------------------------------------------------------------

    def load(self) -> bool:
        """Read the panel. False (with a reason in `status`) is normal."""
        self.loaded = False
        if not ENABLE_DECISION_PANEL:
            self.status = "disabled in the configuration"
            return False
        if not os.path.exists(self.path):
            self.status = ("no decision panel yet -- retrain to build one "
                           "(train_model.py writes it next to the model)")
            return False
        try:
            import joblib
        except ImportError:
            self.status = "joblib is not installed"
            return False
        try:
            payload = joblib.load(self.path)
        except Exception as exc:
            self.status = f"could not load the decision panel: {exc}"
            return False
        if not isinstance(payload, dict) or payload.get("kind") != "decision_panel":
            self.status = "that file is not a decision panel"
            return False

        self.names = list(payload.get("member_order") or [])
        self.vote_name = payload.get("vote_name")
        self.labels = list(payload.get("labels") or [])
        self.members = dict(payload.get("members") or {})
        self.cv = dict(payload.get("cv") or {})
        self.setup = dict(payload.get("setup") or {})
        self.fusion = dict(payload.get("fusion") or {})
        self.protocol = dict(payload.get("protocol") or {})
        self.recordings = list(payload.get("recordings") or [])
        self.space = dict(payload.get("space") or {})
        self.created_iso = payload.get("created_iso", "")
        self.winner = payload.get("winner", "")
        self.n_windows = int(payload.get("n_windows", 0) or 0)
        self.n_recordings = int(payload.get("n_recordings",
                                            len(self.recordings)) or 0)
        self._trail = dict(payload.get("trail") or {})
        self._mtime = os.path.getmtime(self.path)

        if not self.names or not self.members:
            self.status = "the decision panel is empty"
            return False
        self._serialise_members()
        self.loaded = True
        self.status = "ready"
        return True

    def _serialise_members(self) -> None:
        """
        Force the members to predict on one thread.

        Measured on this machine: running the seven candidates over a single
        live window took 2.7 s with the forests left at `n_jobs=-1`, and
        0.03 s with them at 1. The work is one row through 400 trees -- a few
        hundred microseconds -- so the entire 2.7 s was joblib standing up a
        worker pool, twice, for every window. At 7 predictions a second the
        view could not keep up and the GUI thread was the one paying.

        `n_jobs` affects only how a fitted forest is evaluated, never what it
        answers, so this changes no number on screen. It is applied on load
        rather than at training time because the saved estimator is also the
        one the sweep used, where many rows at once do want the cores.
        """
        def walk(estimator, depth=0):
            """Every estimator inside a pipeline or an ensemble."""
            if estimator is None or depth > 4:
                return
            if getattr(estimator, "n_jobs", None) not in (None, 1):
                try:
                    estimator.n_jobs = 1
                except Exception:                          # pragma: no cover
                    pass
            for step in (getattr(estimator, "named_steps", None) or {}).values():
                walk(step, depth + 1)
            #  VotingClassifier: `estimators_` is the fitted list, and it is
            #  the one that runs; `estimators` holds (name, estimator) pairs.
            for member in getattr(estimator, "estimators_", None) or []:
                walk(member, depth + 1)
            for entry in getattr(estimator, "estimators", None) or []:
                walk(entry[1] if isinstance(entry, tuple) else entry,
                     depth + 1)

        for model in self.members.values():
            walk(model)

    @property
    def ready(self) -> bool:
        return self.loaded and bool(self.members)

    def warm_up(self) -> float:
        """
        Pay every candidate's first-call cost once, up front.

        Measured here: the second and later windows cost about 45 ms for all
        seven, but the FIRST costs ~2.5 s, essentially all of it inside
        HistGradientBoosting's initial `predict_proba`, which builds its
        prediction kernel on demand (19 ms RandomForest, 21 ms ExtraTrees,
        under 1 ms for the rest -- and 2416 ms for HistGradBoost's first
        call against 2.5 ms for its second).

        Left unpaid, that lands on whichever live window arrives first and
        freezes the GUI thread for two and a half seconds. The dialog
        therefore runs this on a worker thread before showing anything live.

        A row of zeros is used deliberately: the answer is thrown away, so
        it only has to be the right SHAPE. Nothing it returns is ever shown.
        """
        if not self.ready or self.warm:
            return 0.0
        self.warming = True
        started = time.perf_counter()
        row = np.zeros((1, int(self.setup.get("n_features", 0) or 0)),
                       dtype=np.float32)
        try:
            if row.shape[1]:
                for model in self.members.values():
                    try:
                        model.predict_proba(row)
                    except Exception:                      # pragma: no cover
                        pass
        finally:
            self.warm_ms = (time.perf_counter() - started) * 1000.0
            self.warm = True
            self.warming = False
        return self.warm_ms

    @property
    def has_space(self) -> bool:
        """
        Is there a usable 2-D projection of the feature space?

        Deliberately `loaded` and not `ready`: `ready` also requires the
        fitted members, because the DECISION TRAIL has to run them. Drawing
        the scatter runs no model at all -- it needs the stored transform and
        nothing else -- so a panel that carries a projection can show it even
        if its members are unusable.
        """
        space = self.space or {}
        return bool(self.loaded and space.get("kind") == "pca"
                    and space.get("components") is not None
                    and space.get("coords") is not None)

    def project(self, features) -> Optional[Tuple[float, float]]:
        """
        One live feature vector -> its (x, y) in the stored PCA space.

        The transform is the one fitted at training time, applied here and
        nowhere else, so a live point and a training point are placed by
        exactly the same arithmetic. Refitting a projection on live data
        would move the axes under the scatter and make the two
        incomparable -- the same reason the trail reuses
        `LivePredictor.last_features` instead of recomputing features.

        None when there is no projection, or when the vector is the wrong
        width for it. A silently reshaped vector would land somewhere
        plausible and mean nothing.
        """
        if not self.has_space or features is None:
            return None
        space = self.space
        vec = np.asarray(features, dtype=np.float64).ravel()
        mean = np.asarray(space["mean"], dtype=np.float64)
        if vec.shape[0] != mean.shape[0]:
            return None
        scale = np.asarray(space["scale"], dtype=np.float64)
        comps = np.asarray(space["components"], dtype=np.float64)
        out = ((vec - mean) / scale) @ comps.T
        if not np.all(np.isfinite(out)):
            return None
        return float(out[0]), float(out[1])

    def matches(self, predictor: "LivePredictor") -> bool:
        """
        Is this panel about the model that is loaded?

        A panel built for a different window length or feature set would be
        fed vectors it was never fitted on. Rather than predict nonsense, the
        view refuses and says the panel is stale.
        """
        if not self.ready or not predictor.ready:
            return False
        setup = self.setup
        return (list(self.labels) == list(predictor.labels)
                and setup.get("feature_set") == predictor.feature_set
                and setup.get("representation") == predictor.representation
                and bool(setup.get("freq_features")) == bool(predictor.freq_features)
                and int(setup.get("window_samples", -1)) == int(predictor.window_samples))

    def stale_reason(self, predictor: "LivePredictor") -> str:
        """Why `matches()` said no, in words the dialog can show."""
        if not self.ready:
            return self.status
        if not predictor.ready:
            return "no model is loaded"
        setup = self.setup
        for field, mine, theirs in (
                ("labels", list(self.labels), list(predictor.labels)),
                ("feature set", setup.get("feature_set"), predictor.feature_set),
                ("representation", setup.get("representation"),
                 predictor.representation),
                ("frequency features", bool(setup.get("freq_features")),
                 bool(predictor.freq_features)),
                ("window samples", int(setup.get("window_samples", -1)),
                 int(predictor.window_samples))):
            if mine != theirs:
                return (f"the panel was built for {field} {mine!r}, the loaded "
                        f"model uses {theirs!r} -- retrain to rebuild it")
        return ""

    # -- live -----------------------------------------------------------------

    def step_for_features(self, features: np.ndarray,
                          predictor: "LivePredictor" = None) -> Optional[TrailStep]:
        """
        Run every candidate on one feature vector that was already computed.

        `features` is the (1, n) row LivePredictor built for this window, so
        both see bit-identical input. Returns None when the panel cannot
        answer; it never substitutes an estimate.
        """
        if not self.ready:
            return None
        row = np.asarray(features, dtype=np.float32)
        if row.ndim == 1:
            row = row[None, :]
        expected = int(self.setup.get("n_features", row.shape[1]))
        if row.shape[1] != expected:
            self.status = (f"this panel wants {expected} features, got "
                           f"{row.shape[1]}")
            return None

        per_model, names = [], []
        for name in self.names:
            model = self.members.get(name)
            if model is None:
                continue
            try:
                probabilities = np.asarray(
                    model.predict_proba(row), dtype=float)[0]
            except Exception as exc:
                self.status = f"{name} failed on a live window: {exc}"
                return None
            names.append(name)
            per_model.append([float(p) for p in probabilities])
        if not per_model:
            return None

        mean = np.mean(np.asarray(per_model, dtype=float), axis=0)
        #  The stored ensemble is NOT called here. Its soft vote is the mean
        #  of these same members, which build_decision_panel() checks over
        #  the whole dataset and records in `fusion` -- so calling it would
        #  re-run all seven members a second time for a number already in
        #  hand. That doubling is what made the live view miss windows.
        #  `fusion.verified` is the evidence; when it is False the view shows
        #  the ensemble's recorded output instead of trusting the mean.
        fused = None

        deployed_name = self.winner or (predictor.stats.get("model_name", "")
                                        if predictor else "")
        deployed = None
        if deployed_name in names:
            deployed = per_model[names.index(deployed_name)]

        return TrailStep(
            names=names, per_model=per_model, labels=list(self.labels),
            fused=fused, fused_from_mean=[float(v) for v in mean],
            deployed_name=deployed_name or "?", deployed=deployed,
            source="live",
            note=("these candidates were fitted on the whole dataset, so "
                  "their agreement here demonstrates the vote -- it is not "
                  "an accuracy estimate"),
        )

    # -- recorded -------------------------------------------------------------

    @property
    def has_trail(self) -> bool:
        return bool(self._trail.get("proba") is not None
                    and self._trail.get("truth") is not None)

    def trail_length(self) -> int:
        if not self.has_trail:
            return 0
        return int(np.asarray(self._trail["truth"]).shape[0])

    def windows_of(self, recording_index: int) -> List[int]:
        """Row numbers of one recording's windows, in order."""
        if not self.has_trail or self._trail.get("rec") is None:
            return []
        rec = np.asarray(self._trail["rec"])
        return [int(i) for i in np.flatnonzero(rec == int(recording_index))]

    def recorded_step(self, row: int) -> Optional[TrailStep]:
        """One window of the out-of-fold trail captured during training."""
        if not self.has_trail:
            return None
        total = self.trail_length()
        if not 0 <= row < total:
            return None
        proba = self._trail["proba"]
        names, per_model = [], []
        for name in self.names:
            block = proba.get(name)
            if block is None:
                continue
            names.append(name)
            per_model.append([float(p) for p in np.asarray(block)[row]])
        if not per_model:
            return None

        fused = None
        if self.vote_name and proba.get(self.vote_name) is not None:
            fused = [float(p) for p in np.asarray(proba[self.vote_name])[row]]
        mean = np.mean(np.asarray(per_model, dtype=float), axis=0)
        truth_index = int(np.asarray(self._trail["truth"])[row])
        fold = int(np.asarray(self._trail["fold"])[row]) if \
            self._trail.get("fold") is not None else -1
        deployed = (per_model[names.index(self.winner)]
                    if self.winner in names else None)

        return TrailStep(
            names=names, per_model=per_model, labels=list(self.labels),
            fused=fused, fused_from_mean=[float(v) for v in mean],
            deployed_name=self.winner or "?", deployed=deployed,
            source="recorded",
            note=(f"recorded label {self.labels[truth_index]}"
                  if 0 <= truth_index < len(self.labels) else "")
                 + (f", held out in fold {fold + 1}" if fold >= 0 else ""),
        )

    def recording_index_of(self, row: int) -> int:
        """Which recording a trail row belongs to, or -1."""
        rec = self._trail.get("rec")
        if rec is None:
            return -1
        rec = np.asarray(rec)
        return int(rec[row]) if 0 <= row < rec.shape[0] else -1

    def recorded_truth(self, row: int) -> int:
        if not self.has_trail:
            return -1
        truth = np.asarray(self._trail["truth"])
        return int(truth[row]) if 0 <= row < truth.shape[0] else -1

    def recorded_channels(self, row: int) -> Optional[np.ndarray]:
        """The per-channel level of a recorded window, when it was stored."""
        chan = self._trail.get("chan")
        if chan is None:
            return None
        chan = np.asarray(chan)
        if not 0 <= row < chan.shape[0]:
            return None
        return chan[row]

    def split_windows(self) -> List[int]:
        """
        Rows where the candidates did NOT all agree.

        The only interesting windows on a dataset this separable, and what
        the view's "next disagreement" button steps through.
        """
        if not self.has_trail:
            return []
        proba = self._trail["proba"]
        blocks = [np.asarray(proba[n]) for n in self.names if n in proba]
        if len(blocks) < 2:
            return []
        picks = np.stack([np.argmax(b, axis=1) for b in blocks], axis=1)
        varies = (picks != picks[:, :1]).any(axis=1)
        return [int(i) for i in np.flatnonzero(varies)]


@dataclass
class Prediction:
    """One classifier output. Every field comes from the model, nothing is invented."""
    label: str                      # smoothed label (what to display)
    confidence: float               # probability of `label`, or NaN
    raw_label: str                  # this window's own argmax, unsmoothed
    raw_confidence: float
    probabilities: List[float]      # smoothed, per class, aligned with `labels`
    raw_probabilities: List[float]  # this window alone, unsmoothed
    labels: List[str]
    n_samples: int                  # samples the window was built from
    smoothed_over: int              # how many windows the vote covered
    unix_time: float

    @property
    def certain(self) -> bool:
        if math.isnan(self.confidence) or PREDICTION_MIN_CONFIDENCE <= 0:
            return True
        return self.confidence >= PREDICTION_MIN_CONFIDENCE

    @property
    def percent(self) -> str:
        if math.isnan(self.confidence):
            return "--"
        return f"{self.confidence * 100:.0f} %"


class LivePredictor:
    """
    Real-time gesture prediction.

    Inert until a trained model exists: with no model file nothing is shown
    and the buttons stay disabled. It never invents a label or a percentage.

    The feature path goes through model_feature_vector(), the same function
    train_model.py uses, so the live input cannot drift away from what the
    model was fitted on. The live window is resampled onto the model's own
    grid first (resample_uniform), which is what makes the prediction
    independent of how the BLE link happens to be behaving.

    Expected joblib payload (written by train_model.py):

        {"model": <estimator>, "labels": [...], "fs": 60.0,
         "window_s": 0.2, "window_samples": 12, "representation": "raw",
         "feature_set": "level", "freq_features": True, "stats": {...}}
    """

    def __init__(self, path: str = None):
        self.path = path or MODEL_PATH
        self.model = None
        self.labels: List[str] = []
        self.window_s = WINDOW_MS / 1000.0
        self.window_samples = 12
        self.fs = nominal_rate()
        self.representation = "raw"
        self.feature_set = "level"
        self.freq_features = True
        self.stats: dict = {}
        self.payload_meta: dict = {}
        self.status = "disabled"
        self.history = deque(maxlen=max(1, PREDICTION_SMOOTHING))
        self.prob_history = deque(maxlen=max(1, PREDICTION_SMOOTHING))
        self.last: Optional[Prediction] = None
        # The exact row the last prediction was made from, kept so the
        # decision panel can run the other candidates on the SAME input
        # instead of building a second feature path that could drift.
        self.last_features: Optional[np.ndarray] = None
        self.last_window: Optional[np.ndarray] = None

    # -- loading --------------------------------------------------------------

    @property
    def full_path(self) -> str:
        return (self.path if os.path.isabs(self.path)
                else os.path.join(PROJECT_DIR, self.path))

    def load(self) -> bool:
        self.model = None
        if not ENABLE_LIVE_PREDICTION:
            self.status = "disabled in the configuration"
            return False
        if not os.path.exists(self.full_path):
            self.status = "no model file"
            return False
        try:
            import joblib
        except ImportError:
            self.status = "joblib is not installed"
            return False
        try:
            payload = joblib.load(self.full_path)
        except Exception as exc:
            self.status = f"could not load the model: {exc}"
            return False

        if not isinstance(payload, dict) or "model" not in payload:
            self.status = "unsupported model file (expected a dict payload)"
            return False

        self.model = payload["model"]
        self.labels = list(payload.get("labels", GESTURES))
        self.fs = float(payload.get("fs", nominal_rate()))
        self.window_s = float(payload.get(
            "window_s", payload.get("window_ms", WINDOW_MS) / 1000.0))
        self.window_samples = int(payload.get(
            "window_samples", max(4, round(self.window_s * self.fs))))
        self.representation = payload.get("representation", "raw")
        self.feature_set = payload.get("feature_set", "level")
        self.freq_features = bool(payload.get("freq_features", True))
        self.stats = dict(payload.get("stats", {}))
        self.payload_meta = {k: v for k, v in payload.items()
                             if k not in ("model", "stats")}
        self.loaded_mtime = os.path.getmtime(self.full_path) \
            if os.path.exists(self.full_path) else 0.0

        n_expected = len(model_feature_names(self.feature_set,
                                             self.freq_features))
        self.status = "ready"
        self.reset()

        # Fail loudly here rather than producing nonsense at 7 Hz.
        probe = self.model
        try:
            dummy = np.zeros((1, n_expected), dtype=np.float32)
            probe.predict(dummy)
        except Exception as exc:
            self.model = None
            self.status = (f"model expects different features than "
                           f"{self.feature_set!r} produces ({n_expected}): {exc}")
            return False
        return True

    @property
    def ready(self) -> bool:
        return self.model is not None

    def reset(self) -> None:
        self.history.clear()
        self.prob_history.clear()
        self.last = None
        self.last_features = None
        self.last_window = None

    def describe(self) -> str:
        if not self.ready:
            return self.status
        name = self.stats.get("model_name", "model")
        acc = self.stats.get("recording_vote_accuracy")
        suffix = f", {acc * 100:.0f} % per-gesture CV" if acc else ""
        return (f"{name}  |  {self.window_s * 1000:.0f} ms @ {self.fs:.0f} Hz  |  "
                f"{self.feature_set} features{suffix}")

    def data_status(self, fingerprint: dict = None) -> dict:
        """
        Is a model trained, on what, and is that still what is on disk?

        Delegates to `model_data_status()` so the trainers, the GUI and the
        tests all answer the question the same way.
        """
        if not self.ready:
            return model_data_status(None)
        payload = dict(self.payload_meta)
        payload["stats"] = self.stats
        return model_data_status(payload, fingerprint)

    # -- inference ------------------------------------------------------------

    def window_seconds(self) -> float:
        return float(self.window_s)

    def predict(self, emg: np.ndarray, t: np.ndarray) -> Optional[Prediction]:
        """
        Classify one live window.

        `emg`/`t` are raw ring-buffer samples covering roughly window_s
        seconds; they are resampled onto the model's grid here. Returns None
        when there is not enough data to say anything.
        """
        if not self.ready:
            return None
        emg = np.asarray(emg)
        t = np.asarray(t, dtype=np.float64)
        if emg.shape[0] < 4 or t.size != emg.shape[0]:
            return None

        block = resample_uniform(emg, t, self.window_samples)
        features = model_feature_vector(
            block, self.representation, self.feature_set,
            self.freq_features, self.fs,
        )[None, :]
        self.last_features = features
        self.last_window = block

        try:
            if hasattr(self.model, "predict_proba"):
                probabilities = np.asarray(
                    self.model.predict_proba(features), dtype=float)[0]
                index = int(np.argmax(probabilities))
                confidence = float(probabilities[index])
            else:
                index = int(np.asarray(self.model.predict(features)).ravel()[0])
                probabilities = np.full(len(self.labels), float("nan"))
                confidence = float("nan")
        except Exception as exc:
            self.status = f"prediction failed: {exc}"
            self.model = None
            return None

        raw_label = self._name(index)
        self.history.append(index)

        # Smoothing is SOFT voting: average the probability vectors over the
        # recent windows and take the argmax of the average.
        #
        # Hard majority voting on labels was tried first and is wrong to show:
        # it can report a label whose current probability is 0.16 while the
        # latest window puts 0.84 on something else, so the number next to the
        # label contradicts the label. Averaging the distributions keeps the
        # displayed confidence consistent with the displayed answer, and is
        # also the better estimator.
        if math.isnan(confidence):
            smoothed_probs = probabilities
            smoothed = index
            smooth_conf = confidence
        else:
            self.prob_history.append(probabilities)
            smoothed_probs = np.mean(np.vstack(self.prob_history), axis=0)
            smoothed = int(np.argmax(smoothed_probs))
            smooth_conf = float(smoothed_probs[smoothed])

        prediction = Prediction(
            label=self._name(smoothed),
            confidence=smooth_conf,
            raw_label=raw_label,
            raw_confidence=confidence,
            probabilities=[float(p) for p in smoothed_probs],
            raw_probabilities=[float(p) for p in probabilities],
            labels=list(self.labels),
            n_samples=int(emg.shape[0]),
            smoothed_over=len(self.prob_history) or 1,
            unix_time=time.time(),
        )
        self.last = prediction
        return prediction

    def _name(self, index: int) -> str:
        return (self.labels[index] if 0 <= index < len(self.labels)
                else str(index))


# ------------------------------------------------------------------------------
# Main window
# ------------------------------------------------------------------------------


class MainWindow(QtWidgets.QMainWindow):
    """Everything wired together."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME}  v{APP_VERSION}")
        # Fit the display rather than assuming a big monitor. The minimum is
        # deliberately small: a minimum bigger than the screen is a window the
        # operator cannot resize into view, which is worse than a cramped one.
        self._fitted_size = fit_to_screen(
            self, WINDOW_WIDTH, WINDOW_HEIGHT, min_width=900, min_height=600)

        self.person_id = PERSON_ID
        self.session_id = SESSION_ID
        self.recordings = []              # List[Recording] collected this run
        self.session_markers = []         # List[Marker] logged this session

        # FREE TAKE state. Deliberately NOT routed through SequenceController:
        # a take has no plan, no countdown and no review step, and the
        # acquisition thread has only one capture slot, so the two must never
        # run at the same time (enforced in _refresh_controls).
        self.free_capture = None          # the Capture while a take runs
        self.free_take_label = ""         # intended gesture for that take
        self.free_take_number = 0         # position in the session
        self.export_worker = None
        self.review_recording = None      # the take awaiting a verdict
        self.prediction_active = False    # continuous mode running
        self.one_shot_capture_at = 0.0    # when the classified window starts
        self.one_shot_deadline = 0.0      # when it ends
        self.one_shot_clear_at = 0.0      # when to clear a held one-shot result
        self.training_process = None      # QProcess running train_model.py
        # Which classifiers the next retrain sweeps. Loaded from disk so an
        # edit survives a restart; the defaults reproduce the roster
        # train_model.py has always used.
        self.model_spec, self._spec_problems = load_model_spec(
            model_spec_path(MODEL_PATH))
        self.training_output = []         # its captured stdout
        self.training_started_at = 0.0    # for the elapsed/ETA readout
        self.training_phase_text = ""
        # GPU or CPU. Unknown until the subprocess probe answers, and shown
        # as unknown until then rather than assumed either way.
        self.compute_info = blank_compute_info()
        self.compute_process = None       # QProcess running the probe
        self._link_paused = False         # session paused by a dropped link
        self._last_banner = (None, None, None)

        self.writer = DatasetWriter(
            os.path.join(PROJECT_DIR, DATA_FOLDER), self.person_id, self.session_id
        )

        self.bridge = AcquisitionBridge()
        self.bridge.state_changed.connect(self._on_connection_state)
        self.bridge.log_line.connect(self._append_log)

        self.acq = MyoAcquisition(
            port=MYO_PORT,
            mode=EMG_MODE,
            on_state=lambda state, msg: self.bridge.state_changed.emit(state, msg),
            on_log=lambda text: self.bridge.log_line.emit(text),
        )

        self.sequence = SequenceController(
            on_capture_start=self._on_capture_start,
            on_capture_stop=self._on_capture_stop,
            on_state=self._on_sequence_state,
            on_finished=self._on_sequence_finished,
            on_capture_discard=self._on_capture_discard,
        )

        self.predictor = LivePredictor()
        self.predictor.load()
        # The side file behind MODEL STATS -> DECISION TRAIL. Optional by
        # design: a failure to load it must not touch the classifier, so it
        # is a separate object with its own status string.
        #  Loaded lazily, when MODEL STATS is first opened: it unpickles every
        #  candidate (two 400-tree forests among them) and nothing outside
        #  that dialog reads it, so there is no reason to spend the time at
        #  startup.
        self.decision_panel = DecisionPanel(model_path=MODEL_PATH)
        self.stats_dialog = None

        self._build_ui()
        self._load_session_markers()
        self._refresh_take_history()
        self._refresh_spec_summary()
        self._refresh_gesture_counts()
        self._refresh_label_counts()
        self._start_timers()

        self.acq.start()
        self._append_log(f"{APP_NAME} {APP_VERSION} started. Mode: {EMG_MODE}.")
        self._refresh_plan_info()

    # ======================================================================
    # UI construction
    # ======================================================================

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(8)

        outer.addWidget(self._build_header())

        self.banner = StatusBanner()
        outer.addWidget(self.banner)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._build_plot_area())
        splitter.addWidget(self._build_side_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1180, 440])
        outer.addWidget(splitter, 1)

        outer.addWidget(self._build_button_bar())

        self.status = self.statusBar()
        self.status_conn = QtWidgets.QLabel("--")
        self.status_rate = QtWidgets.QLabel("--")
        self.status_buffer = QtWidgets.QLabel("--")
        self.status_counts = QtWidgets.QLabel("--")
        # GPU or CPU, always on screen. The full badge lives in the LIVE
        # CLASSIFIER panel, which folds away; this one cannot be folded, so
        # the answer is never more than a glance away.
        self.status_compute = QtWidgets.QLabel("GPU/CPU ?")
        for widget in (self.status_conn, self.status_rate,
                       self.status_buffer, self.status_counts,
                       self.status_compute):
            widget.setStyleSheet(f"color: {COL_TEXT_DIM}; padding: 0 10px;")
            self.status.addPermanentWidget(widget)
        self.status.showMessage("Ready.")

    def _build_header(self):
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {COL_PANEL}; border: 1px solid {COL_BORDER};"
            f" border-radius: 8px; }}"
        )
        row = QtWidgets.QHBoxLayout(frame)
        row.setContentsMargins(14, 8, 14, 8)

        row.addWidget(make_label("MYO EMG DATA COLLECTION", 15, True, COL_TEXT))
        row.addSpacing(18)

        self.conn_pill = QtWidgets.QLabel(CONN_DISCONNECTED)
        self.conn_pill.setAlignment(QtCore.Qt.AlignCenter)
        self.conn_pill.setMinimumWidth(150)
        self._set_pill(CONN_DISCONNECTED, COL_TEXT_DIM)
        row.addWidget(self.conn_pill)

        row.addStretch(1)

        self.header_person = make_label("--", 12, True, COL_ACCENT)
        self.header_gesture = make_label("--", 12, True, "#ffffff")
        self.header_count = make_label("0", 12, True, COL_GOOD)
        for caption, widget in (("PERSON", self.header_person),
                                ("GESTURE", self.header_gesture),
                                ("RECORDED", self.header_count)):
            box = QtWidgets.QVBoxLayout()
            box.setSpacing(0)
            box.addWidget(make_label(caption, 8, False, COL_TEXT_DIM))
            box.addWidget(widget)
            container = QtWidgets.QWidget()
            container.setLayout(box)
            container.setStyleSheet("background: transparent;")
            row.addWidget(container)
            row.addSpacing(16)

        return frame

    def _build_plot_area(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ---- graph toolbar ---------------------------------------------------
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(8)

        bar.addWidget(make_label("Window", 10, False, COL_TEXT_DIM))
        self.window_spin = QtWidgets.QDoubleSpinBox()
        # Up to the whole ring buffer: the point of a long buffer is being
        # able to actually look at it.
        self.window_spin.setRange(0.25, float(RING_BUFFER_SECONDS))
        self.window_spin.setSingleStep(0.5)
        self.window_spin.setDecimals(2)
        self.window_spin.setSuffix(" s")
        self.window_spin.setValue(PLOT_WINDOW_S)
        self.window_spin.setToolTip("Visible time window")
        bar.addWidget(self.window_spin)

        self.autoscale_check = QtWidgets.QCheckBox("Autoscale")
        self.autoscale_check.setChecked(PLOT_AUTOSCALE)
        self.autoscale_check.setToolTip(
            "Grows instantly on a peak and shrinks slowly, so a spike is never "
            "clipped out of view"
        )
        bar.addWidget(self.autoscale_check)

        # The sensitivity slider is only meaningful with Autoscale OFF -- its
        # own tooltip said so -- yet it permanently occupied ~190 px of the
        # toolbar, which is what squeezed everything else into "Win" / "Sen:"
        # / "Cha" on a narrow window. It now appears only when it does
        # something.
        self.sensitivity_label = make_label("Sensitivity", FS_SM, False,
                                            COL_TEXT_DIM)
        bar.addWidget(self.sensitivity_label)
        self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale_slider.setRange(10, 1000)
        lo, hi = emg_range()
        self.scale_slider.setValue(int(np.clip((hi - lo) * 0.5, 10, 1000)))
        self.scale_slider.setFixedWidth(130)
        self.scale_slider.setToolTip(
            "Manual vertical scale. Only used when Autoscale is off.")
        bar.addWidget(self.scale_slider)
        self.sensitivity_label.setVisible(not PLOT_AUTOSCALE)
        self.scale_slider.setVisible(not PLOT_AUTOSCALE)

        self.envelope_check = QtWidgets.QCheckBox("RMS envelope")
        self.envelope_check.setChecked(SHOW_ENVELOPE)
        bar.addWidget(self.envelope_check)

        self.freeze_button = QtWidgets.QPushButton("Freeze graph")
        self.freeze_button.setCheckable(True)
        self.freeze_button.setToolTip(
            "Stop updating the plot. Recording and saving are unaffected."
        )
        bar.addWidget(self.freeze_button)

        reset = QtWidgets.QPushButton("Reset view")
        reset.setToolTip("Undo mouse zoom/pan")
        bar.addWidget(reset)

        bar.addStretch(1)
        layout.addLayout(bar)

        # Channels on their own row. Eight checkboxes plus a label is 220 px
        # that the first row cannot spare on a laptop screen.
        chan_bar = QtWidgets.QHBoxLayout()
        chan_bar.setSpacing(SP_SM)
        chan_bar.addWidget(make_label("Channels", FS_SM, False, COL_TEXT_DIM))
        self.channel_checks = []
        for i in range(N_CHANNELS):
            check = QtWidgets.QCheckBox(str(i + 1))
            check.setChecked(True)
            check.setStyleSheet(
                f"QCheckBox {{ color: {CHANNEL_COLORS[i]}; font-weight: 700; }}"
                f"QCheckBox::indicator:checked {{ background: {CHANNEL_COLORS[i]};"
                f" border-color: {CHANNEL_COLORS[i]}; }}"
            )
            check.toggled.connect(
                lambda checked, index=i: self.plot.set_channel_visible(index, checked)
            )
            self.channel_checks.append(check)
            chan_bar.addWidget(check)

        chan_bar.addStretch(1)
        layout.addLayout(chan_bar)

        self.plot = EmgPlotWidget()
        layout.addWidget(self.plot, 1)

        # Review read-out: what the green band currently keeps.
        self.review_info = make_label("", 11, True, "#22c55e")
        self.review_info.setWordWrap(True)
        layout.addWidget(self.review_info)
        self.plot.trim_changed.connect(self._update_review_info)
        self.plot.selection_changed.connect(self._on_selection_changed)

        self.window_spin.valueChanged.connect(self.plot.set_window_seconds)
        self.autoscale_check.toggled.connect(self.plot.set_autoscale)
        self.autoscale_check.toggled.connect(
            lambda on: (self.sensitivity_label.setVisible(not on),
                        self.scale_slider.setVisible(not on)))
        self.scale_slider.valueChanged.connect(
            lambda value: self.plot.set_manual_scale(float(value))
        )
        self.envelope_check.toggled.connect(self.plot.set_envelope_visible)
        self.freeze_button.toggled.connect(self._on_freeze_toggled)
        reset.clicked.connect(self.plot.reset_view)
        self.plot.set_manual_scale(float(self.scale_slider.value()))

        return panel

    def _build_side_panel(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        inner = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(inner)
        column.setContentsMargins(4, 0, 4, 0)
        column.setSpacing(8)

        # Every group folds. The ones open by default are the ones you need
        # to start recording; the rest are one click away and out of the way
        # until then.
        self.sections = {}
        for key, builder, start_open in (
            ("subject", self._group_subject, True),
            ("free_take", self._group_free_take if FREE_TAKE_ENABLED else None,
             True),
            ("label", self._group_label_on_graph, True),
            ("markers", self._group_markers if MARKERS_ENABLED else None, False),
            ("plan", self._group_plan, False),
            ("channels", self._group_channels, False),
            ("recordings", self._group_recordings, True),
            ("prediction", self._group_prediction, False),
            ("training", self._group_training, False),
            ("log", self._group_log, False),
        ):
            if builder is None:
                continue
            section = collapsible(builder(), start_open)
            section.toggled.connect(
                lambda opened, sec=section: self._on_section_toggled(sec, opened))
            self.sections[key] = section
            column.addWidget(section)
        column.addStretch(1)

        scroll.setWidget(inner)
        self._scroll_area = scroll
        # Wide enough for the controls to read properly, but clamped on a
        # small screen so the panel cannot push the graph off the window.
        avail_w, _ = available_screen_size(self)
        scroll.setMinimumWidth(min(410, max(280, int(avail_w * 0.32))))
        scroll.setMaximumWidth(max(410, int(avail_w * 0.45)))
        return scroll

    def _group_subject(self):
        group = QtWidgets.QGroupBox("SUBJECT / SESSION")
        form = QtWidgets.QFormLayout(group)
        form.setLabelAlignment(QtCore.Qt.AlignRight)
        form.setSpacing(6)

        self.person_edit = QtWidgets.QLineEdit(self.person_id)
        self.person_edit.setToolTip(
            "Use a NEW id for every new subject. Data is kept in separate "
            "folders per person so that leave-one-person-out validation works."
        )
        self.session_edit = QtWidgets.QLineEdit(self.session_id)
        self.session_edit.setToolTip(
            "Start a new session every time the armband is taken off and put "
            "back on -- electrode positions shift."
        )
        self.hand_combo = QtWidgets.QComboBox()
        self.hand_combo.addItems(["right", "left"])
        self.hand_combo.setCurrentText(HANDEDNESS)
        self.arm_combo = QtWidgets.QComboBox()
        self.arm_combo.addItems(["right", "left"])
        self.arm_combo.setCurrentText(ARM_WORN_ON)
        self.device_edit = QtWidgets.QLineEdit(DEVICE_NAME)
        self.notes_edit = QtWidgets.QLineEdit(SUBJECT_NOTES)
        self.notes_edit.setPlaceholderText("notes stored with every recording")

        form.addRow("Person id", self.person_edit)
        form.addRow("Session id", self.session_edit)
        form.addRow("Handedness", self.hand_combo)
        form.addRow("Armband on", self.arm_combo)
        form.addRow("Device", self.device_edit)
        form.addRow("Notes", self.notes_edit)

        self.folder_label = make_label("", 10, False, COL_TEXT_DIM)
        self.folder_label.setWordWrap(True)
        form.addRow("Folder", self.folder_label)

        self.person_edit.editingFinished.connect(self._on_identity_changed)
        self.session_edit.editingFinished.connect(self._on_identity_changed)
        for widget in (self.hand_combo, self.arm_combo):
            widget.currentTextChanged.connect(self._on_metadata_changed)
        self.device_edit.editingFinished.connect(self._on_metadata_changed)
        self.notes_edit.editingFinished.connect(self._on_metadata_changed)

        return group

    def _group_label_on_graph(self):
        """
        Free recording: perform a gesture whenever you like, then box it on
        the graph and label it. Nothing enters the dataset until you select
        it, so hesitation and false starts never get recorded.
        """
        group = QtWidgets.QGroupBox("LABEL ON GRAPH  (free recording)")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        self.label_mode_button = QtWidgets.QPushButton("START LABEL MODE")
        self.label_mode_button.setCheckable(True)
        self.label_mode_button.setMinimumHeight(32)
        self.label_mode_button.setToolTip(
            "Stream continuously and label by hand. No countdown: do the "
            "gesture whenever you like, then drag a box around it."
        )
        self.label_mode_button.toggled.connect(self.toggle_label_mode)
        layout.addWidget(self.label_mode_button)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(make_label("Gesture", 10, False, COL_TEXT_DIM))
        self.label_gesture_combo = QtWidgets.QComboBox()
        self.label_gesture_combo.addItems(GESTURES)
        self.label_gesture_combo.setToolTip(
            "What the selection will be labelled as. Set this BEFORE you do "
            "the gesture, so you only have to press ADD afterwards."
        )
        row.addWidget(self.label_gesture_combo, 1)
        layout.addLayout(row)

        # The default cut length. One value, used by SELECT LAST n s here and
        # by the cut dialog that carves clips out of a take, mirrored into
        # LABEL_DEFAULT_SELECTION_S so it lands in every *_meta.json too.
        row_len = QtWidgets.QHBoxLayout()
        row_len.addWidget(make_label("Default cut (s)", 10, False, COL_TEXT_DIM))
        self.cut_length_spin = QtWidgets.QDoubleSpinBox()
        self.cut_length_spin.setDecimals(2)
        self.cut_length_spin.setSingleStep(0.25)
        self.cut_length_spin.setRange(LABEL_MIN_SELECTION_S,
                                      max(LABEL_MIN_SELECTION_S,
                                          RING_BUFFER_SECONDS))
        self.cut_length_spin.setValue(float(LABEL_DEFAULT_SELECTION_S))
        self.cut_length_spin.setToolTip(
            "How long a fresh selection is, in seconds. Applies to every cut "
            "from now on -- here and in the cut-clips dialog -- instead of "
            "the fixed 2 s."
        )
        self.cut_length_spin.valueChanged.connect(self._on_cut_length_changed)
        row_len.addWidget(self.cut_length_spin, 1)
        layout.addLayout(row_len)

        self.select_button = QtWidgets.QPushButton(
            f"SELECT LAST {LABEL_DEFAULT_SELECTION_S:.2f} s")
        self.select_button.setMinimumHeight(30)
        self.select_button.setToolTip(
            "Drop a selection box over the last few seconds, then drag "
            "its edges to fit the gesture exactly."
        )
        self.select_button.clicked.connect(self.arm_selection)
        layout.addWidget(self.select_button)

        self.selection_info = make_label("", 10, False, COL_TEXT_DIM)
        self.selection_info.setWordWrap(True)
        layout.addWidget(self.selection_info)

        self.add_label_button = QtWidgets.QPushButton("ADD SELECTION")
        self.add_label_button.setObjectName("primary")
        self.add_label_button.setMinimumHeight(34)
        self.add_label_button.setToolTip(
            "Save the boxed signal as a labelled recording.  Ctrl+Space"
        )
        self.add_label_button.clicked.connect(self.add_selection_as_label)
        layout.addWidget(self.add_label_button)

        row2 = QtWidgets.QHBoxLayout()
        self.undo_label_button = QtWidgets.QPushButton("UNDO LAST")
        self.undo_label_button.setObjectName("danger")
        self.undo_label_button.setToolTip(
            "Remove the last labelled mark. Its files move to _deleted, "
            "never erased."
        )
        self.undo_label_button.clicked.connect(self.undo_last_label)
        row2.addWidget(self.undo_label_button)

        self.label_counts = make_label("", 10, False, COL_GOOD)
        self.label_counts.setWordWrap(True)
        row2.addWidget(self.label_counts, 1)
        layout.addLayout(row2)

        self._accept_label_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Space"), self)
        self._accept_label_shortcut.activated.connect(self.add_selection_as_label)

        return group

    def _group_free_take(self):
        """
        Record the whole session in one piece now, cut it up later.

        The counterpart to LABEL ON GRAPH: there you must label within
        RING_BUFFER_SECONDS because nothing is on disk until you select it,
        here the whole stretch goes to disk unlabelled and the cutting can
        happen tomorrow, on a machine with no armband.
        """
        group = QtWidgets.QGroupBox("FREE TAKE  (record now, cut later)")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(make_label("Intended gesture", 10, False, COL_TEXT_DIM))
        self.take_gesture_combo = QtWidgets.QComboBox()
        self.take_gesture_combo.addItems(GESTURES)
        self.take_gesture_combo.setToolTip(
            "What you sat down to collect. It names the take and is its "
            "default label when cutting, but it does NOT label the data: "
            "every clip is labelled when you cut it out."
        )
        row.addWidget(self.take_gesture_combo, 1)
        layout.addLayout(row)

        self.free_take_button = QtWidgets.QPushButton("START FREE TAKE")
        self.free_take_button.setObjectName("primary")
        self.free_take_button.setCheckable(True)
        self.free_take_button.setMinimumHeight(38)
        self.free_take_button.setToolTip(
            f"Record continuously for up to {FREE_TAKE_MAX_MINUTES:.0f} "
            "minutes, then write the whole stretch to disk in one piece.  "
            "Ctrl+Shift+R"
        )
        self.free_take_button.toggled.connect(self.toggle_free_take)
        layout.addWidget(self.free_take_button)

        self.free_take_status = make_label(
            "idle -- nothing being recorded", 11, True, COL_TEXT_DIM)
        self.free_take_status.setWordWrap(True)
        layout.addWidget(self.free_take_status)

        self.free_take_detail = make_label("", 10, False, COL_TEXT_DIM)
        self.free_take_detail.setWordWrap(True)
        layout.addWidget(self.free_take_detail)

        layout.addWidget(make_label("TAKES ON DISK", 10, True, COL_TEXT_DIM))
        self.take_list = QtWidgets.QListWidget()
        self.take_list.setMaximumHeight(96)
        # Several takes can be queued up: Ctrl+click or Shift+click. They are
        # opened one after another, and each stays its own recording -- see
        # open_takes_for_cutting().
        self.take_list.setSelectionMode(
            QtWidgets.QAbstractItemView.ExtendedSelection)
        self.take_list.setToolTip(
            "Every take saved for this subject and session. Double-click one "
            "to open it, or Ctrl/Shift+click several and open them in a row. "
            "Each take -- and every clip cut from it -- stays a separate "
            "recording."
        )
        self.take_list.itemDoubleClicked.connect(
            lambda _item: self.open_selected_take())
        layout.addWidget(self.take_list)

        row2 = QtWidgets.QHBoxLayout()
        self.open_take_button = QtWidgets.QPushButton("OPEN TAKE(S) & CUT CLIPS")
        self.open_take_button.setMinimumHeight(30)
        self.open_take_button.setToolTip(
            "Open the selected take(s): select a span, pick a gesture, "
            "EXTRACT CLIP, repeat. Ctrl/Shift+click several to work through "
            "them in a row. Every clip is its own recording -- clips are "
            "never joined together. The takes themselves are never modified."
        )
        self.open_take_button.clicked.connect(self.open_selected_take)
        row2.addWidget(self.open_take_button, 1)

        refresh = QtWidgets.QPushButton("REFRESH")
        refresh.setToolTip("Re-read the _takes folder.")
        refresh.clicked.connect(self._refresh_take_history)
        row2.addWidget(refresh)
        layout.addLayout(row2)

        self.delete_take_button = QtWidgets.QPushButton("DELETE TAKE(S)")
        self.delete_take_button.setMinimumHeight(28)
        # Starts disabled rather than waiting for the first _refresh_controls
        # tick: there is nothing selected yet, and a delete button that is
        # live for 100 ms before it should be is still live.
        self.delete_take_button.setEnabled(False)
        self.delete_take_button.setToolTip(
            "Move the selected take(s) out of _takes and into _deleted. "
            "Nothing is erased, so a take can always be recovered. Clips "
            "already cut from a take are separate recordings and are left "
            "exactly as they are."
        )
        self.delete_take_button.clicked.connect(self.delete_selected_takes)
        layout.addWidget(self.delete_take_button)

        self._free_take_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Shift+R"), self)
        self._free_take_shortcut.activated.connect(
            self.free_take_button.toggle)

        return group

    def _group_markers(self):
        """
        Timestamped notes dropped onto the live stream, and cutting the span
        between the newest one and now into a training recording.
        """
        group = QtWidgets.QGroupBox("MARKERS & NOTES")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        self.marker_text = QtWidgets.QLineEdit()
        self.marker_text.setPlaceholderText(
            "what just happened? e.g. electrode slipped")
        self.marker_text.setToolTip(
            "Typed once, dropped with ADD NOTE. The text is drawn on the "
            "graph at the moment you pressed the button and written to "
            "markers.jsonl in the session folder."
        )
        self.marker_text.returnPressed.connect(self.add_note_marker)
        layout.addWidget(self.marker_text)

        row = QtWidgets.QHBoxLayout()
        self.add_note_button = QtWidgets.QPushButton("ADD NOTE")
        self.add_note_button.setObjectName("primary")
        self.add_note_button.setMinimumHeight(30)
        self.add_note_button.setToolTip(
            "Drop a note at this instant.  Ctrl+M")
        self.add_note_button.clicked.connect(self.add_note_marker)
        row.addWidget(self.add_note_button)

        self.add_break_button = QtWidgets.QPushButton("BREAK")
        self.add_break_button.setObjectName("warn")
        self.add_break_button.setMinimumHeight(30)
        self.add_break_button.setToolTip(
            "Drop a separator: a dashed line that visibly splits what came "
            "before from what comes next.  Ctrl+B")
        self.add_break_button.clicked.connect(self.add_break_marker)
        row.addWidget(self.add_break_button)
        layout.addLayout(row)

        self.marker_cut_check = QtWidgets.QCheckBox("allow cutting at markers")
        self.marker_cut_check.setChecked(bool(MARKER_CUT_ENABLED))
        self.marker_cut_check.setToolTip(
            "When on, CUT SINCE LAST MARKER turns the span between the "
            "newest marker and now into a labelled training recording, using "
            "the gesture chosen in LABEL ON GRAPH above."
        )
        self.marker_cut_check.toggled.connect(self._on_marker_cut_toggled)
        layout.addWidget(self.marker_cut_check)

        self.cut_marker_button = QtWidgets.QPushButton("CUT SINCE LAST MARKER")
        self.cut_marker_button.setMinimumHeight(30)
        self.cut_marker_button.setToolTip(
            "Save everything between the newest marker and now as training "
            "data.  Ctrl+Shift+Space")
        self.cut_marker_button.clicked.connect(self.cut_since_last_marker)
        layout.addWidget(self.cut_marker_button)

        self.marker_status = make_label("no markers yet", 10, False, COL_TEXT_DIM)
        self.marker_status.setWordWrap(True)
        layout.addWidget(self.marker_status)

        self.marker_list = QtWidgets.QListWidget()
        self.marker_list.setMaximumHeight(96)
        self.marker_list.setToolTip(
            "Markers logged this session, newest last. The log is "
            "append-only: a marker is never edited or deleted, a correction "
            "is simply another marker."
        )
        layout.addWidget(self.marker_list)

        self._note_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+M"), self)
        self._note_shortcut.activated.connect(self.add_note_marker)
        self._break_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+B"), self)
        self._break_shortcut.activated.connect(self.add_break_marker)
        self._cut_marker_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Shift+Space"), self)
        self._cut_marker_shortcut.activated.connect(self.cut_since_last_marker)

        self._sync_marker_controls()
        return group

    def _group_plan(self):
        group = QtWidgets.QGroupBox("RECORDING PLAN")
        grid = QtWidgets.QGridLayout(group)
        grid.setSpacing(6)

        grid.addWidget(make_label("Repetitions", 10, False, COL_TEXT_DIM), 0, 0)
        self.reps_spin = QtWidgets.QSpinBox()
        self.reps_spin.setRange(1, 999)
        self.reps_spin.setValue(REPETITIONS)
        grid.addWidget(self.reps_spin, 0, 1)

        grid.addWidget(make_label("Order", 10, False, COL_TEXT_DIM), 0, 2)
        self.order_combo = QtWidgets.QComboBox()
        self.order_combo.addItems(["interleaved", "blocked", "random"])
        self.order_combo.setCurrentText(GESTURE_ORDER)
        self.order_combo.setToolTip(
            "interleaved: one round of every gesture at a time -- spreads "
            "fatigue and electrode drift evenly across the label set"
        )
        grid.addWidget(self.order_combo, 0, 3)

        grid.addWidget(make_label("Manual gesture", 10, False, COL_TEXT_DIM), 1, 0)
        self.gesture_combo = QtWidgets.QComboBox()
        self.gesture_combo.addItems(GESTURES)
        self.gesture_combo.setToolTip("Label used by the manual RECORD button")
        grid.addWidget(self.gesture_combo, 1, 1)

        self.plan_label = make_label("", 10, False, COL_TEXT_DIM)
        self.plan_label.setWordWrap(True)
        grid.addWidget(self.plan_label, 2, 0, 1, 4)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("%v / %m")
        grid.addWidget(self.progress_bar, 3, 0, 1, 4)

        self.next_label = make_label("", 11, True, COL_TEXT)
        grid.addWidget(self.next_label, 4, 0, 1, 4)

        self.reps_spin.valueChanged.connect(self._refresh_plan_info)
        self.order_combo.currentTextChanged.connect(self._refresh_plan_info)
        return group

    def _group_channels(self):
        group = QtWidgets.QGroupBox("CHANNELS")
        grid = QtWidgets.QGridLayout(group)
        grid.setSpacing(3)
        grid.setColumnStretch(4, 1)

        headers = ["", "value", "min", "max", "rms"]
        for col, text in enumerate(headers):
            grid.addWidget(make_label(text, 9, False, COL_TEXT_DIM), 0, col)

        self.channel_labels = []
        for i in range(N_CHANNELS):
            name = make_label(CHANNEL_NAMES[i], 10, True, CHANNEL_COLORS[i])
            grid.addWidget(name, i + 1, 0)
            row = []
            for col in range(1, 5):
                value = make_label("0", 10, False, COL_TEXT)
                value.setFont(QtGui.QFont("Consolas", 9))
                grid.addWidget(value, i + 1, col)
                row.append(value)
            self.channel_labels.append(row)

        return group

    def _group_recordings(self):
        group = QtWidgets.QGroupBox("RECORDINGS (this run)")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        self.table = QtWidgets.QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["#", "Gesture", "Rep", "Dur", "Quality", "Trim"]
        )
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QtWidgets.QTableWidget.SelectRows)
        self.table.setSelectionMode(QtWidgets.QTableWidget.SingleSelection)
        self.table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        self.table.setMinimumHeight(190)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(1, QtWidgets.QHeaderView.Stretch)
        for col in (0, 2, 3, 4, 5):
            header.setSectionResizeMode(col, QtWidgets.QHeaderView.ResizeToContents)
        self.table.doubleClicked.connect(lambda _index: self.trim_selected())
        layout.addWidget(self.table)

        row = QtWidgets.QHBoxLayout()
        self.trim_button = QtWidgets.QPushButton("TRIM")
        self.trim_button.setToolTip("Inspect and cut the selected recording (double-click)")
        self.trim_button.clicked.connect(self.trim_selected)
        row.addWidget(self.trim_button)

        self.open_file_button = QtWidgets.QPushButton("OPEN FILE(S)...")
        self.open_file_button.setToolTip(
            "Pull up any recording or take already on disk -- including from "
            "an earlier session -- to look at its graph, annotate it, and cut "
            "training data out of it. Several files may be selected: each is "
            "opened separately and stays its own recording."
        )
        self.open_file_button.clicked.connect(self.open_recording_file)
        row.addWidget(self.open_file_button)
        layout.addLayout(row)

        row_b = QtWidgets.QHBoxLayout()
        self.autotrim_button = QtWidgets.QPushButton("AUTO-TRIM ALL")
        self.autotrim_button.setToolTip(
            "Run activity detection over every recording of this run and write "
            "the trimmed copies. Raw files are untouched."
        )
        self.autotrim_button.clicked.connect(self.auto_trim_all)
        row_b.addWidget(self.autotrim_button)
        layout.addLayout(row_b)

        row2 = QtWidgets.QHBoxLayout()
        self.delete_button = QtWidgets.QPushButton("DELETE LAST")
        self.delete_button.setObjectName("danger")
        self.delete_button.setToolTip(
            "Moves the last recording into the _deleted folder. Nothing is "
            "ever erased from disk."
        )
        self.delete_button.clicked.connect(self.delete_last)
        row2.addWidget(self.delete_button)

        self.clear_button = QtWidgets.QPushButton("CLEAR LIST")
        self.clear_button.setToolTip(
            "Empties this list and the live graph buffer. Files on disk stay."
        )
        self.clear_button.clicked.connect(self.clear_list)
        row2.addWidget(self.clear_button)
        layout.addLayout(row2)

        return group

    def _group_prediction(self):
        group = QtWidgets.QGroupBox("LIVE CLASSIFIER")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        # Mode selector
        row = QtWidgets.QHBoxLayout()
        row.addWidget(make_label("Mode", 10, False, COL_TEXT_DIM))
        self.pred_mode_combo = QtWidgets.QComboBox()
        self.pred_mode_combo.addItem("One shot (stop after each)", "one_shot")
        self.pred_mode_combo.addItem("Continuous", "continuous")
        index = self.pred_mode_combo.findData(PREDICTION_MODE)
        self.pred_mode_combo.setCurrentIndex(max(0, index))
        self.pred_mode_combo.setToolTip(
            "One shot: one prediction per press, then it waits.\n"
            "Continuous: keeps classifying until you press STOP."
        )
        self.pred_mode_combo.currentIndexChanged.connect(self._on_pred_mode_changed)
        row.addWidget(self.pred_mode_combo, 1)
        layout.addLayout(row)

        # Result
        self.prediction_label = make_label("--", 26, True, COL_TEXT_DIM)
        self.prediction_label.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.prediction_label)

        self.prediction_conf = make_label("", 13, True, COL_TEXT_DIM)
        self.prediction_conf.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(self.prediction_conf)

        # Per-class probability bars -- the honest view of how sure it is.
        self.prob_bars = {}
        self.prob_box = QtWidgets.QWidget()
        bars = QtWidgets.QGridLayout(self.prob_box)
        bars.setContentsMargins(0, 2, 0, 2)
        bars.setSpacing(2)
        bars.setColumnStretch(1, 1)
        layout.addWidget(self.prob_box)

        self.prediction_note = make_label("", 10, False, COL_TEXT_DIM)
        self.prediction_note.setWordWrap(True)
        layout.addWidget(self.prediction_note)

        buttons = QtWidgets.QHBoxLayout()
        self.predict_button = QtWidgets.QPushButton("PREDICT")
        self.predict_button.setObjectName("primary")
        self.predict_button.setMinimumHeight(34)
        self.predict_button.clicked.connect(self.toggle_prediction)
        buttons.addWidget(self.predict_button, 2)

        self.stats_button = QtWidgets.QPushButton("MODEL STATS")
        self.stats_button.setMinimumHeight(34)
        self.stats_button.clicked.connect(self.show_model_stats)
        buttons.addWidget(self.stats_button, 1)
        layout.addLayout(buttons)

        self.retrain_button = QtWidgets.QPushButton("RETRAIN ON ALL DATA")
        self.retrain_button.setMinimumHeight(30)
        self.retrain_button.setToolTip(
            "Retrain using EVERY recording on disk -- the ones you just made "
            "plus all earlier sessions and people. New data is added to the "
            "old, never trained on by itself."
        )
        self.retrain_button.clicked.connect(self.retrain_model)
        layout.addWidget(self.retrain_button)

        self.spec_button = QtWidgets.QPushButton("MODEL SPEC...")
        self.spec_button.setMinimumHeight(30)
        self.spec_button.setToolTip(
            "Choose which classifiers the next retrain sweeps, set their "
            "hyperparameters, and decide whether they also vote as an "
            "ensemble.  Ctrl+Shift+M"
        )
        self.spec_button.clicked.connect(self.edit_model_spec)
        layout.addWidget(self.spec_button)

        self.spec_summary = make_label("", FS_SM, False, COL_TEXT_DIM)
        self.spec_summary.setWordWrap(True)
        layout.addWidget(self.spec_summary)

        #  Is a model trained, on what, and is it still current? The detail
        #  goes in the tooltip and in MODEL STATS; this is the headline.
        self.model_status_label = make_label("", FS_SM, True, COL_TEXT_DIM)
        self.model_status_label.setWordWrap(True)
        layout.addWidget(self.model_status_label)

        self.model_data_label = make_label("", FS_SM, False, COL_TEXT_DIM)
        self.model_data_label.setWordWrap(True)
        layout.addWidget(self.model_data_label)

        #  Which processor the training actually runs on. Filled in by the
        #  background probe; never guessed.
        self.compute_label = make_label("", FS_SM, True, COL_TEXT_DIM)
        self.compute_label.setWordWrap(True)
        layout.addWidget(self.compute_label)
        self._apply_compute_badge()
        self._refresh_model_status(force=True)

        layout.addWidget(make_label("TRAINING DATA ON DISK", FS_SM, True,
                                    COL_TEXT_DIM))
        self.gesture_counts_label = QtWidgets.QLabel()
        self.gesture_counts_label.setWordWrap(True)
        self.gesture_counts_label.setTextFormat(QtCore.Qt.RichText)
        self.gesture_counts_label.setToolTip(
            "Raw recordings per gesture across the whole dataset -- what the "
            "trainer will actually see. A gesture in grey has none yet.")
        layout.addWidget(self.gesture_counts_label)

        self._spec_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Shift+M"), self)
        self._spec_shortcut.activated.connect(self.edit_model_spec)

        self.training_progress = QtWidgets.QProgressBar()
        self.training_progress.setRange(0, 1000)
        self.training_progress.setTextVisible(True)
        self.training_progress.setFormat("%p%")
        self.training_progress.setFixedHeight(14)
        self.training_progress.setVisible(False)
        layout.addWidget(self.training_progress)

        self.training_status = make_label("", 10, False, COL_TEXT_DIM)
        self.training_status.setWordWrap(True)
        layout.addWidget(self.training_status)

        self._rebuild_prob_bars()
        self._refresh_prediction_panel()
        return group

    def _rebuild_prob_bars(self) -> None:
        """One labelled bar per gesture the loaded model knows about."""
        grid = self.prob_box.layout()
        while grid.count():
            item = grid.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.prob_bars = {}

        labels = self.predictor.labels if self.predictor.ready else []
        for row, name in enumerate(labels):
            caption = make_label(name, 9, False, COL_TEXT_DIM)
            bar = QtWidgets.QProgressBar()
            bar.setRange(0, 1000)
            bar.setValue(0)
            bar.setTextVisible(False)
            bar.setFixedHeight(12)
            value = make_label("0 %", 9, False, COL_TEXT_DIM)
            value.setFixedWidth(38)
            value.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
            grid.addWidget(caption, row, 0)
            grid.addWidget(bar, row, 1)
            grid.addWidget(value, row, 2)
            self.prob_bars[name] = (bar, value)
        self.prob_box.setVisible(bool(labels))

    def _group_training(self):
        """
        What the trainer is doing right now.

        Everything here comes from `TRAINEVENT` lines the trainer subprocess
        prints as it finishes real work -- no field is estimated except the
        ETA, which is derived from completions already observed and says so.
        With nothing running it reads "idle", not a stale percentage.
        """
        group = QtWidgets.QGroupBox("TRAINING")
        layout = QtWidgets.QVBoxLayout(group)
        layout.setSpacing(6)

        self.train_job_label = make_label("idle", FS_SM, True, COL_TEXT_DIM)
        self.train_job_label.setWordWrap(True)
        layout.addWidget(self.train_job_label)

        self.train_bar = QtWidgets.QProgressBar()
        self.train_bar.setRange(0, 1000)
        self.train_bar.setTextVisible(True)
        self.train_bar.setFormat("%p%")
        self.train_bar.setFixedHeight(18)
        layout.addWidget(self.train_bar)

        grid = QtWidgets.QGridLayout()
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(3)
        self.train_fields = {}
        for row, (key, caption) in enumerate((
                ("status", "status"),
                ("stage", "stage"),
                ("model", "algorithm"),
                ("setup", "preprocessing"),
                ("candidate", "candidate"),
                ("folds", "folds"),
                ("device", "device"),
                ("elapsed", "elapsed"),
                ("eta", "remaining"),
                ("artifact", "saved to"),
        )):
            grid.addWidget(make_label(caption, FS_SM, False, COL_TEXT_DIM),
                           row, 0)
            value = make_label("--", FS_SM, False, COL_TEXT)
            value.setWordWrap(True)
            grid.addWidget(value, row, 1)
            self.train_fields[key] = value
        grid.setColumnStretch(1, 1)
        layout.addLayout(grid)

        self.train_message = make_label("", FS_SM, False, COL_TEXT_DIM)
        self.train_message.setWordWrap(True)
        layout.addWidget(self.train_message)
        return group

    # -- training progress ----------------------------------------------------

    def _reset_training_panel(self, job: str = "idle") -> None:
        if not hasattr(self, "train_fields"):
            return
        self.train_job_label.setText(job)
        self.train_bar.setValue(0)
        self.train_bar.setFormat("%p%")
        for value in self.train_fields.values():
            value.setText("--")
        self.train_message.setText("")
        self._set_train_field("status", "queued", COL_WARN)

    def _set_train_field(self, key: str, text: str, colour: str = None) -> None:
        widget = getattr(self, "train_fields", {}).get(key)
        if widget is None:
            return
        widget.setText(text)
        widget.setStyleSheet(
            f"color: {colour or COL_TEXT}; background: transparent;")

    def _on_train_event(self, event: dict) -> None:
        """One TRAINEVENT line from the trainer, straight onto the panel."""
        if not hasattr(self, "train_fields"):
            return
        # Every number here came off a pipe, so none of it is trusted: a
        # malformed event should leave the panel as it was, not raise inside
        # the output pump.
        def number(key, fallback=0):
            try:
                return int(float(event.get(key, fallback)))
            except (TypeError, ValueError):
                return fallback

        done = number("done", 0)
        total = max(1, number("total", 1))
        fraction = min(1.0, max(0.0, done / total))
        self.train_bar.setValue(int(fraction * 1000))

        status = str(event.get("status", "running"))
        colour = {"completed": COL_GOOD, "failed": COL_BAD,
                  "cancelled": COL_WARN, "running": COL_ACCENT}.get(
                      status, COL_TEXT)
        self._set_train_field("status", status, colour)
        self._set_train_field("stage", str(event.get("phase", "--")))
        self._set_train_field("model", str(event.get("model", "--")))
        self._set_train_field("candidate", f"{done} / {total}")
        self._set_train_field(
            "device", str(event.get("device", "cpu")).upper(),
            COL_GOOD if str(event.get("device")) != "cpu" else COL_TEXT)

        try:
            window_s = float(event["window_s"])
        except (KeyError, TypeError, ValueError):
            window_s = None
        if window_s is not None:
            self._set_train_field(
                "setup",
                f"{window_s:.2f}s  ov {event.get('overlap', '?')}  "
                f"{event.get('representation', '?')}/"
                f"{event.get('feature_set', '?')}  "
                f"({event.get('n_windows', '?')} windows x "
                f"{event.get('n_features', '?')} feat)")
        if event.get("n_folds") is not None:
            self._set_train_field("folds", f"{event['n_folds']} per candidate")

        elapsed = time.perf_counter() - self.training_started_at
        self._set_train_field("elapsed", fmt_duration(elapsed))
        # ETA from the rate actually achieved, never a fixed guess, and only
        # once enough work is done for the rate to mean anything.
        if done >= 5 and fraction > 0.01 and status == "running":
            remaining = max(0.0, elapsed / fraction - elapsed)
            self._set_train_field("eta", f"about {fmt_duration(remaining)}")
            self.train_bar.setFormat(f"%p%  ({fmt_duration(remaining)} left)")
        elif status == "running":
            self._set_train_field("eta", "estimating...")

        if event.get("artifact"):
            self._set_train_field("artifact", str(event["artifact"]), COL_GOOD)
        if event.get("message"):
            self.train_message.setText(str(event["message"]))

        if status == "completed":
            self._set_train_field("eta", "done")
            self.train_bar.setFormat("%p%  complete")
            try:
                vote = float(event["recording_vote_accuracy"])
            except (KeyError, TypeError, ValueError):
                vote = None
            if vote is not None:
                self.train_message.setText(
                    f"{event.get('model', 'model')}: "
                    f"{vote * 100:.1f} % per gesture, "
                    f"{float(event.get('window_accuracy') or 0) * 100:.1f} "
                    "% per window")

    def _group_log(self):
        group = QtWidgets.QGroupBox("LOG")
        layout = QtWidgets.QVBoxLayout(group)
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(500)
        self.log_box.setMinimumHeight(110)
        self.log_box.setFont(QtGui.QFont("Consolas", 8))
        layout.addWidget(self.log_box)
        return group

    def _build_button_bar(self):
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {COL_PANEL}; border: 1px solid {COL_BORDER};"
            f" border-radius: 8px; }}"
        )
        row = QtWidgets.QHBoxLayout(frame)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(8)

        self.start_button = QtWidgets.QPushButton("START SESSION")
        self.start_button.setObjectName("primary")
        self.start_button.setMinimumHeight(38)
        self.start_button.clicked.connect(self.start_session)

        self.pause_button = QtWidgets.QPushButton("PAUSE")
        self.pause_button.setMinimumHeight(38)
        self.pause_button.clicked.connect(self.toggle_pause)

        self.stop_button = QtWidgets.QPushButton("STOP")
        self.stop_button.setObjectName("danger")
        self.stop_button.setMinimumHeight(38)
        self.stop_button.clicked.connect(self.stop_session)

        self.skip_button = QtWidgets.QPushButton("SKIP")
        self.skip_button.setMinimumHeight(38)
        self.skip_button.clicked.connect(self.skip_current)

        self.accept_button = QtWidgets.QPushButton("ACCEPT && NEXT")
        self.accept_button.setObjectName("primary")
        self.accept_button.setMinimumHeight(38)
        self.accept_button.setToolTip(
            "Keep this take (writing the trimmed copy for the green band) and "
            "start the next repetition.  Space, Enter or Ctrl+Enter"
        )
        self.accept_button.clicked.connect(self.accept_review)

        self.redo_button = QtWidgets.QPushButton("REDO")
        self.redo_button.setObjectName("warn")
        self.redo_button.setMinimumHeight(38)
        self.redo_button.setToolTip(
            "Reject this take and record the same gesture again. The rejected "
            "file is moved to _deleted, never erased.  Ctrl+R"
        )
        self.redo_button.clicked.connect(self.redo_review)

        self.record_button = QtWidgets.QPushButton("RECORD (manual)")
        self.record_button.setMinimumHeight(38)
        self.record_button.setCheckable(True)
        self.record_button.setToolTip(
            "Record the gesture selected above until pressed again"
        )
        self.record_button.clicked.connect(self.toggle_manual_record)

        self.graph_button = QtWidgets.QPushButton("SAVE GRAPH")
        self.graph_button.setMinimumHeight(38)
        self.graph_button.clicked.connect(self.save_graph)

        self.export_button = QtWidgets.QPushButton("EXPORT DATASET")
        self.export_button.setMinimumHeight(38)
        self.export_button.setToolTip(
            "Build a combined windows + features bundle from everything on disk"
        )
        self.export_button.clicked.connect(self.export_dataset)

        for button in (self.start_button, self.accept_button, self.redo_button,
                       self.pause_button, self.stop_button, self.skip_button,
                       self.record_button):
            row.addWidget(button)
        row.addStretch(1)
        for button in (self.graph_button, self.export_button):
            row.addWidget(button)

        # Accept is driven by giving the button focus when the review starts,
        # so Space/Enter activate it the normal Qt way. The explicit shortcuts
        # use a modifier on purpose: a bare "Space" or "R" window shortcut
        # would swallow those characters when you are typing in Notes.
        self._accept_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Return"), self)
        self._accept_shortcut.activated.connect(self.accept_review)
        self._redo_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self)
        self._redo_shortcut.activated.connect(self.redo_review)

        return frame

    # ======================================================================
    # Timers
    # ======================================================================

    def _start_timers(self) -> None:
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.plot_timer.timeout.connect(self._tick_plot)
        self.plot_timer.start(max(8, int(1000 / max(1, GUI_FPS))))

        self.seq_timer = QtCore.QTimer(self)
        self.seq_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.seq_timer.timeout.connect(self._tick_sequence)
        self.seq_timer.start(20)

        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.timeout.connect(self._tick_slow_ui)
        self.ui_timer.start(100)

        # Prediction runs on its own timer so its rate is independent of the
        # graph frame rate and of the recording state machine.
        self.pred_timer = QtCore.QTimer(self)
        self.pred_timer.timeout.connect(self._tick_prediction)

        # Ask what hardware the training will run on. Deferred so the window
        # is already on screen and streaming before the probe starts.
        QtCore.QTimer.singleShot(400, self.start_compute_probe)

    def _tick_plot(self) -> None:
        if self.plot.review_mode:
            return          # the graph is showing a finished take, not the stream

        n = int(self.plot.window_s * max(1.0, self.acq.stats()["rate"]
                                         or nominal_rate())) + 8
        emg, t = self.acq.ring.snapshot(n)
        if emg.shape[0] < 2:
            return
        self.plot.update_data(emg, t)

        # Either kind of capture gets the recording band: a free take is
        # every bit as much "this is being written down" as a cued take.
        capture = self.sequence.active_capture or self.free_capture
        if capture is not None and len(capture) > 1:
            span = len(capture) / max(1.0, self.acq.stats()["rate"] or nominal_rate())
            self.plot.set_recording_span(-min(span, self.plot.window_s), 0.0, True)
        else:
            self.plot.set_recording_span(0, 0, False)

    def _tick_sequence(self) -> None:
        self.sequence.tick()
        self._refresh_banner()

    def _tick_slow_ui(self) -> None:
        stats = self.acq.stats()

        self.status_conn.setText(f"{self.acq.state}")
        self.status_rate.setText(f"{stats['rate']:.0f} Hz")
        self.status_buffer.setText(f"buffer {stats['ring_fill'] * 100:.0f} %")
        # Show the input backlog: this is the number that reveals a lagging
        # link, and a rate readout alone cannot (a constant delay does not
        # change the spacing between samples).
        backlog = stats.get("backlog", 0)
        lag_s = backlog / max(1.0, 25.0 * (stats["rate"] or nominal_rate()))
        self.status_counts.setText(
            f"samples {stats['samples']:,}   dropped {stats['dropped']}   "
            f"reconnects {stats['reconnects']}   "
            f"lag {lag_s:.1f}s ({backlog}B)"
        )
        self.status_counts.setStyleSheet(
            f"color: {COL_BAD if lag_s > 1.0 else COL_WARN if lag_s > 0.35 else COL_TEXT_DIM};"
            " padding: 0 10px;"
        )

        last, mins, maxs, rms = self.plot.channel_stats()
        for i in range(N_CHANNELS):
            self.channel_labels[i][0].setText(f"{int(last[i])}")
            self.channel_labels[i][1].setText(f"{int(mins[i])}")
            self.channel_labels[i][2].setText(f"{int(maxs[i])}")
            self.channel_labels[i][3].setText(f"{rms[i]:.1f}")

        self._tick_free_take()
        # Throttled inside: a recording saved from anywhere -- the sequence,
        # a label-on-graph cut, a clip out of a take -- makes the model one
        # recording more out of date, and this is what notices.
        self._refresh_model_status()

        self.header_count.setText(str(len(self.recordings)))
        item = self.sequence.current
        self.header_gesture.setText(item[0] if item else
                                    self.gesture_combo.currentText())
        self.header_person.setText(f"{self.person_id} / {self.session_id}")

        done, total = self.sequence.progress
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(done)

        self._refresh_controls()

        # Clear a held one-shot result once its time is up.
        if self.one_shot_clear_at and time.perf_counter() > self.one_shot_clear_at:
            self.one_shot_clear_at = 0.0
            self.plot.set_prediction_overlay("")
            self.prediction_conf.setText("ready")

        self._update_plot_title()

    def _refresh_controls(self) -> None:
        """
        Enable/disable every control for the current state.

        Called from the 100 ms UI timer and directly by anything that changes
        state, so a button never stays wrong for a tenth of a second after the
        thing it controls has already happened.
        """
        running = self.sequence.running and not self.sequence.manual
        reviewing = self.sequence.reviewing
        # A free take owns the acquisition thread's single capture slot, so
        # nothing else may open a capture while one is running.
        taking = self.free_capture is not None
        self.start_button.setEnabled(not self.sequence.running and not taking)
        self.pause_button.setEnabled((running or self.sequence.paused)
                                     and not reviewing)
        self.pause_button.setText("RESUME" if self.sequence.paused else "PAUSE")
        self.stop_button.setEnabled(self.sequence.running)
        self.skip_button.setEnabled(running and not reviewing)
        self.record_button.setEnabled(not running and not self.sequence.paused
                                      and not reviewing and not taking)
        self.accept_button.setEnabled(reviewing)
        self.redo_button.setEnabled(reviewing)
        if FREE_TAKE_ENABLED and hasattr(self, "free_take_button"):
            self.free_take_button.setEnabled(not self.sequence.running)
            self.take_gesture_combo.setEnabled(not taking)
            self.open_take_button.setEnabled(not taking)
            # Deleting the take that is still being recorded would pull the
            # file out from under save_take(), so it waits until it is saved.
            self.delete_take_button.setEnabled(
                not taking and self.take_list.count() > 0
                and self.take_list.currentItem() is not None
                and bool(self.take_list.currentItem().data(
                    QtCore.Qt.UserRole + 1)))
        labelling = self.label_mode_button.isChecked()
        self.label_mode_button.setEnabled(not self.sequence.running
                                          or labelling)
        self.select_button.setEnabled(labelling)
        self.add_label_button.setEnabled(
            labelling and self.plot.selection_bounds() is not None)
        self.undo_label_button.setEnabled(labelling and bool(self.plot.marks))
        for widget in (self.person_edit, self.session_edit, self.reps_spin,
                       self.order_combo):
            widget.setEnabled(not self.sequence.running)

        # ---- visibility ------------------------------------------------
        # Enabling is not enough: eight permanently visible buttons, six of
        # them greyed out, is a wall to read past every time. A control that
        # cannot be used in this mode is not shown at all, so what is on
        # screen is exactly what you can do right now.
        idle = not self.sequence.running and not taking
        self.start_button.setVisible(idle)
        self.record_button.setVisible(idle)
        self.pause_button.setVisible(running or self.sequence.paused)
        self.stop_button.setVisible(self.sequence.running)
        self.skip_button.setVisible(running and not reviewing)
        self.accept_button.setVisible(reviewing)
        self.redo_button.setVisible(reviewing)

        # The banner is for "what should I do right now". When nothing is
        # happening it says READY, which is noise occupying 130 px above the
        # graph -- so it is hidden until it has something to report.
        self.banner.setVisible(
            self.sequence.running
            or taking
            or self.acq.state in (CONN_RECONNECTING, CONN_CONNECTING,
                                  CONN_ERROR)
        )

        self.trim_button.setEnabled(self.table.currentRow() >= 0)
        self.autotrim_button.setEnabled(bool(self.recordings))
        self.delete_button.setEnabled(bool(self.recordings))
        self.clear_button.setEnabled(bool(self.recordings))
        self.export_button.setEnabled(self.export_worker is None)
        self.retrain_button.setEnabled(
            not self.sequence.running or self.training_process is not None)
        if hasattr(self, "spec_button"):
            # Editing the spec mid-training would be misleading: the running
            # sweep already has its copy.
            self.spec_button.setEnabled(self.training_process is None)

    # ======================================================================
    # Connection
    # ======================================================================

    def _set_pill(self, text: str, color: str) -> None:
        self.conn_pill.setText(text)
        self.conn_pill.setStyleSheet(
            f"QLabel {{ background: {color}22; color: {color};"
            f" border: 1px solid {color}; border-radius: 11px;"
            f" padding: 4px 12px; font-weight: 700; letter-spacing: 1px; }}"
        )

    def _on_connection_state(self, state: str, message: str) -> None:
        colors = {
            CONN_CONNECTED: COL_GOOD,
            CONN_CONNECTING: COL_WARN,
            CONN_RECONNECTING: COL_WARN,
            CONN_DISCONNECTED: COL_TEXT_DIM,
            CONN_STOPPED: COL_TEXT_DIM,
            CONN_ERROR: COL_BAD,
        }
        text = "MYO " + state
        self._set_pill(text, colors.get(state, COL_TEXT_DIM))
        self._append_log(f"{state}: {message}" if message else state)
        self.status.showMessage(message or state, 6000)

        if not PAUSE_SESSION_ON_LINK_LOSS:
            return

        # A session must not keep counting repetitions into a dead link:
        # it would quietly fill the dataset with empty recordings.
        if state in (CONN_RECONNECTING, CONN_ERROR):
            if self.sequence.running and not self.sequence.paused \
                    and not self.sequence.reviewing:
                self._append_log("Link lost -- session paused automatically.")
                self.sequence.last_message = "Myo disconnected"
                self._link_paused = True
                # Throw away the partial recording and retry the repetition
                # rather than banking an incomplete sample.
                self.sequence.pause(discard_in_flight=True)

        # ...and it should pick up again by itself once the armband is back,
        # so an unattended subject is not left staring at a frozen screen.
        elif state == CONN_CONNECTED and self._link_paused:
            self._link_paused = False
            if self.sequence.paused:
                self._append_log("Myo reconnected -- resuming the session.")
                self.sequence.resume()

    def _append_log(self, text: str) -> None:
        self.log_box.appendPlainText(f"{datetime.now().strftime('%H:%M:%S')}  {text}")

    # ======================================================================
    # Identity / plan
    # ======================================================================

    @property
    def fitted_to_screen(self) -> bool:
        """True when the preferred window size did not fit on this display."""
        want = (WINDOW_WIDTH, WINDOW_HEIGHT)
        return tuple(getattr(self, "_fitted_size", want)) != want

    def _set_global(self, name: str, value) -> None:
        """
        Mirror a GUI field into the module-level config.

        The config block is the single source of truth for what gets written
        into session_config.json and every *_meta.json, so an edit in the GUI
        has to land there too -- otherwise the stored metadata would lie.
        """
        globals()[name] = value

    def _on_identity_changed(self) -> None:
        if self.sequence.running:
            return
        person = safe_slug(self.person_edit.text(), "person")
        session = safe_slug(self.session_edit.text(), "session")
        if (person, session) == (self.person_id, self.session_id):
            self._refresh_plan_info()
            return

        self.person_id, self.session_id = person, session
        self.person_edit.setText(person)
        self.session_edit.setText(session)
        self._set_global("PERSON_ID", person)
        self._set_global("SESSION_ID", session)
        self.writer = DatasetWriter(
            os.path.join(PROJECT_DIR, DATA_FOLDER), person, session
        )
        existing = self.writer.count_existing()
        self._append_log(
            f"Subject set to {person} / {session} "
            f"({existing} recording(s) already on disk)."
        )
        # Markers belong to a person/session pair, so switching subject has to
        # swap the marker log too -- otherwise notes would be drawn over, and
        # cut from, somebody else's session. Takes are per-session as well.
        self._load_session_markers()
        self._refresh_take_history()
        self._refresh_plan_info()

    def _on_metadata_changed(self) -> None:
        self._set_global("HANDEDNESS", self.hand_combo.currentText())
        self._set_global("ARM_WORN_ON", self.arm_combo.currentText())
        self._set_global("DEVICE_NAME", self.device_edit.text())
        self._set_global("SUBJECT_NOTES", self.notes_edit.text())

    def _refresh_plan_info(self) -> None:
        self._set_global("REPETITIONS", int(self.reps_spin.value()))
        self._set_global("GESTURE_ORDER", self.order_combo.currentText())
        plan = build_plan()
        per_item = (REST_TIME + PREPARE_TIME + COUNTDOWN_TIME + RECORD_TIME
                    + POST_GESTURE_MS / 1000.0 + PAUSE_TIME)
        self.plan_label.setText(
            f"{len(GESTURES)} gestures x {self.reps_spin.value()} reps = "
            f"{len(plan)} recordings,  about {fmt_duration(len(plan) * per_item)} "
            f"({per_item:.1f} s each).  Gestures: {', '.join(GESTURES)}"
        )
        self.folder_label.setText(
            os.path.relpath(self.writer.session_dir, PROJECT_DIR)
            + f"   ({self.writer.count_existing()} on disk)"
        )
        self.progress_bar.setMaximum(max(1, len(plan)))

    # ======================================================================
    # Banner
    # ======================================================================

    def _refresh_banner(self) -> None:
        if self.acq.state in (CONN_RECONNECTING, CONN_CONNECTING) and \
                (not self.sequence.running or self._link_paused):
            # A dead link outranks everything else: whatever is "recording"
            # is recording nothing. The open take is still named, so it is
            # clear that it was not the take that failed.
            message = self.acq.state_message or "Reconnecting..."
            if self.free_capture is not None:
                message = (f"free take {self.free_take_number:03d} is open and "
                           f"receiving nothing -- {message}")
            self.banner.update_state(
                STATE_RECONNECTING, "MYO DISCONNECTED", message, None,
            )
            self.plot.set_overlay("")
            return

        # A free take owns the banner while it runs: the sequence is idle, so
        # its own banner would read READY while the armband is recording.
        if self.free_capture is not None and not self.sequence.running:
            elapsed = time.time() - self.free_capture.started_unix
            cap = max(1.0, FREE_TAKE_MAX_MINUTES * 60.0)
            self.banner.update_state(
                STATE_RECORDING, "FREE TAKE -- RECORDING",
                f"{self.free_take_label}   {fmt_duration(elapsed)} of "
                f"{fmt_duration(cap)} max   {len(self.free_capture):,} samples"
                "   --   press STOP & SAVE TAKE when you are done",
                min(1.0, elapsed / cap),
            )
            self.plot.set_overlay("REC", "rgba(239,68,68,150)")
            return

        state = self.sequence.state
        big, small = self.sequence.banner()

        fraction = None
        if self.sequence.state_duration > 0 and not self.sequence.paused:
            fraction = self.sequence.elapsed / self.sequence.state_duration
        if state == STATE_RECORDING and self.sequence.manual:
            small = f"{self.sequence.current[0]}   {self.sequence.elapsed:.1f} s"

        self.banner.update_state(state, big, small, fraction)

        if state == STATE_COUNTDOWN:
            self.plot.set_overlay(big, "rgba(251,146,60,220)")
        elif state == STATE_RECORDING:
            self.plot.set_overlay("REC", "rgba(239,68,68,150)")
        elif state == STATE_PREPARE:
            self.plot.set_overlay(self.sequence.current[0]
                                  if self.sequence.current else "",
                                  "rgba(251,191,36,180)")
        else:
            self.plot.set_overlay("")

    def _update_plot_title(self) -> None:
        item = self.sequence.current
        gesture = item[0] if item else self.gesture_combo.currentText()
        self.plot.set_title(
            f"{APP_NAME}  |  {self.person_id} / {self.session_id}  |  "
            f"gesture: {gesture}  |  mode: {EMG_MODE} "
            f"@ {self.acq.stats()['rate']:.0f} Hz  |  "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # ======================================================================
    # Session control
    # ======================================================================

    def start_session(self) -> None:
        if not self.acq.connected:
            answer = QtWidgets.QMessageBox.question(
                self, "Myo not connected",
                f"The armband is currently {self.acq.state}.\n\n"
                "Start the session anyway? Recordings made while the link is "
                "down will be flagged as incomplete.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return

        self._on_identity_changed()
        self._on_metadata_changed()
        self._refresh_plan_info()

        self.writer.write_session_config({
            "started_iso": iso_time(time.time()),
            "plan_length": len(build_plan()),
        })

        self.expand_section("plan")
        self.sequence.start(build_plan())
        self._append_log(
            f"Session started: {len(self.sequence.plan)} recordings "
            f"({self.order_combo.currentText()})."
        )

    def toggle_pause(self) -> None:
        if self.sequence.paused:
            self.sequence.resume()
            self._append_log("Session resumed.")
        else:
            self.sequence.pause()
            self._append_log("Session paused.")

    def stop_session(self) -> None:
        if self.record_button.isChecked():
            self.record_button.setChecked(False)
        self.sequence.stop()
        self.acq.stop_capture()
        self._append_log("Session stopped.")

    def skip_current(self) -> None:
        item = self.sequence.current
        self.sequence.skip()
        if item:
            self._append_log(f"Skipped {item[0]} rep {item[1]}.")

    def toggle_manual_record(self, checked: bool) -> None:
        if checked:
            label = self.gesture_combo.currentText()
            repetition = self.writer.next_repetition(label)
            self.sequence.start_manual(label, repetition)
            self.record_button.setText("STOP RECORDING")
            self._append_log(f"Manual recording: {label} rep {repetition}.")
        else:
            self.sequence.stop_manual()
            self.record_button.setText("RECORD (manual)")

    # ======================================================================
    # Capture callbacks (called from the sequence controller, GUI thread)
    # ======================================================================

    def _on_sequence_state(self, state: str, previous: str) -> None:
        if state == STATE_RECORDING:
            self.acq.vibrate(1)
        elif previous == STATE_RECORDING:
            self.acq.vibrate(1)

        if previous == STATE_REVIEW and state != STATE_REVIEW:
            self.plot.exit_review()
            self.review_recording = None
            self.review_info.setText("")

        # A manual recording that has been reviewed leaves the button stuck in
        # its "STOP RECORDING" state otherwise.
        if previous == STATE_RECORDING and state != STATE_RECORDING \
                and self.record_button.isChecked():
            self.record_button.setChecked(False)
            self.record_button.setText("RECORD (manual)")

    # ======================================================================
    # Review step
    # ======================================================================

    def _begin_review(self, rec: Recording) -> None:
        """
        Freeze the graph on the take that just finished and wait for a verdict.

        The live stream keeps running into the ring buffer behind this, so
        nothing is lost while you look.
        """
        self.review_recording = rec
        bounds = self.plot.review(rec, rec.activity)
        self._update_review_info(*bounds)
        # Focus the button so Space/Enter accept, without stealing those keys
        # from the text fields the rest of the time.
        self.accept_button.setFocus(QtCore.Qt.OtherFocusReason)

    def _update_review_info(self, start_s: float = None,
                            end_s: float = None) -> None:
        rec = self.review_recording
        if rec is None:
            self.review_info.setText("")
            return

        start, end = self.plot.trim_indices(rec.n_samples)
        kept = end - start
        fraction = kept / max(1, rec.n_samples) * 100
        issues = ", ".join(sorted({i.code for i in rec.issues})) or "ok"
        self.review_info.setText(
            f"{rec.label} rep {rec.repetition}:  keeping {kept} of "
            f"{rec.n_samples} samples ({fraction:.0f} %), "
            f"{kept / max(rec.fs, 1.0):.2f} s   |   quality: {issues}"
        )

    def accept_review(self) -> None:
        """Keep the take, write the trimmed copy, go to the next repetition."""
        if not self.sequence.reviewing:
            return
        rec = self.review_recording

        if rec is not None and REVIEW_SAVE_TRIM_ON_ACCEPT:
            start, end = self.plot.trim_indices(rec.n_samples)
            # Writing a "trimmed" file identical to the raw one is just noise.
            if end - start >= 2 and (start > 0 or end < rec.n_samples):
                trimmed = rec.slice(start, end, STAGE_TRIMMED, {
                    "trim_method": "review",
                    "trim_reviewed_by_operator": True,
                })
                try:
                    self.writer.save(trimmed, STAGE_TRIMMED)
                    self._mark_table_trimmed(rec)
                    self._append_log(
                        f"{rec.label} rep {rec.repetition}: accepted, trimmed "
                        f"to {trimmed.n_samples} samples."
                    )
                except OSError as exc:
                    self._append_log(f"could not write the trimmed copy: {exc}")
                    QtWidgets.QMessageBox.warning(
                        self, "Trim",
                        f"Could not write the trimmed copy:\n\n{exc}\n\n"
                        "The raw recording is safe on disk.",
                    )
            else:
                self._append_log(
                    f"{rec.label} rep {rec.repetition}: accepted, kept in full."
                )
        elif rec is not None:
            self._append_log(f"{rec.label} rep {rec.repetition}: accepted.")

        self.sequence.accept_review()

    def redo_review(self) -> None:
        """Reject the take, quarantine its files, record the same one again."""
        if not self.sequence.reviewing:
            return
        rec = self.review_recording

        if rec is not None:
            if REVIEW_QUARANTINE_ON_REDO:
                moved = self._quarantine(rec)
                self._append_log(
                    f"{rec.label} rep {rec.repetition}: REDO -- "
                    f"{moved} file(s) moved to _deleted."
                )
            else:
                self._append_log(
                    f"{rec.label} rep {rec.repetition}: REDO -- "
                    "rejected take kept on disk."
                )
            self._remove_recording(rec)

        self.sequence.redo_review()
        self._refresh_plan_info()

    def _mark_table_trimmed(self, rec: Recording) -> None:
        if rec in self.recordings:
            item = self.table.item(self.recordings.index(rec), 5)
            if item:
                item.setText("yes")

    def _remove_recording(self, rec: Recording) -> None:
        if rec not in self.recordings:
            return
        row = self.recordings.index(rec)
        self.recordings.pop(row)
        self.table.removeRow(row)
        for i in range(self.table.rowCount()):
            item = self.table.item(i, 0)
            if item:
                item.setText(str(i + 1))

    def _on_sequence_finished(self) -> None:
        self._append_log(f"Session complete: {len(self.recordings)} recordings.")
        QtWidgets.QMessageBox.information(
            self, "Session complete",
            f"Collected {len(self.recordings)} recordings for "
            f"{self.person_id} / {self.session_id}.\n\n"
            f"{self.writer.session_dir}",
        )

    def _on_capture_start(self, label: str, repetition: int) -> Capture:
        return self.acq.start_capture(label, repetition, meta={
            "emg_mode": EMG_MODE,
            "handedness": HANDEDNESS,
            "arm_worn_on": ARM_WORN_ON,
            "device": DEVICE_NAME,
            "notes": self.notes_edit.text(),
            "plan_index": self.sequence.index,
            "plan_length": len(self.sequence.plan),
            "gesture_order": GESTURE_ORDER,
        })

    def _on_capture_discard(self, capture: Capture, label: str,
                            repetition: int) -> None:
        """An incomplete recording, abandoned because the link dropped."""
        self.acq.stop_capture()
        self._append_log(
            f"{label} rep {repetition}: discarded after {len(capture)} "
            "sample(s) -- link lost. The repetition will be retried."
        )

    def _on_capture_stop(self, capture: Capture, label: str,
                         repetition: int) -> None:
        self.acq.stop_capture()

        rec = capture.to_recording(self.person_id, self.session_id, {
            "myo_state_at_stop": self.acq.state,
        })
        if rec.n_samples < 2:
            self._append_log(
                f"{label} rep {repetition}: no samples captured -- nothing saved."
            )
            return

        expected = RECORD_TIME + (PRE_GESTURE_MS + POST_GESTURE_MS) / 1000.0
        rec.run_quality_checks(expected_seconds=expected,
                               had_disconnect=capture.had_disconnect)
        if AUTO_TRIM:
            rec.detect_activity()

        trimmed = None
        if AUTO_SAVE:
            try:
                self.writer.save(rec, STAGE_RAW)
                # With the review step on, the trimmed copy is written when the
                # operator accepts (using the band they may have adjusted), so
                # writing one here too would only create a duplicate.
                if AUTO_TRIM and AUTO_TRIM_WRITE_FILE and not REVIEW_EACH_RECORDING:
                    trimmed = rec.auto_trimmed()
                    if trimmed is not None:
                        self.writer.save(trimmed, STAGE_TRIMMED)
            except OSError as exc:
                self._append_log(f"SAVE FAILED for {label} rep {repetition}: {exc}")
                QtWidgets.QMessageBox.critical(
                    self, "Save failed",
                    f"Could not write the recording:\n\n{exc}\n\n"
                    "The samples are still in memory -- fix the problem and "
                    "use TRIM to re-save, or stop the session.",
                )

        self.recordings.append(rec)
        self._add_table_row(rec, trimmed is not None)

        level = rec.worst_issue_level
        self._append_log(
            f"{label} rep {repetition}: {rec.n_samples} samples, "
            f"{rec.duration:.2f} s"
            + (f", quality: {level}" if level else ", quality: ok")
        )
        self._refresh_plan_info()

        if REVIEW_EACH_RECORDING:
            # The review screen shows the quality findings inline, so a modal
            # on top of it would just be in the way.
            self._begin_review(rec)
        elif level == "error" and QC_PAUSE_ON in ("error", "warning"):
            self._handle_bad_recording(rec)
        elif level == "warning" and QC_PAUSE_ON == "warning":
            self._handle_bad_recording(rec)

    def _handle_bad_recording(self, rec: Recording) -> None:
        was_running = self.sequence.running and not self.sequence.paused
        if was_running:
            self.sequence.pause()

        dialog = QualityDialog(rec, self)
        dialog.exec()

        if dialog.repeat_requested:
            # Re-queue the same gesture at the current position.
            self.sequence.plan.insert(self.sequence.index, (rec.label, rec.repetition))
            self._append_log(f"Re-queued {rec.label} rep {rec.repetition}.")

        if was_running:
            self.sequence.resume()

    # ======================================================================
    # Recording list actions
    # ======================================================================

    def _add_table_row(self, rec: Recording, trimmed: bool) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)

        level = rec.worst_issue_level
        quality_text = {None: "ok", "info": "info",
                        "warning": "warn", "error": "ERROR"}[level]
        quality_color = {None: COL_GOOD, "info": COL_TEXT_DIM,
                         "warning": COL_WARN, "error": COL_BAD}[level]

        values = [
            str(row + 1), rec.label, str(rec.repetition),
            f"{rec.duration:.2f}s", quality_text, "yes" if trimmed else "-",
        ]
        for col, text in enumerate(values):
            item = QtWidgets.QTableWidgetItem(text)
            if col == 4:
                item.setForeground(QtGui.QColor(quality_color))
            if level:
                item.setToolTip("\n".join(str(i) for i in rec.issues))
            self.table.setItem(row, col, item)

        self.table.scrollToBottom()
        self.table.selectRow(row)

    def _selected_recording(self):
        row = self.table.currentRow()
        if 0 <= row < len(self.recordings):
            return row, self.recordings[row]
        return -1, None

    def trim_selected(self) -> None:
        row, rec = self._selected_recording()
        if rec is None:
            QtWidgets.QMessageBox.information(
                self, "Trim", "Select a recording in the list first."
            )
            return
        dialog = TrimDialog(rec, self.writer, self)
        accepted = dialog.exec()
        added = getattr(dialog, "notes_added", 0)
        if added:
            self._append_log(
                f"Added {added} note(s) to {rec.label} rep {rec.repetition}.")
        if accepted and dialog.result_recording is not None:
            item = self.table.item(row, 5)
            if item:
                item.setText("yes")
            self._append_log(
                f"Trimmed {rec.label} rep {rec.repetition} to "
                f"{dialog.result_recording.n_samples} samples."
            )

    def open_recording_file(self) -> None:
        """
        Load one or more recordings from disk and open each in turn.

        This is the route to a take from an earlier session: the RECORDINGS
        table only holds what this run produced, but any file under the
        dataset root can be pulled up, annotated and cut.

        Several files may be selected at once. They are handled one at a time
        and each stays its own recording -- see open_paths_for_review().
        """
        start_dir = self.writer.session_dir
        if not os.path.isdir(start_dir):
            start_dir = os.path.join(PROJECT_DIR, DATA_FOLDER)

        paths, _filter = QtWidgets.QFileDialog.getOpenFileNames(
            self, "Open recordings or takes (several may be selected)",
            start_dir,
            "Recordings (*.csv *.npz *.parquet);;All files (*)")
        if not paths:
            return
        self.open_paths_for_review(paths)

    def open_paths_for_review(self, paths) -> None:
        """
        Work through a list of files from disk, one dialog each.

        Each file is its own recording and is kept that way: selecting three
        2 s clips gives three training inputs of 2 s, never one of 6 s.
        Joining them would invent a gesture nobody performed and would hide a
        discontinuity from the windowing code, which assumes the samples of a
        recording are contiguous in time.
        """
        paths = [p for p in paths if p]
        takes = [p for p in paths if _stage_from_filename(p) == STAGE_TAKE]
        others = [p for p in paths if p not in takes]

        if len(paths) > 1:
            self._append_log(
                f"Opening {len(paths)} file(s) separately"
                + (f" ({len(takes)} take(s))" if takes else "") + ".")

        # Takes first: they produce new recordings, and doing them as a batch
        # keeps their progress messages together in the log.
        if takes:
            self.open_takes_for_cutting(takes)

        for index, path in enumerate(others, start=1):
            if len(others) > 1:
                self._append_log(
                    f"File {index} of {len(others)}: {os.path.basename(path)}")
            self._review_recording_file(path)

    def _review_recording_file(self, path: str) -> None:
        """One already-labelled recording: graph, notes, optional trim."""
        try:
            rec = load_recording(path)
        except Exception as exc:                 # noqa: BLE001 -- report anything
            QtWidgets.QMessageBox.critical(
                self, "Open recording",
                f"Could not read this file:\n\n{os.path.basename(path)}\n\n{exc}")
            return

        if rec.stage == STAGE_TAKE:              # stage came from the metadata
            self.open_take_for_cutting(path)
            return

        # Annotations and any trimmed copy have to land next to the file the
        # operator actually opened, which may belong to another person or
        # session than the one currently selected in the GUI.
        writer = DatasetWriter(
            self._dataset_root_for(path), rec.person_id, rec.session_id)
        rec.source = path
        if not rec.files:
            rec.files[f"{rec.stage}:{os.path.splitext(path)[1].lstrip('.')}"] = (
                os.path.relpath(path, writer.root))

        self._append_log(f"Opened {os.path.basename(path)} for review.")
        dialog = TrimDialog(rec, writer, self)
        dialog.exec()
        added = getattr(dialog, "notes_added", 0)
        if added:
            self._append_log(
                f"Added {added} note(s) to {os.path.basename(path)}.")
        if dialog.result_recording is not None:
            self._append_log(
                f"Cut {dialog.result_recording.n_samples} samples out of "
                f"{os.path.basename(path)} as training data.")

    @staticmethod
    def _dataset_root_for(path: str) -> str:
        """
        The dataset root a recording sits under:
        <root>/<person>/<session>/<GESTURE>/<file>  ->  <root>

        Falls back to the file's own folder if the layout is not recognised,
        which keeps sidecars next to the data rather than somewhere surprising.
        """
        folder = os.path.dirname(os.path.abspath(path))
        root = os.path.dirname(os.path.dirname(os.path.dirname(folder)))
        return root if os.path.isdir(root) else folder

    def auto_trim_all(self) -> None:
        if not self.recordings:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Auto-trim all",
            f"Run activity detection over {len(self.recordings)} recording(s) "
            "and write a trimmed copy of each?\n\n"
            "Raw files are not modified.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.Yes,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        written = skipped = 0
        for row, rec in enumerate(self.recordings):
            trimmed = rec.auto_trimmed()
            if trimmed is None:
                skipped += 1
                continue
            try:
                self.writer.save(trimmed, STAGE_TRIMMED)
            except OSError as exc:
                self._append_log(f"auto-trim failed for row {row + 1}: {exc}")
                skipped += 1
                continue
            written += 1
            item = self.table.item(row, 5)
            if item:
                item.setText("yes")

        self._append_log(f"Auto-trim: {written} written, {skipped} skipped.")
        QtWidgets.QMessageBox.information(
            self, "Auto-trim complete",
            f"{written} trimmed file(s) written.\n"
            f"{skipped} recording(s) skipped (no clear activity, REST, or an "
            "error).\n\nAll raw files are unchanged.",
        )

    def delete_last(self) -> None:
        if not self.recordings:
            return
        rec = self.recordings[-1]
        answer = QtWidgets.QMessageBox.question(
            self, "Delete last recording",
            f"Move {rec.label} rep {rec.repetition} out of the dataset?\n\n"
            "The files are moved into the _deleted folder, not erased, so the "
            "raw data can always be recovered.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        moved = self._quarantine(rec)
        self.recordings.pop()
        self.table.removeRow(self.table.rowCount() - 1)
        self._append_log(
            f"Moved {rec.label} rep {rec.repetition} to _deleted "
            f"({moved} file(s))."
        )
        self._refresh_plan_info()

    def _quarantine(self, rec: Recording) -> int:
        """
        Move a recording's files into <session>/_deleted/.

        Deliberately not os.remove: a mis-click during a long session should
        cost a folder move, not an hour of a subject's time.
        """
        target = ensure_dir(os.path.join(self.writer.session_dir,
                                         DELETED_DIRNAME))
        moved = 0
        for key, relative in list(rec.files.items()):
            source = os.path.join(self.writer.root, relative)
            if not os.path.exists(source):
                continue
            destination = unique_path(os.path.join(target, os.path.basename(source)))
            try:
                os.replace(source, destination)
                moved += 1
                rec.files[key] = os.path.relpath(destination, self.writer.root)
            except OSError as exc:
                self._append_log(f"could not move {source}: {exc}")

        if moved:
            entry = {
                "uid": rec.uid, "stage": "deleted", "label": rec.label,
                "repetition": rec.repetition, "person_id": rec.person_id,
                "session_id": rec.session_id,
                "deleted_iso": iso_time(time.time()),
                "files": dict(rec.files),
            }
            with open(self.writer.manifest_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
        return moved

    def clear_list(self) -> None:
        if not self.recordings:
            return
        answer = QtWidgets.QMessageBox.question(
            self, "Clear list",
            f"Remove {len(self.recordings)} recording(s) from this list and "
            "reset the live graph?\n\nFiles already written to disk are kept.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.recordings.clear()
        self.table.setRowCount(0)
        self.acq.ring.clear()
        self._append_log("List cleared (files on disk kept).")

    # ======================================================================
    # Graph / dataset export
    # ======================================================================

    def save_graph(self) -> None:
        item = self.sequence.current
        gesture = item[0] if item else self.gesture_combo.currentText()
        default = os.path.join(
            self.writer.graphs_dir,
            f"{self.person_id}_{self.session_id}_{safe_slug(gesture)}_"
            f"{timestamp_slug()}.png",
        )
        path, selected = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save graph", default,
            "PNG image (*.png);;SVG vector (*.svg)",
        )
        if not path:
            return
        if "svg" in selected.lower() and not path.lower().endswith(".svg"):
            path += ".svg"
        elif not os.path.splitext(path)[1]:
            path += ".png"

        self._update_plot_title()
        try:
            written = self.plot.export_image(path)
        except Exception as exc:
            traceback.print_exc()
            QtWidgets.QMessageBox.critical(self, "Save graph",
                                           f"Export failed:\n\n{exc}")
            return
        self._append_log(f"Graph saved: {written}")
        self.status.showMessage(f"Graph saved to {written}", 8000)

    def export_dataset(self) -> None:
        if self.export_worker is not None:
            return
        out_dir = os.path.join(PROJECT_DIR, "exports")
        self.export_worker = DatasetExportWorker(
            os.path.join(PROJECT_DIR, DATA_FOLDER), out_dir, "prefer_trimmed", self
        )
        self.export_worker.progress.connect(
            lambda text: self.status.showMessage(text, 0)
        )
        self.export_worker.done.connect(self._on_export_done)
        self.export_worker.failed.connect(self._on_export_failed)
        self.export_worker.finished.connect(self._on_export_finished)
        self.export_button.setEnabled(False)
        self._append_log("Dataset export started...")
        self.export_worker.start()

    def _on_export_done(self, message: str) -> None:
        self._append_log("Export complete.")
        QtWidgets.QMessageBox.information(self, "Export complete", message)

    def _on_export_failed(self, message: str) -> None:
        self._append_log(f"Export failed: {message}")
        QtWidgets.QMessageBox.warning(self, "Export failed", message)

    def _on_export_finished(self) -> None:
        self.export_worker = None
        self.status.clearMessage()

    # ======================================================================
    # Live prediction
    # ======================================================================

    # -- panel state ----------------------------------------------------------

    def _refresh_prediction_panel(self) -> None:
        ready = self.predictor.ready
        self.predict_button.setEnabled(ready)
        self.pred_mode_combo.setEnabled(ready)
        # The model just changed state, so the provenance line is stale.
        self._refresh_model_status(force=True)

        if ready:
            self.prediction_note.setText(self.predictor.describe())
            return

        self.prediction_label.setText("--")
        self.prediction_label.setStyleSheet(
            f"color: {COL_TEXT_DIM}; background: transparent;")
        self.prediction_conf.setText("")
        reasons = {
            "disabled in the configuration":
                "Disabled. Set ENABLE_LIVE_PREDICTION = True in the config.",
            "no model file":
                f"No model at {self.predictor.path}.\n"
                "Collect data, then run:  python train_model.py",
            "joblib is not installed":
                "joblib is missing:  python -m pip install joblib",
        }
        self.prediction_note.setText(
            reasons.get(self.predictor.status,
                        f"Unavailable: {self.predictor.status}"))

    def _on_pred_mode_changed(self) -> None:
        self._set_global("PREDICTION_MODE", self.prediction_mode)
        self.stop_prediction()

    @property
    def prediction_mode(self) -> str:
        return self.pred_mode_combo.currentData() or "one_shot"

    # -- control --------------------------------------------------------------

    def toggle_prediction(self) -> None:
        if not self.predictor.ready:
            return
        if self.prediction_active or self.one_shot_deadline:
            self.stop_prediction()
        elif self.prediction_mode == "continuous":
            self.start_continuous_prediction()
        else:
            self.start_one_shot_prediction()

    def start_continuous_prediction(self) -> None:
        self.predictor.reset()
        self.prediction_active = True
        self.one_shot_deadline = 0.0
        self.predict_button.setText("STOP")
        self.predict_button.setObjectName("danger")
        self.predict_button.setStyleSheet("")
        self.pred_timer.start(max(30, int(PREDICTION_INTERVAL_MS)))
        self._append_log("Continuous prediction started.")

    def start_one_shot_prediction(self) -> None:
        """
        One prediction, then stop.

        The window is taken AFTER a short delay so you have time to get into
        the gesture -- classifying the instant the button is pressed would
        just classify whatever you were doing while reaching for the mouse.
        """
        self.predictor.reset()
        self.prediction_active = False
        now = time.perf_counter()
        # Two phases: a grace period to get into the gesture, then the window
        # that is actually classified. The gesture must be held through BOTH.
        self.one_shot_capture_at = now + ONE_SHOT_DELAY_S
        self.one_shot_deadline = (self.one_shot_capture_at
                                  + self.predictor.window_seconds())
        self.predict_button.setText("CANCEL")
        self.prediction_label.setText("DO THE GESTURE")
        self.prediction_label.setStyleSheet(
            f"color: {COL_WARN}; background: transparent;")
        self.prediction_conf.setText("and hold it...")
        self.plot.set_prediction_overlay(
            "DO THE GESTURE NOW", COL_WARN, "hold it until the result appears")
        self.pred_timer.start(30)
        self._append_log(
            f"One-shot armed: hold the gesture for "
            f"{ONE_SHOT_DELAY_S + self.predictor.window_seconds():.1f} s.")

    def stop_prediction(self) -> None:
        was = self.prediction_active or bool(self.one_shot_deadline)
        self.prediction_active = False
        self.one_shot_deadline = 0.0
        self.one_shot_capture_at = 0.0
        self.pred_timer.stop()
        self.predict_button.setText("PREDICT")
        self.predict_button.setObjectName("primary")
        self.predict_button.setStyleSheet("")
        self.plot.set_prediction_overlay("")
        if was:
            self._append_log("Prediction stopped.")

    # -- the prediction tick --------------------------------------------------

    def _tick_prediction(self) -> None:
        if not self.predictor.ready:
            self.stop_prediction()
            self._refresh_prediction_panel()
            return

        # One-shot: wait out the grace period and the capture window, then
        # fire exactly once. The two phases are shown separately so it is
        # never ambiguous whether the gesture should be happening yet.
        if self.one_shot_deadline:
            now = time.perf_counter()
            remaining = self.one_shot_deadline - now
            if remaining > 0:
                if now < self.one_shot_capture_at:
                    self.prediction_label.setText("DO THE GESTURE")
                    self.prediction_conf.setText(
                        f"hold it -- recording the signal in "
                        f"{self.one_shot_capture_at - now:.1f} s")
                else:
                    self.prediction_label.setText("HOLD...")
                    self.prediction_conf.setText(
                        f"reading the signal now ({remaining:.1f} s left)")
                    self.plot.set_prediction_overlay(
                        "READING THE SIGNAL", COL_BAD, "keep holding")
                return
            self.one_shot_deadline = 0.0
            self.one_shot_capture_at = 0.0
            prediction = self._run_prediction()
            self.pred_timer.stop()
            self.predict_button.setText("PREDICT")
            if prediction is None:
                self.prediction_label.setText("NO DATA")
                self.prediction_label.setStyleSheet(
                    f"color: {COL_BAD}; background: transparent;")
                self.prediction_conf.setText("not enough samples")
                self.plot.set_prediction_overlay("")
            else:
                self.one_shot_clear_at = time.perf_counter() + ONE_SHOT_HOLD_S
            return

        if not self.prediction_active:
            return
        self._run_prediction()

    def _run_prediction(self):
        """Pull the newest window out of the ring and classify it."""
        rate = self.acq.stats()["rate"] or nominal_rate()
        want = int(self.predictor.window_seconds() * rate * 1.25) + 4
        emg, t = self.acq.ring.snapshot(max(8, want))
        if emg.shape[0] < 4:
            return None

        # Keep only the samples inside the model's own window length.
        if t.size and (t[-1] - t[0]) > self.predictor.window_seconds():
            keep = t >= (t[-1] - self.predictor.window_seconds())
            emg, t = emg[keep], t[keep]
        if emg.shape[0] < 4:
            return None

        prediction = self.predictor.predict(emg, t)
        if prediction is None:
            self._refresh_prediction_panel()
            self.stop_prediction()
            return None
        self._show_prediction(prediction)
        return prediction

    def _show_prediction(self, prediction) -> None:
        certain = prediction.certain
        colour = COL_GOOD if certain else COL_WARN
        self.prediction_label.setText(prediction.label if certain
                                      else f"{prediction.label}?")
        self.prediction_label.setStyleSheet(
            f"color: {colour}; background: transparent;")
        self.prediction_conf.setText(
            f"{prediction.percent}"
            + ("" if certain else "   (below the confidence threshold)")
            + (f"   ·  {prediction.smoothed_over} window vote"
               if prediction.smoothed_over > 1 else ""))
        self.prediction_conf.setStyleSheet(
            f"color: {colour}; background: transparent;")

        for name, (bar, value) in self.prob_bars.items():
            if name in prediction.labels:
                p = prediction.probabilities[prediction.labels.index(name)]
            else:
                p = float("nan")
            if math.isnan(p):
                bar.setValue(0)
                value.setText("--")
                continue
            bar.setValue(int(np.clip(p, 0, 1) * 1000))
            value.setText(f"{p * 100:.0f} %")
            top = name == prediction.label
            bar.setStyleSheet(
                f"QProgressBar {{ background: #0b0e13; border: 1px solid "
                f"{COL_BORDER}; border-radius: 3px; }}"
                f"QProgressBar::chunk {{ background: "
                f"{colour if top else COL_TEXT_DIM}; border-radius: 2px; }}")

        self.plot.set_prediction_overlay(
            f"{prediction.label}  {prediction.percent}", colour,
            "" if certain else "low confidence")

    def show_model_stats(self) -> None:
        """
        Open MODEL STATS, without blocking the window behind it.

        Deliberately NOT modal. The DECISION TRAIL view follows the live
        classifier, and a modal dialog would mean you could watch it only
        when PREDICT happened to be running already -- you could not reach
        the button to start it. Reusing one instance keeps a second copy from
        polling the panel on its own timer.
        """
        if self.stats_dialog is not None and self.stats_dialog.isVisible():
            self.stats_dialog.raise_()
            self.stats_dialog.activateWindow()
            return
        self.decision_panel.load()
        dialog = ModelStatsDialog(self.predictor, self,
                                  panel=self.decision_panel)
        dialog.setAttribute(QtCore.Qt.WA_DeleteOnClose, True)
        dialog.finished.connect(self._on_stats_closed)
        self.stats_dialog = dialog
        dialog.show()

    def _on_stats_closed(self, _result=None) -> None:
        self.stats_dialog = None
        self._rebuild_prob_bars()
        self._refresh_prediction_panel()

    # ======================================================================
    # LABEL ON GRAPH
    # ======================================================================

    def toggle_label_mode(self, enabled: bool) -> None:
        if enabled and self.sequence.running:
            QtWidgets.QMessageBox.information(
                self, "Label on graph",
                "Stop the cued session first -- the two recording styles "
                "would fight over the same labels.")
            self.label_mode_button.setChecked(False)
            return

        if enabled:
            self.expand_section("label")
        self.plot.set_label_mode(enabled)
        self.label_mode_button.setText(
            "STOP LABEL MODE" if enabled else "START LABEL MODE")
        if enabled:
            self._append_log(
                "Label mode on: do a gesture, then SELECT and ADD it.")
        else:
            self.plot.clear_selection()
            self.selection_info.setText("")
        self._refresh_label_counts()

    def arm_selection(self) -> None:
        if not self.label_mode_button.isChecked():
            self.label_mode_button.setChecked(True)
        _, t_now = self.acq.ring.latest()
        if not t_now:
            self.selection_info.setText("no signal yet")
            return
        self.plot.arm_selection(t_now, LABEL_DEFAULT_SELECTION_S)
        self._on_selection_changed(*self.plot.selection_bounds())

    def _on_selection_changed(self, start_abs: float, end_abs: float) -> None:
        emg, _ = self._slice_ring(start_abs, end_abs)
        duration = max(0.0, end_abs - start_abs)
        self.selection_info.setText(
            f"selection {duration:.2f} s, {emg.shape[0]} samples"
            + ("   -- too short" if duration < LABEL_MIN_SELECTION_S else ""))
        self.selection_info.setStyleSheet(
            f"color: {COL_BAD if duration < LABEL_MIN_SELECTION_S else COL_TEXT_DIM};"
            " background: transparent;")

    def _slice_ring(self, start_abs: float, end_abs: float):
        """Pull the samples between two absolute times out of the ring."""
        emg, t = self.acq.ring.snapshot()
        if t.size == 0:
            return np.zeros((0, N_CHANNELS), np.int32), np.zeros(0, np.float64)
        mask = (t >= start_abs) & (t <= end_abs)
        return emg[mask], t[mask]

    def _write_span_as_recording(self, start_abs: float, end_abs: float,
                                 label: str, capture_style: str,
                                 extra_meta: dict = None, dialog_title: str = "Add"):
        """
        Cut a span out of the ring buffer and write it as a labelled
        recording. Shared by ADD SELECTION and CUT SINCE LAST MARKER so the
        two cannot drift apart -- a recording written by one route has to be
        indistinguishable from one written by the other.

        Returns the Recording on success, None if it was rejected or failed.
        """
        emg, t = self._slice_ring(start_abs, end_abs)

        if (end_abs - start_abs) < LABEL_MIN_SELECTION_S or emg.shape[0] < 4:
            QtWidgets.QMessageBox.information(
                self, dialog_title,
                f"The span is too short ({end_abs - start_abs:.2f} s, "
                f"{emg.shape[0]} samples).\n\nMake it wider, or lower "
                "LABEL_MIN_SELECTION_S.")
            return None

        meta = {
            "emg_mode": EMG_MODE,
            "handedness": HANDEDNESS,
            "arm_worn_on": ARM_WORN_ON,
            "device": DEVICE_NAME,
            "notes": self.notes_edit.text(),
            "capture_style": capture_style,
            "selected_by_operator": True,
            "selection_seconds": round(end_abs - start_abs, 4),
        }
        meta.update(extra_meta or {})

        # Any marker dropped inside this span is part of the story of the
        # recording, so it travels with it in the metadata.
        inside = markers_between(self.session_markers, start_abs, end_abs)
        if inside:
            meta["markers"] = [m.to_dict() for m in inside]

        repetition = self.writer.next_repetition(label)
        rec = Recording(
            emg=emg, timestamps=t, label=label,
            person_id=self.person_id, session_id=self.session_id,
            repetition=repetition, stage=STAGE_RAW,
            # The whole span IS the gesture: the operator drew the
            # boundaries, so there is no pre/post roll to exclude.
            label_start_index=0, label_end_index=emg.shape[0],
            meta=meta,
        )
        rec.run_quality_checks(expected_seconds=end_abs - start_abs)
        if AUTO_TRIM:
            rec.detect_activity()

        try:
            self.writer.save(rec, STAGE_RAW)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, dialog_title, f"Could not write the recording:\n{exc}")
            return None

        self.recordings.append(rec)
        self.expand_section("recordings")
        self._add_table_row(rec, False)
        self.plot.add_mark(start_abs, end_abs, label, rec.uid)
        self._append_log(
            f"Labelled {label} rep {repetition}: {rec.n_samples} samples, "
            f"{rec.duration:.2f} s"
            + (f", {len(inside)} marker(s) inside" if inside else "")
            + (f" (quality: {rec.worst_issue_level})"
               if rec.worst_issue_level else ""))
        self.acq.vibrate(1)
        self._refresh_label_counts()
        self._refresh_gesture_counts()
        self._refresh_plan_info()
        return rec

    def add_selection_as_label(self) -> None:
        """Turn the boxed region into a labelled recording on disk."""
        bounds = self.plot.selection_bounds()
        if bounds is None:
            self.selection_info.setText("nothing selected -- press SELECT first")
            return

        start_abs, end_abs = bounds
        label = self.label_gesture_combo.currentText()
        rec = self._write_span_as_recording(
            start_abs, end_abs, label, "label_on_graph",
            dialog_title="Add selection")
        if rec is None:
            return

        self.plot.clear_selection()
        self.selection_info.setText(
            f"added {label}: {rec.n_samples} samples, {rec.duration:.2f} s")
        self.selection_info.setStyleSheet(
            f"color: {COL_GOOD}; background: transparent;")

    # ======================================================================
    # Free take  (record the whole stretch now, cut clips out of it later)
    # ======================================================================

    def _on_cut_length_changed(self, value: float) -> None:
        """
        The default cut length, changed in the GUI.

        Mirrored into the module global so it reaches *_meta.json via
        recording_config_snapshot(), and the button is relabelled so what it
        will do is readable without hovering it.
        """
        self._set_global("LABEL_DEFAULT_SELECTION_S", float(value))
        self.select_button.setText(f"SELECT LAST {value:.2f} s")

    def toggle_free_take(self, checked: bool) -> None:
        if checked:
            self.start_free_take()
        else:
            self.stop_free_take()

    def start_free_take(self) -> None:
        """Open a long capture. Nothing is written until STOP & SAVE."""
        if self.sequence.running:
            # The acquisition thread has one capture slot, so a take and the
            # cued sequence cannot both own it.
            self._set_free_take_status(
                "cannot start: a recording session is running", COL_BAD)
            self.free_take_button.setChecked(False)
            return

        label = self.take_gesture_combo.currentText()
        self.free_take_label = label
        self.free_take_number = self.writer.next_take_number()
        self.free_capture = self.acq.start_capture(
            label, 0,
            meta={
                "emg_mode": EMG_MODE,
                "handedness": HANDEDNESS,
                "arm_worn_on": ARM_WORN_ON,
                "device": DEVICE_NAME,
                "notes": self.notes_edit.text(),
                "capture_style": "free_take",
                "intended_gesture": label,
                "take_number": self.free_take_number,
                "free_take_max_minutes": FREE_TAKE_MAX_MINUTES,
            },
            max_seconds=FREE_TAKE_MAX_MINUTES * 60.0,
        )
        self.free_capture.mark_label_start()

        self.expand_section("free_take")
        self.free_take_button.setText("STOP & SAVE TAKE")
        self._set_free_take_status(
            f"RECORDING take {self.free_take_number:03d} -- {label}", COL_BAD)
        self._append_log(
            f"Free take {self.free_take_number:03d} started "
            f"({label}, up to {FREE_TAKE_MAX_MINUTES:.0f} min).")
        self.acq.vibrate(1)
        self._refresh_controls()

    def stop_free_take(self) -> None:
        """Close the capture and write the whole stretch into _takes/."""
        capture = self.free_capture
        self.free_capture = None
        self.free_take_button.setText("START FREE TAKE")

        if capture is None:
            self._set_free_take_status("idle -- nothing being recorded",
                                       COL_TEXT_DIM)
            return

        self.acq.stop_capture()
        self.acq.vibrate(2)

        rec = capture.to_recording(self.person_id, self.session_id)
        if rec.n_samples < 2 or rec.duration < FREE_TAKE_MIN_SECONDS:
            # Not worth a file. Nothing was written, so nothing to clean up.
            self._set_free_take_status(
                f"discarded: only {rec.duration:.2f} s "
                f"({rec.n_samples} samples), the minimum is "
                f"{FREE_TAKE_MIN_SECONDS:.0f} s",
                COL_WARN)
            self._append_log(
                f"Free take discarded ({rec.duration:.2f} s, too short).")
            self._refresh_controls()
            return

        # A take is not a gesture instance, so it gets no quality verdict on
        # whether it "contains activity": it is supposed to contain long
        # stretches of nothing. Only the honest signal checks run.
        rec.run_quality_checks(had_disconnect=capture.had_disconnect)

        # Markers dropped during the take travel with it, which is what makes
        # them useful later: they are the landmarks you cut against.
        inside = markers_between(self.session_markers,
                                 float(rec.timestamps[0]),
                                 float(rec.timestamps[-1]))
        if inside:
            rec.meta["markers"] = [m.to_dict() for m in inside]
        rec.meta["capture_overflowed"] = capture.overflowed

        try:
            written = self.writer.save_take(rec, self.free_take_number)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, "Free take", f"Could not write the take:\n{exc}")
            self._set_free_take_status(f"SAVE FAILED: {exc}", COL_BAD)
            return

        name = os.path.basename(written.get("csv") or written.get("npz") or "")
        self._set_free_take_status(
            f"saved take {self.free_take_number:03d}: "
            f"{fmt_duration(rec.duration)}, {rec.n_samples} samples", COL_GOOD)
        self.free_take_detail.setText(
            f"{name}\nOpen it below to cut clips out of it."
            + (f"   {len(inside)} marker(s) inside." if inside else "")
            + ("   NOTE: the length cap was reached, so the end may be "
               "missing." if capture.overflowed else ""))
        self._append_log(
            f"Free take {self.free_take_number:03d} saved: "
            f"{rec.n_samples} samples, {fmt_duration(rec.duration)}"
            + (f", {len(inside)} marker(s)" if inside else "")
            + (f" (quality: {rec.worst_issue_level})"
               if rec.worst_issue_level else ""))

        self._refresh_take_history()
        self._refresh_controls()

    def _tick_free_take(self) -> None:
        """
        Keep the operator informed, and stop at the cap.

        Called from the 100 ms UI timer, which is also what bounds how long a
        take can overrun its cap: the acquisition Capture stops accepting
        samples by itself, so overrunning costs nothing but a stale readout.
        """
        capture = self.free_capture
        if capture is None:
            return

        elapsed = time.time() - capture.started_unix
        cap = max(1.0, FREE_TAKE_MAX_MINUTES * 60.0)
        samples = len(capture)
        fraction = elapsed / cap

        self._set_free_take_status(
            f"RECORDING take {self.free_take_number:03d} -- "
            f"{self.free_take_label}   {fmt_duration(elapsed)}",
            COL_BAD)
        self.free_take_detail.setText(
            f"{samples:,} samples   {fraction * 100:.0f} % of the "
            f"{FREE_TAKE_MAX_MINUTES:.0f} min limit   --   "
            "drop BREAK markers as you go, they become the cut points"
        )
        self.free_take_detail.setStyleSheet(
            f"color: {COL_WARN if fraction > FREE_TAKE_WARN_FRACTION else COL_TEXT_DIM};"
            " background: transparent;")

        if elapsed >= cap or capture.overflowed:
            self._append_log(
                f"Free take {self.free_take_number:03d} hit the "
                f"{FREE_TAKE_MAX_MINUTES:.0f} min cap -- saving it.")
            # Drives stop_free_take() through the toggled signal, so the
            # button and the state can never disagree.
            self.free_take_button.setChecked(False)

    def _set_free_take_status(self, text: str, color: str) -> None:
        self.free_take_status.setText(text)
        self.free_take_status.setStyleSheet(
            f"color: {color}; background: transparent;")

    def _refresh_take_history(self) -> None:
        """Re-read _takes/ for the current subject and session."""
        if not hasattr(self, "take_list"):
            return
        self.take_list.clear()
        try:
            takes = self.writer.list_takes()
        except OSError as exc:                               # pragma: no cover
            self._append_log(f"could not read takes: {exc}")
            return

        if not takes:
            item = QtWidgets.QListWidgetItem(
                "no takes yet for this subject/session")
            item.setForeground(QtGui.QColor(COL_TEXT_DIM))
            self.take_list.addItem(item)
            return

        for info in takes:
            stamp = datetime.fromtimestamp(info["created_unix"]).strftime(
                "%Y-%m-%d %H:%M")
            item = QtWidgets.QListWidgetItem(
                f"{info['label']}   {stamp}   "
                f"{fmt_duration(info['duration_s'])}   "
                f"{info['n_samples']:,} samples")
            item.setData(QtCore.Qt.UserRole, info["path"])
            # The stem, not the path: DELETE TAKE(S) moves every format the
            # take was written in, and the list only carries one of them.
            item.setData(QtCore.Qt.UserRole + 1, info["stem"])
            item.setToolTip(info["path"])
            self.take_list.addItem(item)
        self.take_list.setCurrentRow(0)

    def open_selected_take(self) -> None:
        """Open every selected take in turn and cut clips out of each."""
        if not hasattr(self, "take_list"):
            return
        items = self.take_list.selectedItems()
        if not items:
            item = self.take_list.currentItem()
            items = [item] if item is not None else []

        paths = [i.data(QtCore.Qt.UserRole) for i in items]
        paths = [p for p in paths if p]
        if not paths:
            self._set_free_take_status(
                "select a take in the list first", COL_WARN)
            return
        self.open_takes_for_cutting(paths)

    def delete_selected_takes(self) -> None:
        """
        Move the selected take(s) into _deleted/ after confirming.

        The confirmation names how many clips were cut from each take,
        because that is the thing someone would want to know before throwing
        a take away: the clips themselves are untouched (they are ordinary
        recordings with their own uids), but their `parent_take_file` will
        point into _deleted/ afterwards and the span cannot be re-cut.
        """
        if not hasattr(self, "take_list"):
            return
        if self.free_capture is not None:
            self._set_free_take_status(
                "stop the running take before deleting takes", COL_WARN)
            return

        items = self.take_list.selectedItems()
        if not items:
            item = self.take_list.currentItem()
            items = [item] if item is not None else []
        stems = [i.data(QtCore.Qt.UserRole + 1) for i in items]
        stems = [s for s in stems if s]
        if not stems:
            self._set_free_take_status(
                "select a take in the list first", COL_WARN)
            return

        lines, total_clips = [], 0
        for stem in stems:
            uid = take_uid_from_stem(stem)
            n_clips = len(self.writer.clips_from_take(uid)) if uid else 0
            total_clips += n_clips
            lines.append(f"    {stem}"
                         + (f"   ({n_clips} clip(s) cut from it)"
                            if n_clips else ""))

        detail = "\n".join(lines)
        note = ""
        if total_clips:
            note = (f"\n\n{total_clips} clip(s) were cut from these takes. "
                    "Those clips are separate recordings and are NOT "
                    "deleted -- they stay in the dataset. You will not be "
                    "able to cut any more clips from these takes, though.")

        answer = QtWidgets.QMessageBox.question(
            self, "Delete take(s)",
            f"Move {len(stems)} take(s) out of _takes?\n\n{detail}\n\n"
            "The files move into the _deleted folder, not erased, so the "
            "signal can always be recovered." + note,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        deleted, moved = 0, 0
        for stem in stems:
            result = self.writer.delete_take(stem)
            if result["moved"]:
                deleted += 1
                moved += len(result["moved"])
                self._append_log(
                    f"Moved take {result['label']} {stem} to "
                    f"{DELETED_DIRNAME} ({len(result['moved'])} file(s)).")
            else:
                self._append_log(f"could not delete take {stem}: no files "
                                 "matched -- was it already moved?")

        self._refresh_take_history()
        if deleted:
            self._set_free_take_status(
                f"{deleted} take(s) moved to {DELETED_DIRNAME} "
                f"({moved} file(s))", COL_TEXT_DIM)
        else:
            self._set_free_take_status(
                "nothing was moved -- see the log", COL_WARN)

    def open_takes_for_cutting(self, paths) -> None:
        """
        Work through several takes, one dialog each.

        Each take is opened separately and every clip cut from it is written
        as its own recording. Nothing is ever joined up: three 2 s clips are
        three training inputs of 2 s, never one of 6 s. Concatenating them
        would be wrong twice over -- it would invent a gesture nobody
        performed, and it would hide from the windowing code that the samples
        either side of a join are seconds or minutes apart.
        """
        paths = [p for p in paths if p]
        if not paths:
            return

        total = 0
        for index, path in enumerate(paths, start=1):
            if len(paths) > 1:
                self._append_log(
                    f"Take {index} of {len(paths)}: {os.path.basename(path)}")
            total += len(self.open_take_for_cutting(path) or [])

        if len(paths) > 1:
            self._set_free_take_status(
                f"extracted {total} clip(s) from {len(paths)} takes, "
                f"each a separate recording", COL_GOOD)
            self._append_log(
                f"Finished {len(paths)} takes: {total} separate "
                f"recording(s) written.")

    def open_take_for_cutting(self, path: str):
        """
        Load a take from disk and open the cut dialog on it.

        Shared by the history list and by OPEN FILE..., so a take opened
        either way behaves identically. Returns the clips that were written,
        so a caller working through several takes can total them up.
        """
        try:
            rec = load_recording(path)
        except Exception as exc:                 # noqa: BLE001 -- report anything
            QtWidgets.QMessageBox.critical(
                self, "Open take", f"Could not read this take:\n\n{exc}")
            return []

        # Clips land next to the take's own person/session, which may not be
        # the subject currently selected in the GUI.
        writer = DatasetWriter(
            self._dataset_root_for_take(path), rec.person_id, rec.session_id)
        rec.source = path
        if not rec.files:
            rec.files[f"{STAGE_TAKE}:{os.path.splitext(path)[1].lstrip('.')}"] = (
                os.path.relpath(path, writer.root))

        self._append_log(
            f"Opened take {os.path.basename(path)} "
            f"({rec.n_samples} samples, {fmt_duration(rec.duration)}).")
        dialog = TrimDialog(rec, writer, self, extract_mode=True)
        dialog.exec()

        clips = list(getattr(dialog, "extracted", []))
        for clip in clips:
            self.recordings.append(clip)
            self._add_table_row(clip, False)
        if clips:
            self._append_log(
                f"Extracted {len(clips)} clip(s) from "
                f"{os.path.basename(path)}: "
                + ", ".join(f"{c.label} rep {c.repetition}" for c in clips))
            self._set_free_take_status(
                f"extracted {len(clips)} clip(s) from this take", COL_GOOD)
        else:
            self._set_free_take_status("no clips extracted", COL_TEXT_DIM)

        added = getattr(dialog, "notes_added", 0)
        if added:
            self._append_log(
                f"Added {added} note(s) to {os.path.basename(path)}.")
        self._refresh_label_counts()
        self._refresh_gesture_counts()
        self._refresh_plan_info()
        self._refresh_controls()
        return clips

    @staticmethod
    def _dataset_root_for_take(path: str) -> str:
        """
        The dataset root a TAKE sits under:
        <root>/<person>/<session>/_takes/<file>  ->  <root>

        Same depth as a gesture folder, but spelled out separately because a
        take's parent folder is _takes rather than a gesture name, and getting
        this wrong would scatter clips outside the dataset.
        """
        folder = os.path.dirname(os.path.abspath(path))
        root = os.path.dirname(os.path.dirname(os.path.dirname(folder)))
        return root if os.path.isdir(root) else folder

    # ======================================================================
    # Markers / notes
    # ======================================================================

    def _marker_now(self) -> float:
        """
        The timestamp a marker should carry: the newest sample's time, so the
        marker lines up with the signal rather than with the GUI clock. Falls
        back to wall time when nothing is streaming yet.
        """
        _, t_now = self.acq.ring.latest()
        return float(t_now) if t_now else time.time()

    def _sync_marker_controls(self) -> None:
        if not hasattr(self, "cut_marker_button"):
            return
        allowed = bool(self.marker_cut_check.isChecked())
        have = bool(self.session_markers)
        self.cut_marker_button.setEnabled(allowed and have)

    def _on_marker_cut_toggled(self, checked: bool) -> None:
        self._set_global("MARKER_CUT_ENABLED", bool(checked))
        self._sync_marker_controls()
        self._append_log(
            f"Cutting at markers {'enabled' if checked else 'disabled'}.")

    def add_note_marker(self) -> None:
        self._add_marker("note")

    def add_break_marker(self) -> None:
        self._add_marker("break")

    def _add_marker(self, kind: str) -> None:
        """Log a marker and draw it. Nothing already on disk is touched."""
        text = self.marker_text.text().strip()
        if kind == "note" and not text:
            self.marker_status.setText("type the note first, then ADD NOTE")
            self.marker_status.setStyleSheet(
                f"color: {COL_BAD}; background: transparent;")
            self.marker_text.setFocus()
            return

        marker = Marker(
            t_unix=self._marker_now(), text=text, kind=kind,
            person_id=self.person_id, session_id=self.session_id,
        )
        try:
            self.writer.append_marker(marker)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, "Add marker", f"Could not write the marker:\n{exc}")
            return

        self.session_markers.append(marker)
        self.plot.add_marker(marker.t_unix, marker.text, marker.kind, marker.uid)
        self.marker_text.clear()

        shown = marker.short_text(40)
        self.marker_status.setText(
            f"{'break' if marker.is_break else 'note'} at "
            f"{datetime.fromtimestamp(marker.t_unix).strftime('%H:%M:%S')}: {shown}")
        self.marker_status.setStyleSheet(
            f"color: {COL_GOOD}; background: transparent;")
        self._refresh_marker_list()
        self._redraw_markers_now()
        self._append_log(
            f"Marker ({marker.kind}) at "
            f"{datetime.fromtimestamp(marker.t_unix).strftime('%H:%M:%S')}: "
            f"{marker.text or '(no text)'}")
        self.acq.vibrate(1)
        self._sync_marker_controls()

    def _redraw_markers_now(self) -> None:
        """
        Draw the markers straight away instead of waiting for the next frame.

        update_data() is what normally redraws them, and it returns early when
        the graph is frozen, in review mode, or -- the case that actually bit
        -- when no samples are arriving because the armband is not connected.
        Dropping a note and seeing nothing happen reads as the note being
        lost, so this paints it on the spot.
        """
        try:
            self.plot._draw_markers(self.plot._t_now or time.time())
        except Exception:                                    # pragma: no cover
            pass            # drawing a caption must never break recording

    def _refresh_marker_list(self) -> None:
        """Newest last, capped -- the file keeps all of them regardless."""
        self.marker_list.clear()
        for marker in self.session_markers[-40:]:
            stamp = datetime.fromtimestamp(marker.t_unix).strftime("%H:%M:%S")
            prefix = "|" if marker.is_break else "-"
            item = QtWidgets.QListWidgetItem(
                f"{prefix} {stamp}  {marker.short_text(44)}")
            item.setForeground(QtGui.QColor(
                MARKER_COLORS.get(marker.kind, MARKER_COLORS["note"])))
            item.setToolTip(marker.text or "(no text)")
            self.marker_list.addItem(item)
        self.marker_list.scrollToBottom()

    def _load_session_markers(self) -> None:
        """
        Pick up markers already logged for this person/session, so restarting
        the tool mid-session does not lose the context. Ones older than the
        ring buffer simply sit off the left edge of the graph.
        """
        try:
            self.session_markers = self.writer.load_markers()
        except OSError:                                      # pragma: no cover
            self.session_markers = []
        self.plot.clear_markers()
        for marker in self.session_markers:
            self.plot.add_marker(marker.t_unix, marker.text,
                                 marker.kind, marker.uid)
        if hasattr(self, "marker_list"):
            self._refresh_marker_list()
            self._redraw_markers_now()
            if self.session_markers:
                self.marker_status.setText(
                    f"{len(self.session_markers)} marker(s) already logged "
                    "for this session")
                self.marker_status.setStyleSheet(
                    f"color: {COL_TEXT_DIM}; background: transparent;")
        self._sync_marker_controls()

    def cut_since_last_marker(self) -> None:
        """Save the span between the newest marker and now as training data."""
        if not self.marker_cut_check.isChecked():
            self.marker_status.setText(
                "cutting at markers is switched off -- tick the box above")
            self.marker_status.setStyleSheet(
                f"color: {COL_BAD}; background: transparent;")
            return
        if not self.session_markers:
            self.marker_status.setText("no marker to cut from -- add one first")
            self.marker_status.setStyleSheet(
                f"color: {COL_BAD}; background: transparent;")
            return

        marker = self.session_markers[-1]
        start_abs = marker.t_unix
        end_abs = self._marker_now()
        label = self.label_gesture_combo.currentText()

        rec = self._write_span_as_recording(
            start_abs, end_abs, label, "cut_at_marker",
            extra_meta={
                "cut_from_marker_uid": marker.uid,
                "cut_from_marker_text": marker.text,
                "cut_from_marker_kind": marker.kind,
            },
            dialog_title="Cut since last marker",
        )
        if rec is None:
            return

        self.marker_status.setText(
            f"cut {label}: {rec.duration:.2f} s since "
            f"{marker.short_text(24)}")
        self.marker_status.setStyleSheet(
            f"color: {COL_GOOD}; background: transparent;")

    def undo_last_label(self) -> None:
        mark = self.plot.remove_last_mark()
        if mark is None:
            return
        rec = next((r for r in reversed(self.recordings)
                    if r.uid == mark.get("uid")), None)
        if rec is None:
            self._refresh_label_counts()
            return
        moved = self._quarantine(rec)
        self._remove_recording(rec)
        self._append_log(
            f"Removed {rec.label} rep {rec.repetition} "
            f"({moved} file(s) moved to _deleted).")
        self._refresh_label_counts()
        self._refresh_plan_info()

    def _gesture_counts(self) -> dict:
        """
        How many raw recordings exist per gesture, across the WHOLE dataset.

        Counted from disk rather than from this run, because that is what the
        trainer will see: the question being answered is "do I have enough to
        train", not "what did I just do".
        """
        counts = {gesture: 0 for gesture in GESTURES}
        try:
            summary = dataset_summary(os.path.join(PROJECT_DIR, DATA_FOLDER))
        except OSError:                                      # pragma: no cover
            return counts
        for person in summary.get("persons", {}).values():
            for session in person.values():
                for gesture, number in session.items():
                    counts[gesture] = counts.get(gesture, 0) + int(number)
        return counts

    def _refresh_gesture_counts(self) -> None:
        """The 'how much data do I have' readout in the classifier panel."""
        if not hasattr(self, "gesture_counts_label"):
            return
        counts = self._gesture_counts()
        total = sum(counts.values())
        classes = [g for g, n in counts.items() if n]

        rows = []
        for gesture in GESTURES:
            number = counts.get(gesture, 0)
            colour = (LABEL_COLORS.get(gesture, COL_TEXT)
                      if number else COL_TEXT_DIM)
            rows.append(
                f'<span style="color:{colour}">{gesture}&nbsp;<b>{number}</b>'
                f'</span>')
        warning = ""
        if len(classes) < 2:
            warning = (f' &mdash; <span style="color:{COL_WARN}">a model '
                       f'needs at least 2</span>')
        headline = (f'<div style="color:{COL_TEXT_DIM};font-size:{FS_SM}px">'
                    f'{total} recording(s) on disk across {len(classes)} '
                    f'gesture(s){warning}</div>')
        self.gesture_counts_label.setText(
            headline + '<div style="padding-top:3px">'
            + " &nbsp; ".join(rows) + '</div>')

    def _refresh_label_counts(self) -> None:
        """How many of each gesture are on disk for this subject/session."""
        counts = {}
        for entry in self.writer.read_manifest():
            if entry.get("stage") != STAGE_RAW:
                continue
            counts[entry.get("label", "?")] = \
                counts.get(entry.get("label", "?"), 0) + 1
        if not counts:
            self.label_counts.setText("nothing recorded in this session yet")
            return
        total = sum(counts.values())
        self.label_counts.setText(
            f"{total} on disk:  "
            + "  ".join(f"{name} {counts[name]}" for name in sorted(counts)))

    # ======================================================================
    # Retraining
    # ======================================================================

    def _on_section_toggled(self, section, opened: bool) -> None:
        """
        Bring a section the operator just opened into view.

        Opening the eighth section used to reveal nothing, because it unfolded
        below the fold of the scroll area. Qt needs the layout to settle
        before the new geometry is known, hence the single-shot timer.
        """
        if not opened:
            return
        QtCore.QTimer.singleShot(
            0, lambda: self._scroll_area.ensureWidgetVisible(section, 0, 0)
            if getattr(self, "_scroll_area", None) else None)

    def expand_section(self, key: str) -> None:
        """
        Unfold a side-panel section.

        Folding is a convenience, but a folded section also hides whatever
        happens inside it -- start a retrain with LIVE CLASSIFIER folded and
        the progress bar reports to nobody. Anything that begins showing
        progress calls this first.
        """
        section = getattr(self, "sections", {}).get(key)
        if section is not None and not section.expanded:
            section.set_expanded(True)

    def _refresh_spec_summary(self) -> None:
        """The one-line 'what the next retrain will train' readout."""
        if not hasattr(self, "spec_summary"):
            return
        text = describe_model_spec(self.model_spec)
        if self._spec_problems:
            text += "   (spec file was corrected on load)"
        self.spec_summary.setText("next retrain: " + text)
        self.spec_summary.setStyleSheet(
            f"color: {COL_WARN if self._spec_problems else COL_TEXT_DIM};"
            " background: transparent;")

    def edit_model_spec(self) -> None:
        """
        Open the model specification editor.

        Changing the spec only affects the NEXT retrain -- the model currently
        loaded was produced by whatever spec was in force when it was
        trained, which is recorded inside it.
        """
        path = model_spec_path(self.predictor.path or MODEL_PATH)
        dialog = ModelSpecDialog(self.model_spec, path, self)
        dialog.exec()
        if not dialog.saved:
            return

        self.model_spec, self._spec_problems = load_model_spec(path)
        self._refresh_spec_summary()
        self._append_log(
            f"Model spec saved: {describe_model_spec(self.model_spec)}.")
        enabled = enabled_models(self.model_spec)
        self._append_log(
            "  will train: " + ", ".join(m["name"] for m in enabled))
        if self.predictor.model is not None:
            self._append_log(
                "  the loaded model is unchanged -- retrain to apply this.")

    # -- model status ---------------------------------------------------------

    def _refresh_model_status(self, force: bool = False) -> None:
        """
        Show whether a model exists, what it was trained on, and whether the
        dataset has moved on since.

        The fingerprint is a directory walk (~30 ms for 119 recordings), so
        it is throttled rather than recomputed on every UI tick. `force` is
        for the moments when the answer definitely changed: a model loading,
        a recording being saved, a retrain finishing.
        """
        if not hasattr(self, "model_status_label"):
            return
        now = time.perf_counter()
        due = now - getattr(self, "_fingerprint_at", 0.0) > \
            getattr(self, "_fingerprint_every", 5.0)
        # Never walk the tree while samples are being captured. The walk is
        # on the GUI thread, and a recording in progress is the one moment
        # the interface must not stutter.
        if due and self.acq.capturing:
            due = False
        if force or due:
            t0 = time.perf_counter()
            try:
                self._fingerprint = dataset_fingerprint(
                    os.path.join(PROJECT_DIR, DATA_FOLDER))
            except OSError:
                self._fingerprint = None
            took = time.perf_counter() - t0
            # Back off on a big dataset rather than hitching the UI every
            # five seconds: 30 ms at 119 recordings, but this grows linearly
            # and the answer changes rarely.
            self._fingerprint_every = 5.0 if took < 0.15 else min(300.0,
                                                                  took * 400)
            self._fingerprint_at = now

        status = self.predictor.data_status(getattr(self, "_fingerprint", None))
        colour = {"good": COL_GOOD, "warn": COL_WARN,
                  "bad": COL_BAD}.get(status["severity"], COL_TEXT_DIM)
        self.model_status_label.setStyleSheet(
            f"color: {colour}; background: transparent;")
        self.model_status_label.setText(f"MODEL   {status['headline']}")
        self.model_status_label.setToolTip(status["detail"])

        #  The second line is the provenance itself -- what data went in --
        #  because that is the question that gets asked weeks later.
        payload = dict(self.predictor.payload_meta) if self.predictor.ready else {}
        data = payload.get("data") or {}
        if self.predictor.ready and data:
            self.model_data_label.setText(
                f"trained on  {describe_training_data(data)}")
        elif self.predictor.ready:
            self.model_data_label.setText(
                "trained on  (not recorded by this model)")
        else:
            on_disk = getattr(self, "_fingerprint", None) or {}
            self.model_data_label.setText(
                f"on disk  {describe_training_data(on_disk)}"
                if on_disk.get("n_recordings") else "no recordings on disk yet")
        self.model_data_label.setToolTip(status["detail"])

    # -- compute device badge -------------------------------------------------

    def _apply_compute_badge(self) -> None:
        """Paint the GPU/CPU badge from whatever the probe knows so far."""
        if not hasattr(self, "compute_label"):
            return
        head, detail, severity = compute_badge(self.compute_info)
        colour = {"good": COL_GOOD, "warn": COL_WARN}.get(severity,
                                                          COL_TEXT_DIM)
        self.compute_label.setStyleSheet(
            f"color: {colour}; background: transparent;")
        self.compute_label.setText(f"COMPUTE   {head}")
        self.compute_label.setToolTip(detail)

        if getattr(self, "status_compute", None) is not None:
            if not self.compute_info.get("probed"):
                short = ("GPU/CPU UNKNOWN" if self.compute_info.get("failed")
                         else "GPU/CPU ?")
            elif self.compute_info.get("cuda"):
                short = f"GPU  {self.compute_info.get('gpu') or 'CUDA'}"
            else:
                short = "CPU ONLY"
            self.status_compute.setText(short)
            self.status_compute.setStyleSheet(
                f"color: {colour}; padding: 0 10px;")
            self.status_compute.setToolTip(f"{head}\n{detail}")

    def start_compute_probe(self) -> None:
        """
        Ask a subprocess whether CUDA is usable, and badge the answer.

        A subprocess because importing torch here would take seconds and
        attach a GPU context to the process that must not drop EMG samples
        (CLAUDE.md section 13). Asynchronous because that import is slow
        enough -- measured ~7 s on this machine -- to visibly freeze the
        window if it ran inline.
        """
        if self.compute_process is not None:
            return
        self.compute_process = QtCore.QProcess(self)
        self.compute_process.setProgram(sys.executable)
        self.compute_process.setArguments(["-c", COMPUTE_PROBE_SNIPPET])
        self.compute_process.setProcessChannelMode(
            QtCore.QProcess.MergedChannels)
        self.compute_process.finished.connect(self._on_compute_probe_finished)
        self.compute_process.errorOccurred.connect(
            lambda *_: self._finish_compute_probe(
                blank_compute_info("the device probe could not be started",
                                   failed=True)))
        self.compute_process.start()
        # A probe that never answers would leave the badge saying "checking"
        # for the rest of the session. Say so instead.
        QtCore.QTimer.singleShot(
            int(COMPUTE_PROBE_TIMEOUT_S * 1000), self._compute_probe_timeout)
        self._apply_compute_badge()

    def _compute_probe_timeout(self) -> None:
        if self.compute_process is None:
            return                      # it answered in time
        try:
            self.compute_process.kill()
        except RuntimeError:
            pass
        self._finish_compute_probe(
            blank_compute_info("the device probe did not answer in "
                               f"{COMPUTE_PROBE_TIMEOUT_S:.0f} s",
                               failed=True))

    def _on_compute_probe_finished(self, *_) -> None:
        text = ""
        if self.compute_process is not None:
            try:
                text = bytes(
                    self.compute_process.readAllStandardOutput()).decode(
                        "utf-8", errors="replace")
            except RuntimeError:
                text = ""
        self._finish_compute_probe(parse_compute_probe(text))

    def _finish_compute_probe(self, info: dict) -> None:
        if self.compute_process is not None:
            try:
                self.compute_process.deleteLater()
            except RuntimeError:
                # Both finished and errorOccurred can land here, and during
                # interpreter teardown the C++ side may already be gone. The
                # answer still matters; the handle does not.
                pass
            self.compute_process = None
        self.compute_info = info
        self._apply_compute_badge()
        head, _, _ = compute_badge(info)
        self._append_log(f"Compute device: {head}.")

    def retrain_model(self, confirm: bool = True, quick: bool = False) -> None:
        """
        Retrain on everything recorded so far.

        `confirm=False` skips the "are you sure" dialog and uses `quick`
        directly. That is for `--train` on the command line, where asking the
        question again would be asking it twice; every other guard (a session
        running, too little data, a training already going) still applies.

        train_model.py reads the whole dataset tree, so a retrain always uses
        the new recordings PLUS every earlier session and subject. There is no
        mode that trains on just the latest batch -- that would throw away
        most of the data and is never what you want.

        It runs as a separate process: scikit-learn with n_jobs=-1 would
        otherwise fight the acquisition thread for the CPU, and a crash in
        training must not take the recorder down with it.
        """
        if self.training_process is not None:
            answer = QtWidgets.QMessageBox.question(
                self, "Training in progress",
                "Training is already running. Cancel it?",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer == QtWidgets.QMessageBox.Yes:
                self._training_cancelled = True
                self.training_process.kill()
            return

        if self.sequence.running:
            QtWidgets.QMessageBox.information(
                self, "Retrain",
                "Finish or stop the recording session first.\n\n"
                "Training saturates the CPU and would disturb the live "
                "stream, which could corrupt the recordings in progress.",
            )
            return

        script = os.path.join(PROJECT_DIR, "train_model.py")
        if not os.path.exists(script):
            QtWidgets.QMessageBox.warning(
                self, "Retrain", f"train_model.py not found at:\n{script}")
            return

        summary = dataset_summary(os.path.join(PROJECT_DIR, DATA_FOLDER))
        if summary["total_raw"] < 12:
            QtWidgets.QMessageBox.information(
                self, "Retrain",
                f"Only {summary['total_raw']} recordings on disk. Collect "
                "more before training -- a handful cannot be validated.",
            )
            return

        people = summary["persons"]
        per_person = "\n".join(
            f"    {person}: " + ", ".join(
                f"{session} ({sum(counts.values())})"
                for session, counts in sorted(people[person].items()))
            for person in sorted(people))

        box = QtWidgets.QMessageBox(self) if confirm else None
        if box is None:
            self._append_log(
                f"Retrain requested on the command line "
                f"({'quick' if quick else 'full sweep'}).")
        else:
            box.setWindowTitle("Retrain on all data")
            box.setIcon(QtWidgets.QMessageBox.Question)
            box.setText(
                f"Retrain using ALL {summary['total_raw']} recordings on disk."
            )
        if box is not None:
            box.setInformativeText(
                f"{len(people)} subject(s):\n{per_person}\n\n"
                "Everything is used together -- the recordings from this run "
                "are added to the earlier ones, not trained on separately.\n\n"
                "Quick: a reduced grid, about a minute.\n"
                # Named from the spec rather than hard-coded: the GUI always
                # passes --spec, so "21 models" stopped being true the moment
                # MODEL SPEC could switch them off.
                # "every setting" was true of the old exhaustive sweep,
                # which measured 3 h 16 m. It now SCREENS every setting
                # cheaply and fully probes only a shortlist -- measured at
                # 0.02 points of accuracy. Saying "every setting" would
                # overstate it; saying "a while" would overstate the wait.
                f"Full:  screens every setting, then fully probes the best "
                f"few on all {len(enabled_models(self.model_spec))} enabled "
                "model(s) -- roughly 15 minutes.\n\n"
                "The current model is archived, not overwritten.\n\n"
                # Said here, before a long sweep, rather than leaving the GPU
                # badge above to imply something it does not mean.
                f"Runs on: CPU ({os.cpu_count()} logical cores) -- "
                "scikit-learn has no GPU backend, and there is no GPU build "
                "of these algorithms for Windows/Python 3.14."
            )
            quick_button = box.addButton("Quick",
                                         QtWidgets.QMessageBox.AcceptRole)
            full_button = box.addButton("Full sweep",
                                        QtWidgets.QMessageBox.AcceptRole)
            box.addButton(QtWidgets.QMessageBox.Cancel)
            box.setDefaultButton(quick_button)
            box.exec()

            clicked = box.clickedButton()
            if clicked not in (quick_button, full_button):
                return
            quick = clicked is quick_button

        self.stop_prediction()
        # Both: the detailed panel is in TRAINING, and the original bar and
        # status line are in LIVE CLASSIFIER next to the RETRAIN button that
        # was just pressed.
        self.expand_section("prediction")
        self.expand_section("training")
        self.training_started_at = time.perf_counter()
        self._reset_training_panel(
            f"train_model.py  --  {summary['total_raw']} recordings"
            + (" (quick)" if quick else " (full sweep)"))
        self.training_output = []
        self.training_process = QtCore.QProcess(self)
        self.training_process.setProgram(sys.executable)
        # Pass the paths the APP is configured with, rather than letting
        # train_model.py fall back to its own defaults -- otherwise a
        # changed DATA_FOLDER or MODEL_PATH would silently train on, and
        # overwrite, something else entirely.
        root = os.path.join(PROJECT_DIR, DATA_FOLDER)
        out_dir = os.path.dirname(os.path.abspath(
            self.predictor.full_path)) or PROJECT_DIR
        args = ["-u", script, "--root", root, "--out-dir", out_dir,
                "--model-path", self.predictor.full_path]

        # Write the spec out first so the subprocess reads exactly what the
        # GUI is showing, even if it was never saved from the dialog.
        spec_file = model_spec_path(self.predictor.full_path)
        try:
            save_model_spec(spec_file, self.model_spec)
            args += ["--spec", spec_file]
            self._append_log(
                f"Training with {describe_model_spec(self.model_spec)}.")
        except OSError as exc:
            # Not fatal: without --spec the trainer uses its built-in roster,
            # which is the old behaviour. Say so rather than failing silently.
            self._append_log(
                f"could not write the model spec ({exc}); training with the "
                "built-in roster instead")

        if quick:
            args.append("--quick")
        self.training_process.setArguments(args)
        self.training_process.setWorkingDirectory(PROJECT_DIR)
        self.training_process.setProcessChannelMode(
            QtCore.QProcess.MergedChannels)
        self.training_process.readyReadStandardOutput.connect(
            self._on_training_output)
        self.training_process.finished.connect(self._on_training_finished)
        self.training_process.errorOccurred.connect(self._on_training_error)
        self.training_process.start()

        self.retrain_button.setText("CANCEL TRAINING")
        self.training_started_at = time.perf_counter()
        self.training_phase_text = (
            f"starting on {summary['total_raw']} recordings"
            + (" (quick)" if quick else " (full sweep)") + "...")
        self.training_progress.setValue(0)
        self.training_progress.setFormat("%p%")
        self.training_progress.setVisible(True)
        self.training_status.setStyleSheet(
            f"color: {COL_WARN}; background: transparent;")
        self._refresh_training_status()
        self._append_log(
            f"Retraining started on all {summary['total_raw']} recordings "
            f"({'quick' if quick else 'full sweep'}), on the CPU."
        )

    def _on_training_output(self) -> None:
        if self.training_process is None:
            return
        raw = bytes(self.training_process.readAllStandardOutput())
        text = raw.decode("utf-8", errors="replace")
        for line in text.splitlines():
            line = line.rstrip()
            if not line:
                continue
            if line.startswith("PROGRESS "):
                self._update_training_progress(line)
                continue
            if line.startswith("TRAINEVENT "):
                # The detailed panel feed. A malformed line is ignored
                # rather than allowed to break the output pump.
                try:
                    self._on_train_event(json.loads(line[len("TRAINEVENT "):]))
                except (ValueError, TypeError, KeyError):
                    pass
                continue

            self.training_output.append(line)
            if len(self.training_output) > 4000:
                del self.training_output[:2000]
            # Keep the log readable: show the lines that mean something.
            if any(key in line for key in (
                    "Stage", "Evaluated", "WINNER", "saved:", "per-window",
                    "per-recording", "...", "recordings", "subject")):
                self._append_log(f"train: {line.strip()}")
            self.training_phase_text = line.strip()[:90]
            self._refresh_training_status()

    def _update_training_progress(self, line: str) -> None:
        """
        Turn a `PROGRESS done total phase` line into a bar and a time estimate.

        The estimate comes from the rate achieved so far rather than a fixed
        guess, so it corrects itself as the sweep goes: combinations differ a
        lot in cost and an up-front guess would be wrong by minutes.
        """
        parts = line.split()
        try:
            done, total = int(parts[1]), int(parts[2])
        except (IndexError, ValueError):
            return
        phase = " ".join(parts[3:]) or "working"

        fraction = done / max(1, total)
        self.training_progress.setVisible(True)
        self.training_progress.setValue(int(np.clip(fraction, 0, 1) * 1000))

        elapsed = time.perf_counter() - self.training_started_at
        eta = None
        # Ignore the first few steps: the rate is meaningless until the
        # process has warmed up and loaded the dataset.
        if done >= 5 and fraction > 0.01:
            eta = max(0.0, elapsed / fraction - elapsed)

        self.training_phase_text = (
            f"{phase}  {done}/{total}"
            + (f"  --  about {fmt_duration(eta)} left" if eta is not None
               else "  --  estimating...")
        )
        self.training_progress.setFormat(
            f"%p%  ({fmt_duration(eta)} left)" if eta is not None else "%p%")
        self._refresh_training_status()

    def _refresh_training_status(self) -> None:
        elapsed = time.perf_counter() - self.training_started_at
        self.training_status.setText(
            f"[CPU]  [{fmt_duration(elapsed)} elapsed]  "
            f"{self.training_phase_text}")

    def _on_training_error(self, _error) -> None:
        if self.training_process is None:
            return
        self._append_log(
            f"Training process error: {self.training_process.errorString()}")

    def _on_training_finished(self, exit_code: int, _status) -> None:
        self.training_process = None
        self.retrain_button.setText("RETRAIN ON ALL DATA")
        self.training_progress.setVisible(False)

        # Terminal state on the panel. The completed case is painted by the
        # trainer's own final TRAINEVENT, which carries the real metrics;
        # these two are the paths where no such event ever arrives.
        cancelled = bool(getattr(self, "_training_cancelled", False))
        if cancelled:
            self._set_train_field("status", "cancelled", COL_WARN)
            self._set_train_field("eta", "--")
            self.train_message.setText("Cancelled before finishing. "
                                       "The previous model is untouched.")
        elif exit_code != 0:
            self._set_train_field("status", "failed", COL_BAD)
            self._set_train_field("eta", "--")
            self.train_message.setText(
                f"train_model.py exited with code {exit_code}. "
                "The previous model is untouched.")
        self._training_cancelled = False
        took = fmt_duration(time.perf_counter() - self.training_started_at)
        tail = "\n".join(self.training_output[-25:])

        # Exit code 2 is the trainer's "there is nothing here to learn from"
        # -- not enough gesture classes. That is a fact about the dataset, not
        # a crash, so it gets advice instead of a red failure box.
        if exit_code == TRAIN_EXIT_NEED_MORE_DATA:
            counts = self._gesture_counts()
            have = [g for g, n in counts.items() if n]
            self.training_status.setStyleSheet(
                f"color: {COL_WARN}; background: transparent;")
            self.training_status.setText(
                f"not trained: only {len(have)} gesture class"
                + ("" if len(have) == 1 else "es") + " recorded")
            self._append_log(
                "Training skipped: a classifier needs at least two gestures, "
                f"and only {', '.join(have) or 'none'} is on disk.")
            missing = [g for g in GESTURES if not counts.get(g)]
            QtWidgets.QMessageBox.information(
                self, "More gestures needed",
                "There is only one gesture on disk, so there is nothing to "
                "tell apart yet.\n\n"
                + "On disk:\n"
                + "\n".join(f"    {g}: {counts[g]}" for g in have)
                + ("\n\nNot recorded yet:\n    " + ", ".join(missing)
                   if missing else "")
                + "\n\nRecord a second gesture and press RETRAIN again. "
                "Nothing was changed; your existing model is untouched.")
            self._refresh_gesture_counts()
            return

        if cancelled:
            # The operator stopped it. That is not a failure, and a red error
            # box claiming one would be wrong -- and would also pop up behind
            # their back after they deliberately pressed cancel.
            self.training_status.setStyleSheet(
                f"color: {COL_WARN}; background: transparent;")
            self.training_status.setText(f"training cancelled after {took}")
            self._append_log(
                f"Retraining cancelled after {took}. The previous model is "
                "untouched.")
            return

        if exit_code != 0:
            self.training_status.setStyleSheet(
                f"color: {COL_BAD}; background: transparent;")
            self.training_status.setText(f"training failed (exit {exit_code})")
            self._append_log(f"Retraining failed with exit code {exit_code}.")
            QtWidgets.QMessageBox.warning(
                self, "Training failed",
                f"train_model.py exited with code {exit_code}.\n\n{tail}")
            return

        before = self.predictor.stats.get("recording_vote_accuracy")
        ok = self.predictor.load()
        # The trainer writes a fresh panel beside the model. Reload it only if
        # it was already in memory; otherwise leave it for the first MODEL
        # STATS open, exactly as at startup.
        if self.decision_panel.loaded:
            self.decision_panel.warm = False
            self.decision_panel.load()
        self._rebuild_prob_bars()
        self._refresh_prediction_panel()

        if not ok:
            self.training_status.setStyleSheet(
                f"color: {COL_BAD}; background: transparent;")
            self.training_status.setText(
                f"trained, but the model would not load: {self.predictor.status}")
            self._append_log(
                f"Training finished but loading failed: {self.predictor.status}")
            return

        stats = self.predictor.stats
        after = stats.get("recording_vote_accuracy")
        delta = ""
        if before is not None and after is not None:
            change = (after - before) * 100
            delta = f"   ({change:+.1f} points vs the previous model)"

        self.training_status.setStyleSheet(
            f"color: {COL_GOOD}; background: transparent;")
        self.training_status.setText(
            (f"{stats.get('model_name', 'model')}: "
             f"{after * 100:.1f} % per gesture, took {took}") if after
            else f"model reloaded in {took}")
        self._append_log(
            f"Retraining done: {stats.get('model_name')} "
            f"{(after or 0) * 100:.1f} % per gesture{delta}")

        QtWidgets.QMessageBox.information(
            self, "Retraining complete",
            f"New model: {stats.get('model_name', '?')}\n\n"
            f"Per gesture : {(after or 0) * 100:.1f} %{delta}\n"
            f"Per window  : {stats.get('window_accuracy', 0) * 100:.1f} %\n"
            f"Trained on  : {stats.get('n_recordings', '?')} of "
            f"{stats.get('n_recordings_total', '?')} recordings, "
            f"{len(stats.get('subjects', []))} subject(s)\n"
            # Spelled out at the one moment the question is live, so it does
            # not have to be reconstructed from the panel a week later.
            f"Data        : "
            f"{describe_training_data(self.predictor.payload_meta.get('data'))}"
            "\n\nOpen MODEL STATS for the full breakdown.",
        )

    # ======================================================================
    # Shutdown
    # ======================================================================

    def _on_freeze_toggled(self, frozen: bool) -> None:
        self.plot.set_frozen(frozen)
        self.freeze_button.setText("Resume graph" if frozen else "Freeze graph")

    def closeEvent(self, event) -> None:
        if self.sequence.running:
            answer = QtWidgets.QMessageBox.question(
                self, "Quit",
                "A recording session is still running. Quit anyway?\n\n"
                "Everything already saved stays on disk.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return

        self.sequence.stop()
        self.stop_prediction()
        if self.training_process is not None:
            self.training_process.kill()
            self.training_process.waitForFinished(2000)
            self.training_process = None
        if self.compute_process is not None:
            try:
                self.compute_process.kill()
                self.compute_process.waitForFinished(1000)
            except RuntimeError:
                pass
            self.compute_process = None
        for timer in (self.plot_timer, self.seq_timer, self.ui_timer,
                      self.pred_timer):
            timer.stop()
        if self.export_worker is not None:
            self.export_worker.wait(3000)
        self.acq.stop()
        self.acq.join(timeout=3.0)
        super().closeEvent(event)


# ==============================================================================
# SECTION 8 -- ENTRY POINT
# ==============================================================================


def force_utf8_console() -> None:
    """
    Make console output safe for non-ASCII paths.

    This project lives under "OneDrive\\Pocitac\\EMG" -- with a c-caron. The
    Windows console defaults to cp1252, so printing that path raises
    UnicodeEncodeError and takes the whole program down with it (observed on
    `--summary`, and it would also hit any warning that quotes a file path).
    errors="replace" makes a stray character cosmetic instead of fatal.

    Only called from main(), so importing this module never reconfigures
    somebody else's streams.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def print_cli_summary() -> None:
    """`python emg_dataset_tool.py --summary` -- dataset stats without a GUI."""
    summary = dataset_summary(os.path.join(PROJECT_DIR, DATA_FOLDER))
    print(f"\nDataset root: {summary['root']}")
    print(f"Raw recordings: {summary['total_raw']}   "
          f"Trimmed: {summary['total_trimmed']}\n")
    if not summary["persons"]:
        print("No recordings yet.\n")
        return
    for person in sorted(summary["persons"]):
        print(f"  {person}")
        for session in sorted(summary["persons"][person]):
            counts = summary["persons"][person][session]
            total = sum(counts.values())
            print(f"    {session}  ({total})")
            for label in sorted(counts):
                print(f"      {label:<14} {counts[label]}")
    print()


def main(argv=None) -> int:
    argv = list(sys.argv if argv is None else argv)
    force_utf8_console()

    if "--summary" in argv:
        print_cli_summary()
        return 0

    if not HAVE_QT:
        print("PySide6 and pyqtgraph are required for the GUI.")
        print(f"Import error: {QT_IMPORT_ERROR}")
        print("\nInstall them with:\n    python -m pip install -r requirements.txt")
        return 1

    t_start = time.perf_counter()

    QtCore.QCoreApplication.setAttribute(
        QtCore.Qt.AA_DontCreateNativeWidgetSiblings, True
    )
    app = QtWidgets.QApplication(argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    apply_theme(app)

    print(f"[startup] building the interface ...", flush=True)
    window = MainWindow()
    if WINDOW_MAXIMIZE_IF_SMALL and window.fitted_to_screen:
        # The preferred size did not fit, so the screen is the constraint:
        # maximised is then strictly better than a window the operator would
        # immediately maximise by hand.
        window.showMaximized()
    else:
        window.show()
    print(
        f"[startup] window up after {time.perf_counter() - t_start:.1f} s "
        f"(press Ctrl+C in this console, or close the window, to quit)",
        flush=True,
    )

    # `--train` / `--train-quick`: start a retrain as soon as the window is
    # up, so the sweep can be launched and watched without clicking. It takes
    # exactly the path the RETRAIN button takes -- same subprocess, same
    # progress panel, same completion dialog -- with the confirmation dialog
    # answered for it, because the flag IS the confirmation.
    if "--train" in argv or "--train-quick" in argv:
        quick = "--train-quick" in argv
        QtCore.QTimer.singleShot(
            1200, lambda: window.retrain_model(
                confirm=False, quick=quick))

    # Ctrl+C in the console must end the app cleanly. Qt's event loop runs in
    # C++, so Python never gets to run its SIGINT handler while the loop is
    # blocked: the KeyboardInterrupt instead surfaces at whatever arbitrary
    # point the interpreter next regains control, which in practice is inside
    # a pyqtgraph paintEvent -- and pyqtgraph catches it, prints a traceback,
    # repaints, gets interrupted again, and floods the console. The no-op
    # timer gives the interpreter a slice every 200 ms so the handler below
    # actually runs and shuts the window down the same way the X button does.
    _sigint_timer = QtCore.QTimer()
    _sigint_timer.start(200)
    _sigint_timer.timeout.connect(lambda: None)

    def _on_sigint(_signum, _frame):
        print("", flush=True)
        print("[startup] Ctrl+C -- shutting down ...", flush=True)
        window.close()
        app.quit()

    try:
        signal.signal(signal.SIGINT, _on_sigint)
    except (ValueError, OSError):                        # pragma: no cover
        pass                                             # not on the main thread

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
