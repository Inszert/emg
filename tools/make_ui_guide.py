# -*- coding: utf-8 -*-
"""Render every screen of the GUI and annotate it at real widget coordinates.

Two things make these pictures trustworthy rather than decorative:

* Every arrow is placed from `widget.geometry()` mapped into the rendered
  widget's coordinates. A renamed or deleted attribute raises here instead of
  producing a picture that points confidently at the wrong control.
* The real model, the real decision panel and a real take are loaded, so the
  numbers on screen are the project's own rather than invented.

The native platform plugin is used, not `offscreen`, because offscreen ships
no font database and renders every glyph as a tofu box. `WA_DontShowOnScreen`
keeps the window off the operator's display while still running layout.

Usage:  python tools/make_ui_guide.py . docs/images
"""
import io
import os
import sys

PROJ, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, PROJ)
os.chdir(PROJ)

import numpy as np                                               # noqa: E402
import emg_dataset_tool as T                                     # noqa: E402
from PySide6 import QtCore, QtGui, QtWidgets                      # noqa: E402

os.makedirs(OUT, exist_ok=True)
T.MYO_PORT = "COM254"            # cannot exist -> no hardware is touched

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
T.apply_theme(app)               # main() does this; without it panels are white

MARGIN = 470
BADGE_R = 15
FONT = "Consolas"


def _qc(n):
    return QtGui.QColor(n)


def resolve(root, spec):
    """Dotted attribute path, or 'section:TITLE', or 'find:ClassName'."""
    if spec.startswith("section:"):
        want = spec[8:].lower()
        for sec in root.findChildren(T.CollapsibleSection):
            if sec.header.text().lower().startswith(want):
                return sec
        have = [s.header.text() for s in root.findChildren(T.CollapsibleSection)]
        raise AttributeError(f"no section {spec[8:]!r}; have {have}")
    if spec.startswith("box:"):
        want = spec[4:].lower()
        for box in root.findChildren(QtWidgets.QGroupBox):
            if box.title().lower().startswith(want):
                return box
        have = [b.title() for b in root.findChildren(QtWidgets.QGroupBox)]
        raise AttributeError(f"no group box {spec[4:]!r}; have {have}")
    obj = root
    for part in spec.split("."):
        obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
    return obj


def annotate(widget, origin, shots, path, title):
    """`widget` is rendered; `origin` is what attribute paths resolve against."""
    for _ in range(4):
        app.processEvents()

    targets = []
    for spec, text in shots:
        w = resolve(origin, spec)
        tl = w.mapTo(widget, QtCore.QPoint(0, 0))
        r = QtCore.QRect(tl, w.size())
        if not (r.intersects(widget.rect())):
            text += "   [scrolled out of view here]"
        targets.append((r, text))

    shot = QtGui.QPixmap(widget.size())
    shot.fill(_qc(T.COL_BG))
    widget.render(shot)

    W, H = shot.width(), shot.height()
    canvas = QtGui.QPixmap(W + MARGIN, H + 70)
    canvas.fill(_qc("#0a0c10"))
    p = QtGui.QPainter(canvas)
    p.setRenderHint(QtGui.QPainter.Antialiasing)
    p.drawPixmap(0, 70, shot)
    p.setPen(_qc(T.COL_ACCENT))
    p.setFont(QtGui.QFont(FONT, 19, QtGui.QFont.Bold))
    p.drawText(26, 46, title)

    order = sorted(range(len(targets)), key=lambda i: targets[i][0].center().y())
    slot_h = max(50, (H - 40) // max(len(targets), 1))
    lf = QtGui.QFont(FONT, 11)
    fm = QtGui.QFontMetrics(lf)

    for slot, idx in enumerate(order):
        rect, text = targets[idx]
        rect = rect.translated(0, 70)
        n, ly, lx = idx + 1, 70 + 32 + slot * slot_h, W + 34

        p.setPen(QtGui.QPen(_qc(T.COL_ACCENT), 2.4))
        p.setBrush(QtGui.QBrush(QtGui.QColor(56, 189, 248, 26)))
        p.drawRoundedRect(rect.adjusted(-3, -3, 3, 3), 5, 5)

        ax = min(rect.right() + 4, W - 2)
        ay = max(76, min(rect.center().y(), H + 66))
        bx, by, mid = lx - 10, ly, (min(rect.right(), W) + lx) // 2
        p.setPen(QtGui.QPen(_qc(T.COL_ACCENT), 1.8))
        p.setBrush(QtCore.Qt.NoBrush)
        pth = QtGui.QPainterPath(QtCore.QPointF(ax, ay))
        pth.cubicTo(mid, ay, mid, by, bx, by)
        p.drawPath(pth)
        p.setBrush(QtGui.QBrush(_qc(T.COL_ACCENT)))
        p.drawPolygon(QtGui.QPolygonF([
            QtCore.QPointF(ax, ay), QtCore.QPointF(ax + 11, ay - 5),
            QtCore.QPointF(ax + 11, ay + 5)]))

        for cx, cy in ((max(rect.left() - 2, 14), max(rect.top() - 2, 84)),
                       (lx, ly)):
            p.setBrush(QtGui.QBrush(_qc(T.COL_ACCENT)))
            p.setPen(QtGui.QPen(_qc("#061018"), 1))
            p.drawEllipse(QtCore.QPoint(cx, cy), BADGE_R, BADGE_R)
            p.setPen(_qc("#061018"))
            p.setFont(QtGui.QFont(FONT, 11, QtGui.QFont.Bold))
            p.drawText(QtCore.QRect(cx - BADGE_R, cy - BADGE_R,
                                    BADGE_R * 2, BADGE_R * 2),
                       QtCore.Qt.AlignCenter, str(n))

        p.setFont(lf)
        p.setPen(_qc(T.COL_TEXT))
        line, yy = "", ly - 7
        for word in text.split():
            trial = (line + " " + word).strip()
            if fm.horizontalAdvance(trial) > MARGIN - 70 and line:
                p.drawText(lx + BADGE_R + 10, yy + fm.ascent(), line)
                yy += fm.height()
                line = word
            else:
                line = trial
        if line:
            p.drawText(lx + BADGE_R + 10, yy + fm.ascent(), line)

    p.end()
    canvas.save(path, "PNG")
    print(f"  {os.path.basename(path):24} {canvas.width()}x{canvas.height()}  "
          f"{len(targets)} callouts")


def inject(window, seconds=12.0, fs=50.0):
    rng = np.random.RandomState(7)
    n = int(seconds * fs)
    t0 = T.time.time() - seconds
    prof = [np.array([.3, .4, 1., .9, .5, .2, .2, .25]),
            np.array([1., .8, .3, .2, .2, .3, .6, .9]),
            np.array([.2, .2, .3, .5, 1., .9, .7, .3])]
    for i in range(n):
        frac, base = i / n, 22.0 + rng.randn(8) * 3.0
        for k, (s, e) in enumerate(((.12, .30), (.42, .60), (.72, .90))):
            if s <= frac <= e:
                r = np.sin(np.pi * min(1., max(0., (frac - s) / (e - s)))) ** .5
                base += prof[k] * 210.0 * r * (1 + .25 * rng.randn())
        window.acq.ring.push(t0 + i / fs, np.clip(base, 0, 2047))
    window._tick_plot()
    window._tick_slow_ui()
    for _ in range(3):
        app.processEvents()


# ================================================================== main window
print("rendering ...")
win = T.MainWindow()
win.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
win.resize(1880, 1120)
win.show()
for _ in range(8):
    app.processEvents()
inject(win)

annotate(win, win, [
    ("plot", "THE GRAPH -- eight live traces, one per pod, newest on the "
             "right. It stays live through every other operation: the review "
             "step draws a finished take over these same curves without ever "
             "pausing capture."),
    ("banner", "THE BANNER -- the one thing to look at during a session. It "
               "says what to do now (REST / GET READY / RECORD NOW) and "
               "turns orange when the link drops."),
    ("settings_button", "SETTINGS (gear, Ctrl+,). Add gestures, choose which "
                        "are recorded, and switch a gesture on or off in the "
                        "ML model AND the neural net -- no retrain."),
    ("conn_pill", "LINK STATE. Red until the armband is found. Only the "
                  "acquisition thread ever touches the serial port -- a "
                  "write from the GUI thread corrupts the BLE framing."),
    ("window_spin", "GRAPH WINDOW in seconds. Display only; it changes "
                    "nothing that gets recorded."),
    ("autoscale_check", "AUTOSCALE grows instantly but shrinks slowly, so a "
                        "peak is never clipped out of view and the axis does "
                        "not jitter."),
    ("envelope_check", "RMS ENVELOPE -- a smoothed overlay. A display aid; "
                       "the raw samples are what get saved."),
    ("freeze_button", "FREEZE holds the picture. Capture keeps running "
                      "underneath -- freezing never drops a sample."),
    ("start_button", "START SESSION runs the cued plan: REST -> GET READY -> "
                     "3-2-1 -> RECORD, for every gesture x repetition."),
    ("record_button", "RECORD (manual) takes one take right now, outside the "
                      "plan."),
    ("export_button", "EXPORT DATASET writes a combined ML bundle "
                      "(npz + index csv + summary) into exports/."),
    ("graph_button", "SAVE GRAPH exports the current view as PNG or SVG into "
                     "the session's _graphs/ folder."),
    ("status_compute", "GPU/CPU chip. Answered by a subprocess probe, never "
                       "by importing torch into the recorder. Reads "
                       "'checking...' until it knows -- never a guess."),
], os.path.join(OUT, "ui_main.png"),
    "1. MAIN WINDOW  --  the graph, the banner, the session controls")

# ================================================================== side panel
scrolls = [s for s in win.findChildren(QtWidgets.QScrollArea)
           if s.widget() and s.widget().findChildren(T.CollapsibleSection)]
panel = scrolls[0].widget()

PANEL_SHOTS = [
    ("section:SUBJECT", "SUBJECT / SESSION -- who is wearing the band and "
     "which session this is. Use a NEW person id for every new subject: "
     "folders are per person, and that is what makes leave-one-person-out "
     "validation possible at all."),
    ("section:LABEL ON GRAPH", "LABEL ON GRAPH -- select a span on the live "
     "plot and save it as a labelled clip. No countdown, no plan. SELECT "
     "LAST n s sets the span, ADD SELECTION writes it, UNDO LAST takes it "
     "back."),
    ("section:FREE TAKE", "FREE TAKE -- record now, cut later. One "
     "continuous capture, minutes long; carve clips out of it whenever. A "
     "take is NEVER training data: it lives in _takes/, which discovery "
     "prunes, and carries a stage no selector asks for. DELETE TAKE(S) "
     "quarantines, it does not erase."),
    ("section:MARKERS", "MARKERS & NOTES -- Ctrl+M for a note, Ctrl+B for a "
     "break. Stamped off the newest SAMPLE rather than the GUI clock, so "
     "they line up with the signal. Append-only: a correction is another "
     "marker, never an edit."),
    ("section:RECORDING PLAN", "RECORDING PLAN -- gestures x repetitions for "
     "the cued sequence. Interleaved order spreads fatigue and electrode "
     "drift across every label instead of baking it into the last one."),
]
PANEL_SHOTS_2 = [
    ("section:CHANNELS", "CHANNELS -- show or hide individual pods. Display "
     "only; all eight are always recorded."),
    ("section:RECORDINGS", "RECORDINGS (this run) -- what this sitting has "
     "produced, with quality and trim state per row. DELETE LAST "
     "quarantines into _deleted/. AUTO-TRIM ALL is the one button to avoid "
     "on clips cut from a take: it keeps the STRONGEST region, and a "
     "relaxed hold is the weakest part of the clip."),
    ("section:LIVE CLASSIFIER", "LIVE CLASSIFIER -- the deployed ML model "
     "(the app's answer): one-shot or continuous, drawn over the graph with "
     "per-class probability bars. The provenance line reads UP TO DATE / "
     "OUT OF DATE / TRAINED / NO MODEL. CALIBRATE is REQUIRED once per "
     "wearer -- the model refuses to predict without it. Picture 6 has every "
     "button."),
    ("section:NEURAL NETWORK", "NEURAL NETWORK -- the GRU, a separate second "
     "opinion with its own START, bars and NN STATS. It runs in its own "
     "process on the GPU and is never averaged into the answer above; where "
     "the two disagree, the panel says so."),
    ("section:TRAINING", "TRAINING -- progress, ETA and the live log of the "
     "running trainer. Training runs as a background job: close the app and "
     "it keeps going; reopen and it reattaches."),
    ("section:LOG", "LOG -- every save, quarantine, reconnect and failure, "
     "in order."),
]

for fname, shots, title in (
    ("ui_panel_1.png", PANEL_SHOTS,
     "2a. CONTROL PANEL (top)"),
    ("ui_panel_2.png", PANEL_SHOTS_2,
     "2b. CONTROL PANEL (bottom)"),
):
    wanted = {spec for spec, _ in shots}
    for sec in panel.findChildren(T.CollapsibleSection):
        want = any(sec.header.text().lower().startswith(w[8:].lower())
                   for w in wanted)
        sec.header.setChecked(want)
        app.processEvents()
    panel.resize(panel.width(), panel.sizeHint().height())
    for _ in range(4):
        app.processEvents()
    annotate(panel, win, shots, os.path.join(OUT, fname), title)

# ============================================================ MODEL STATS
win.decision_panel.load()
stats = T.ModelStatsDialog(win.predictor, win, panel=win.decision_panel)
stats.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
stats.resize(1500, 1180)
stats.show()
for _ in range(10):
    app.processEvents()
# LIVE needs the armband, so the trail would just read "waiting for a live
# window". RECORDED replays the out-of-fold windows captured during training,
# which is both available here and the honest half of the view.
stats.trail_view.recorded_button.click()
for _ in range(10):
    app.processEvents()

annotate(stats, stats, [
    ("space_view", "FEATURE SPACE -- every training window as one point, "
     "with the live window on the same axes. PCA, not LDA: an LDA picks axes "
     "that separate the labels, so it would look clean even when the "
     "separation is electrode drift. Colour by TAKE -- if that looks tidier "
     "than colour by GESTURE, the model is reading the recording session."),
    ("trail_view", "DECISION TRAIL -- one window, every candidate's vote, and "
     "how they fuse. LIVE shows members fitted on the whole dataset "
     "(demonstration, and the caption says so); RECORDED replays out-of-fold "
     "probabilities, where the window's recording was held out of the fold "
     "that scored it."),
], os.path.join(OUT, "ui_model_stats.png"),
    "3. MODEL STATS  --  how the model decided, and what it really sees")

# ====================================================== OPEN TAKE & CUT CLIPS
takes = win.writer.list_takes()
if takes:
    rec = T.load_recording(takes[0]["path"] if "path" in takes[0]
                           else takes[0]["npz"])
    cut = T.TrimDialog(rec, win.writer, win, extract_mode=True)
    cut.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
    cut.resize(1560, 950)
    cut.show()
    for _ in range(8):
        app.processEvents()
    annotate(cut, cut, [
        ("start_spin", "The clip's START, in seconds into the take. Drag the "
         "green band on the plot or type it here."),
        ("end_spin", "The clip's END. Keep the span on the settled, RELAXED "
         "part of the hold -- exclude flat rest at either edge, because a "
         "pure-rest window labelled as a gesture is the false-positive bug."),
        ("extract_gesture", "Which gesture this clip is. Subject metadata is "
         "copied from the TAKE, not from today's config."),
        ("clip_length_spin", "Default clip length, shared with SELECT LAST "
         "n s on the live graph. ~3 s suits the winning 0.3 s window."),
        ("extract_button", "EXTRACT CLIP (Ctrl+Space) writes an ordinary "
         "stage-raw recording with its OWN uid and full provenance "
         "(parent_take_uid, take_start_index). The take itself stays "
         "byte-identical."),
    ], os.path.join(OUT, "ui_cut_clips.png"),
        "4. OPEN TAKE(S) & CUT CLIPS  --  carving training clips out of a take")
else:
    print("  (no takes on disk -- skipped ui_cut_clips.png)")

# ================================================================ MODEL SPEC
import json as _json                                              # noqa: E402
_spec_path = os.path.join("models", "model_spec.json")
_spec = _json.load(io.open(_spec_path, encoding="utf-8")) if os.path.isfile(
    _spec_path) else {}
spec_dlg = T.ModelSpecDialog(_spec, _spec_path, win)
spec_dlg.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
spec_dlg.resize(1500, 940)
spec_dlg.show()
for _ in range(8):
    app.processEvents()

annotate(spec_dlg, spec_dlg, [
    ("model_list", "THE CANDIDATE LIST -- which classifiers the sweep trains. "
     "Tick to enable. This is DATA, not code: it is written to "
     "models/model_spec.json and the GUI passes --spec on every retrain, so "
     "what the subprocess trains is always what this dialog was showing."),
    ("add_combo", "ADD another candidate of this kind. You can have several "
     "of the same kind with different hyperparameters -- SVM at C=1, C=10 and "
     "C=100 are three rows, each scored separately."),
    ("box:HYPERPARAMETERS", "HYPERPARAMETERS for the selected candidate. A "
     "hand-edited model_spec.json cannot crash a sweep an hour in: "
     "validate_model_spec() never raises and never returns something the "
     "trainer cannot build -- unknown kinds are dropped, bad values coerced, "
     "out-of-range clamped, duplicate names renamed, and every correction is "
     "reported as a string."),
    ("box:VOTING", "VOTING ENSEMBLE -- soft averages the member probabilities, "
     "hard counts the votes. The ensemble is added as one more CANDIDATE, not "
     "as a replacement, so it is scored against its own members by the same "
     "cross-validation. 'always' forces it to be saved regardless, which is a "
     "choice rather than an improvement."),
    ("box:AUTO MODE", "AUTO MODE expands, it does not search. Each enabled "
     "model becomes up to max_per_model concrete candidates that go through "
     "the EXISTING evaluation path -- no second scoring route to keep honest, "
     "every combination gets its own row, and the sampling is seeded so a "
     "sweep is reproducible. The configured values are always combination #1."),
    ("auto_spin", "How many candidates per model. The cost is SELECTION BIAS: "
     "choosing the best of N on the same folds makes the winner's score "
     "optimistic. That is stated here, in the training log and in MODEL STATS "
     "rather than hidden -- safety rule 11."),
], os.path.join(OUT, "ui_model_spec.png"),
    "5. MODEL SPEC  --  which classifiers the sweep trains, and in what modes")

# ============================================ CLASSIFIER + NEURAL NET buttons
for sec in panel.findChildren(T.CollapsibleSection):
    title = sec.header.text().lower()
    sec.header.setChecked(title.startswith(("live classifier",
                                            "neural network")))
    app.processEvents()
panel.resize(panel.width(), panel.sizeHint().height())
for _ in range(4):
    app.processEvents()
annotate(panel, win, [
    ("pred_mode_combo", "ONE-SHOT or CONTINUOUS prediction."),
    ("predict_button", "PREDICT -- starts the live classifier. Needs a "
     "trained model AND a calibration for the person in the SUBJECT box."),
    ("calibration_button", "CALIBRATE / RECALIBRATE -- REST, then every "
     "gesture once (picture 7). Once per wearer; it stays valid until you "
     "recalibrate. Do it again whenever the band is put back on."),
    ("test_button", "TEST MODEL -- cues every gesture several times and "
     "scores the model on YOU, live (picture 9). The number cross-validation "
     "cannot give you."),
    ("retrain_button", "RETRAIN ON ALL DATA -- the ML model only, from the "
     "pinned spec (about 2 min). Runs in the background."),
    ("retrain_all_button", "RETRAIN ML + NEURAL NET -- one button, both "
     "models, one after the other (about 15 min). Closing the app does not "
     "stop it."),
    ("spec_button", "MODEL SPEC -- which classifiers are in the soft vote "
     "and on which setup (picture 5)."),
    ("nn_start_button", "START NN -- runs the GRU next to the classifier, on "
     "the GPU. Its answer is shown separately, never mixed in."),
    ("nn_stats_button", "NN STATS -- the net's own numbers and the side-by-"
     "side table against the deployed model, including SIZE (picture 8)."),
    ("nn_train_button", "TRAIN NEURAL NET -- the GRU 0.7 s only (about 10 "
     "min on the GPU)."),
    ("nn_setup_button", "NEURAL SETUP -- architecture and hyperparameters. "
     "GRU at 0.7 s is the only one selected and the recommended one."),
], os.path.join(OUT, "ui_classifier.png"),
    "6. CLASSIFIER + NEURAL NET")

# =============================================================== CALIBRATE
cal = T.CalibrationDialog(win, win.acq, T.PERSON_ID)
cal.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
cal.resize(1100, 900)
cal.show()
for _ in range(8):
    app.processEvents()
annotate(cal, cal, [
    ("state_label", "WHO is being calibrated and what they have now. A "
     "calibration belongs to the person in the SUBJECT box."),
    ("stream_label", "Is the armband sending samples? A capture needs a live "
     "stream -- a connected band that went to sleep sends nothing."),
    ("sweep_button", "START SWEEP -- the recommended way: cues REST, then "
     "each gesture in turn. Each channel's reference is the loudest it got "
     "across ALL gestures, so no channel is scaled from noise."),
    ("redo_button", "REDO SELECTED -- repeat one row. A capture that did not "
     "register (too weak, or a tense REST over 50 counts) is rejected, shown "
     "with the reason, and simply redone."),
    ("both_button", "QUICK -- relax 3 s, then one strong squeeze 3 s. Faster, "
     "but a squeeze leaves some channels quiet; the sweep above is better."),
    ("derive_button", "DERIVE FROM RECORDED DATA -- no armband needed: "
     "estimates the levels from this person's recordings. This is what "
     "TRAINING always uses."),
    ("save_button", "SAVE -- stores it in datasets/<person>/calibration.json "
     "(every calibration also goes into an append-only log)."),
    ("clear_button", "CLEAR -- removes it (the log keeps the record)."),
], os.path.join(OUT, "ui_calibration.png"),
    "7. CALIBRATE  --  rest + every gesture, once per wearer")

# ================================================================== NN STATS
nn_dlg = T.NeuralStatsDialog(win.nn_engine, win, predictor=win.predictor)
nn_dlg.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
nn_dlg.resize(1150, 1180)
nn_dlg.show()
for _ in range(8):
    app.processEvents()
if hasattr(nn_dlg, "compare_table"):
    for sa in nn_dlg.findChildren(QtWidgets.QScrollArea):
        sa.ensureWidgetVisible(nn_dlg.compare_table, 0, 40)
    for _ in range(6):
        app.processEvents()
    annotate(nn_dlg, nn_dlg, [
        ("compare_table", "AGAINST THE DEPLOYED MODEL -- the two models side "
         "by side, scored on the same protocol (a person held out). Accuracy "
         "per window and per gesture, macro F1, the input each one reads, "
         "SIZE on disk, LEARNED NUMBERS (tree nodes / weights) per vote "
         "member, and what each needs to run. Two independent results, "
         "never one blended figure."),
    ], os.path.join(OUT, "ui_nn_stats.png"),
        "8. NN STATS  --  the neural net against the deployed model")
else:
    print("  (no neural net trained -- skipped ui_nn_stats.png)")

# ================================================================ TEST MODEL
if win.predictor.ready:
    test = T.ModelTestDialog(win)
    test.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
    test.resize(1000, 860)
    test.show()
    for _ in range(8):
        app.processEvents()
    annotate(test, test, [
        ("honesty", "Whether this sitting is in the model's training data. "
         "If it is, the score is optimistic: re-don the band and use a new "
         "Session id for an honest number."),
        ("rounds_spin", "ROUNDS -- how many times every gesture is cued "
         "(interleaved, never the same twice in a row)."),
        ("hold_spin", "HOLD -- seconds per cue. The first moment of each "
         "hold is skipped while the hand settles."),
        ("cue", "THE CUE -- which gesture to make now."),
        ("bar", "Progress through the cues."),
        ("results", "RESULTS -- per trial (comparable with the trainer's "
         "per-recording vote), smoothed (what you see live) and per window, "
         "each with its confusion matrix. Saved to "
         "datasets/<person>/model_tests.jsonl."),
        ("start_button", "START TEST -- needs the armband streaming."),
    ], os.path.join(OUT, "ui_model_test.png"),
        "9. TEST MODEL  --  score the model on yourself, live")
else:
    print(f"  (model not usable here: {win.predictor.status} -- skipped "
          "ui_model_test.png)")

# ================================================================ SETTINGS
#  The dialog behind the gear icon. Shown with FIST switched off so the
#  picture says what an off gesture looks like; nothing is applied (the dialog
#  is never accepted), so the app's gesture set is untouched.
gest = T.SettingsDialog(win, model_labels=win._model_label_sets(),
                        selected=list(T.GESTURES), classifier_off=["FIST"])
gest.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
gest.resize(980, 1000)
gest.show()
#  A demo gesture, added to THIS dialog only.
gest.editor.custom_edit.setText("WRIST_FLEX")
gest.editor.cue_edit.setText("bend the wrist down, fingers relaxed")
gest.editor._add_custom()
for _ in range(8):
    app.processEvents()
annotate(gest, gest, [
    ("editor.boxes.REST", "RECORD -- cued by START SESSION, offered in every "
     "dropdown and calibrated by the sweep. Under each name: the "
     "instruction, how many recordings and people, and WHETHER THE MODELS "
     "KNOW IT."),
    ("editor.cls_boxes.OPEN", "USE IN CLASSIFIER -- on. Untick to switch "
     "this gesture OFF in the ML model and the neural net together."),
    ("editor.cls_boxes.FIST", "FIST is OFF here: masked out of both models' "
     "answers. NOTHING is retrained -- tick it again and it is back at once."),
    ("editor.cls_boxes.WRIST_FLEX", "A gesture you just added has no switch "
     "(n/a): no model has been trained on it. Record it, then RETRAIN."),
    ("editor.retrain_note", "What a retrain would pick up -- said before you "
     "have to ask."),
    ("editor.custom_edit", "NAME of the new gesture (it becomes a folder "
     "name)."),
    ("editor.cue_edit", "INSTRUCTION -- shown under the name in the session "
     "banner, the calibration sweep and TEST MODEL."),
    ("apply_button", "APPLY. When a recorded gesture is missing from a model, "
     "APPLY + RETRAIN appears beside it."),
], os.path.join(OUT, "ui_settings.png"),
    "10. SETTINGS (gear icon)  --  gestures: add, record, switch on/off")
gest.reject()

# ====================================================== SETTINGS: the other tabs
#  Nothing is applied (the dialogs are never accepted).
def settings_tab(index, filename, title, shots, prepare=None):
    dlg = T.SettingsDialog(win, model_labels=win._model_label_sets(),
                           selected=list(T.GESTURES), classifier_off=[],
                           defaults_note=win._defaults_note())
    dlg.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
    dlg.resize(1100, 1000)
    dlg.show()
    dlg.tabs.setCurrentIndex(index)
    if prepare:
        prepare(dlg)
    for _ in range(8):
        app.processEvents()
    annotate(dlg, dlg, shots, os.path.join(OUT, filename), title)
    dlg.reject()


def _demo_models(dlg):
    mt = dlg.models_tab
    mt.enable_box.setChecked(True)
    mt._select_stars("ml")


settings_tab(1, "ui_settings_models.png",
             "10b. SETTINGS -> Models  --  pick models, stars, sizes, soft voting", [
    ("models_tab.enable_box", "USE THE SELECTED MODELS -- off (default) = the "
     "original deployed classifier and net, exactly as before. On = the "
     "models ticked below."),
    ("models_tab.tables.ml", "THE ML ZOO, best first. \u2605 = top 3 of the "
     "family by accuracy per window with a person held out (ties by "
     "macro-F1); hover a star for the rank, score and size. Columns: score, "
     "size on disk, parameters, latency, what it was trained with."),
    ("models_tab.summaries.ml", "THE SELECTION -- how many, the total size on "
     "disk and parameters."),
    ("models_tab.vote_notes.ml", "SEVERAL MODELS = A SOFT VOTE (the mean of "
     "their probabilities). A selection that cannot be averaged -- different "
     "window, features, calibration or adaptation -- is flagged here and "
     "blocks APPLY, with the reason."),
    ("models_tab.tables.nn", "THE NEURAL NETS, the same columns. Nets are "
     "voted with nets only; ML and nets are never blended."),
], _demo_models)

settings_tab(2, "ui_settings_training.png",
             "10c. SETTINGS -> Training  --  augmentation and adaptation, per family", [
    ("training_tab.aug_boxes.NN.rotate", "AUGMENTATION, per family: gain, "
     "noise, time warp, band rotation, dead electrode. Applied to training "
     "windows only; a model already trained keeps what it learned."),
    ("training_tab.adapt_boxes.NN", "ADAPTATION TO THE PERSON: each person's "
     "input standardised with statistics from ~9 s of their own unlabelled "
     "signal (the calibration sweep)."),
    ("training_tab.adapt_live", "USE ADAPTATION LIVE -- off = adapted models "
     "use their stored global statistics (measured: no loss)."),
])

settings_tab(3, "ui_settings_general.png",
             "10d. SETTINGS -> General  --  the two questions", [
    ("general_tab.ask_calibration", "Ask whether to calibrate the hand before "
     "PREDICT / START NN."),
    ("general_tab.ask_retrain", "Ask whether to retrain (all / only the "
     "selected) when new gestures or recordings make models out of date."),
])

# ====================================================================== the questions
import time as _time
_offer = {"available": True, "age_h": 2.3, "created": "2026-10-27 08:10",
          "reason": "reuse it only if the band has stayed on your arm since "
                    "then; if you took it off or moved it, calibrate again",
          "rest_check": {"ok": True, "worst_channel": 0}}
_box, *_ = win._calibration_box("PREDICT", win.person_id, {"x": 1}, _offer)
_box.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
_box.show()
for _ in range(6):
    app.processEvents()
_box.grab().save(os.path.join(OUT, "ui_question_calibrate.png"))
_box.close()
print("  wrote ui_question_calibrate.png")
_box, *_ = win._retrain_box(["The ML classifier: 12 recordings were added "
                             "since it was trained",
                             "The neural net does not know WRIST_FLEX yet "
                             "(40 recordings of it exist)",
                             "5 model(s) in the model zoo were measured on an "
                             "older dataset"])
_box.setAttribute(QtCore.Qt.WA_DontShowOnScreen, True)
_box.show()
for _ in range(6):
    app.processEvents()
_box.grab().save(os.path.join(OUT, "ui_question_retrain.png"))
_box.close()
print("  wrote ui_question_retrain.png")

print("done")
sys.stdout.flush()
os._exit(0)          # skip Qt teardown: the compute probe outlives its label
