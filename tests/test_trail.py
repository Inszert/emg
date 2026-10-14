"""
The DECISION TRAIL: every candidate's vote, the fusion, and the honesty rules.

What this is really protecting:

  * the panel is a SIDE file -- the live classifier still loads one estimator
    from gesture_model.joblib and never touches the panel;
  * the live trail runs on the feature vector LivePredictor already built, so
    there is exactly one feature path, not two that can drift;
  * soft voting really is the mean of the members, checked numerically rather
    than asserted in a comment;
  * a panel that does not match the loaded model is REFUSED, not fed vectors
    it was never fitted on;
  * with no panel the view explains itself and shows no numbers at all;
  * the forests predict on one thread, which is the difference between 50 ms
    and 2.7 s per live window.

Most of it runs on a synthetic dataset so the test is self-contained. Where
the real models/gesture_panel.joblib exists, it is additionally checked.

Run from the project root:

    python tests\\test_trail.py
"""
import os
import sys
import tempfile
import shutil

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import emg_dataset_tool as T                                      # noqa: E402

OK = []


def check(name, condition, detail=""):
    OK.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail else ""))


def section(title):
    print(f"\n--- {title} ---")


# ==============================================================================
#  A synthetic panel, built through the trainer's own code
# ==============================================================================
section("building a panel from synthetic data")

try:
    import train_model as TM
    import joblib
except ImportError as exc:                                   # pragma: no cover
    print(f"scikit-learn and joblib are required: {exc}")
    sys.exit(1)

LABELS = ["FIST", "REST"]
FEATURES = T.model_feature_names("level", True)
N_FEATURES = len(FEATURES)

#  Two separable classes with per-channel levels in the first eight columns,
#  so the stored channel strip is real data and not a placeholder. One
#  deliberately weak FIST recording keeps the problem from being trivial, so
#  the candidates have something to disagree about.
rng = np.random.default_rng(4)
rows, ys, groups = [], [], []
for rec in range(24):
    label = rec % 2                                   # 0 = FIST, 1 = REST
    uid = f"rec{rec:03d}"
    weak = (label == 0 and rec in (2, 6))
    base = 40.0 if label else (130.0 if weak else 420.0)
    for _ in range(9):
        row = rng.normal(base, 26.0, N_FEATURES)
        row[:8] = np.abs(rng.normal(base, 30.0, 8))
        rows.append(row)
        ys.append(label)
        groups.append(uid)

X = np.asarray(rows, dtype=np.float32)
y = np.asarray(ys, dtype=int)
groups_arr = np.asarray(groups, dtype=object)

SPEC = {
    "version": T.MODEL_SPEC_VERSION,
    "models": [
        {"name": "LDA-shrinkage", "kind": "lda", "enabled": True,
         "scaler": "standard",
         "params": {"solver": "lsqr", "shrinkage": "auto"}},
        {"name": "LogReg", "kind": "logreg", "enabled": True,
         "scaler": "standard", "params": {"C": 1.0, "max_iter": 500}},
        {"name": "RandomForest-50", "kind": "random_forest", "enabled": True,
         "scaler": "none", "params": {"n_estimators": 50}},
    ],
    "voting": {"enabled": True, "mode": "soft", "always": False},
}
BEST = {"model": "LDA-shrinkage", "window_s": 0.2, "overlap": 0.5,
        "representation": "raw", "feature_set": "level",
        "freq_features": True, "exclude": "all"}
META = {"window_samples": 12, "step": 6}

panel_dict = TM.build_decision_panel(SPEC, False, X, y, groups_arr, LABELS,
                                     BEST, META)
check("build_decision_panel returns a panel", panel_dict is not None)
check("it is tagged as a decision panel",
      panel_dict.get("kind") == "decision_panel")
check("the three members are there, in spec order",
      panel_dict["member_order"] == ["LDA-shrinkage", "LogReg",
                                     "RandomForest-50"],
      str(panel_dict["member_order"]))
check("the ensemble is named after its member count",
      panel_dict["vote_name"] == "Vote-soft-3", str(panel_dict["vote_name"]))
check("the deployed winner is recorded",
      panel_dict["winner"] == "LDA-shrinkage")
check("every member is fitted and stored",
      all(n in panel_dict["members"] for n in panel_dict["member_order"]))
check("the channel strip was captured by name",
      panel_dict["trail"]["chan"] is not None
      and panel_dict["trail"]["chan"].shape == (len(y), 8),
      str(None if panel_dict["trail"]["chan"] is None
          else panel_dict["trail"]["chan"].shape))

section("the fusion rule is measured, not asserted")
check("the panel records the fusion rule",
      "mean" in panel_dict["fusion"]["rule"])
check("soft voting equals the mean of the members",
      panel_dict["fusion"]["verified"] is True,
      f"max|diff| = {panel_dict['fusion']['max_abs_diff_vs_mean']:.2e}")

#  And the same thing checked here, independently of what the trainer stored.
proba = panel_dict["trail"]["proba"]
members_mean = np.mean([np.asarray(proba[n])
                        for n in panel_dict["member_order"]], axis=0)
worst = float(np.max(np.abs(members_mean
                            - np.asarray(proba[panel_dict["vote_name"]]))))
check("recomputed here as well", worst < 1e-5, f"max|diff| = {worst:.2e}")

section("the trail is out-of-fold")
folds = np.asarray(panel_dict["trail"]["fold"])
check("every window was held out in some fold", (folds >= 0).all())
check("there are several folds", len(set(folds.tolist())) >= 2,
      f"{len(set(folds.tolist()))} folds")
#  The property that makes the numbers mean anything: a recording never
#  straddles a split, so no window is scored by a model that trained on its
#  own recording.
straddle = [uid for uid in set(groups)
            if len(set(folds[np.flatnonzero(groups_arr == uid)].tolist())) > 1]
check("no recording is split across folds", not straddle, str(straddle[:3]))

section("the scores are the winner's setup, not each model's best")
for name, stats in panel_dict["cv"].items():
    ok = 0.0 <= stats["window_accuracy"] <= 1.0
    if not ok:
        check(f"{name} accuracy is a probability", False,
              str(stats["window_accuracy"]))
        break
else:
    check("every candidate has a real out-of-fold score", True,
          ", ".join(f"{n} {s['window_accuracy'] * 100:.0f}%"
                    for n, s in panel_dict["cv"].items()))
check("the ensemble is scored like any other candidate",
      panel_dict["vote_name"] in panel_dict["cv"])
check("per-recording votes are recorded for every candidate",
      all(set(r["models"]) == set(panel_dict["cv"])
          for r in panel_dict["recordings"]))

# ==============================================================================
#  Loading it back
# ==============================================================================
section("loading the panel")

TEMP = tempfile.mkdtemp(prefix="emgtrail_")
PANEL_PATH = os.path.join(TEMP, T.PANEL_FILENAME)
joblib.dump(panel_dict, PANEL_PATH)

panel = T.DecisionPanel(path=PANEL_PATH)
check("it loads", panel.load(), panel.status)
check("it reports itself ready", panel.ready)
check("the labels survive", panel.labels == LABELS)
check("the trail survives", panel.trail_length() == len(y),
      f"{panel.trail_length()} of {len(y)}")
check("the recording list survives",
      panel.n_recordings == 24 and len(panel.recordings) == 24,
      str(panel.n_recordings))

section("the forests are forced onto one thread")
#  This is a 50x difference per live window, measured: with n_jobs=-1 the
#  seven real candidates took 2.7 s for one row because joblib stood up a
#  worker pool per call. Pinned here so it cannot quietly come back.
offenders = []


def walk(estimator, depth=0):
    if estimator is None or depth > 4:
        return
    if getattr(estimator, "n_jobs", None) not in (None, 1):
        offenders.append((type(estimator).__name__, estimator.n_jobs))
    for step in (getattr(estimator, "named_steps", None) or {}).values():
        walk(step, depth + 1)
    for member in getattr(estimator, "estimators_", None) or []:
        walk(member, depth + 1)
    for entry in getattr(estimator, "estimators", None) or []:
        walk(entry[1] if isinstance(entry, tuple) else entry, depth + 1)


for model in panel.members.values():
    walk(model)
check("no loaded estimator predicts on more than one thread",
      not offenders, str(offenders[:3]))

section("warm-up")
check("it starts cold", not panel.warm)
spent = panel.warm_up()
check("warming up marks the panel warm", panel.warm and not panel.warming,
      f"{spent:.0f} ms")
check("a second warm-up is a no-op", panel.warm_up() == 0.0)

# ==============================================================================
#  A real model, a real feature vector, one feature path
# ==============================================================================
section("the live path uses the predictor's own feature vector")

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # noqa: E402
from sklearn.pipeline import Pipeline                              # noqa: E402
from sklearn.preprocessing import StandardScaler                   # noqa: E402

MODEL_PATH = os.path.join(TEMP, "gesture_model.joblib")
estimator = Pipeline([("scale", StandardScaler()),
                      ("clf", LinearDiscriminantAnalysis(solver="lsqr",
                                                         shrinkage="auto"))])
estimator.fit(X, y)
joblib.dump({
    "model": estimator, "labels": LABELS, "fs": 60.0, "window_s": 0.2,
    "window_samples": 12, "representation": "raw", "feature_set": "level",
    "freq_features": True, "feature_names": FEATURES,
    "emg_mode": "filtered", "created_iso": T.iso_time(0),
    "stats": {"model_name": "LDA-shrinkage", "labels": LABELS,
              "overlap": 0.5, "excluded": "all"},
}, MODEL_PATH)

predictor = T.LivePredictor(MODEL_PATH)
check("the model loads", predictor.load(), predictor.status)
check("the panel matches it", panel.matches(predictor),
      panel.stale_reason(predictor))

n = 21
times = np.arange(n) / 47.0
emg = np.abs(rng.normal(430.0, 40.0, (n, 8)))
prediction = predictor.predict(emg, times)
check("a real prediction comes out", prediction is not None)
check("the predictor kept the vector it used",
      predictor.last_features is not None
      and predictor.last_features.shape == (1, N_FEATURES),
      str(None if predictor.last_features is None
          else predictor.last_features.shape))

#  The vector the panel runs on must be the SAME OBJECT the deployed model
#  was given -- not an equal one rebuilt here, which is what would let the
#  two feature paths drift apart.
features = predictor.last_features
step = panel.step_for_features(features, predictor)
check("the panel answers on that vector", step is not None)
check("it ran every member", step is not None and len(step.names) == 3,
      str(None if step is None else step.names))
check("it did not alter the predictor's vector",
      np.array_equal(features, predictor.last_features))

independent = T.model_feature_vector(
    T.resample_uniform(emg, times, predictor.window_samples),
    predictor.representation, predictor.feature_set,
    predictor.freq_features, predictor.fs)[None, :]
check("that vector is what model_feature_vector() produces",
      np.array_equal(independent, features))

section("the live step reports the deployed model, not the vote")
check("the deployed name is the saved winner",
      step.deployed_name == "LDA-shrinkage", step.deployed_name)
check("the deployed probabilities are one of the members",
      step.deployed is not None
      and step.deployed == step.per_model[step.names.index("LDA-shrinkage")])
deployed_live = predictor.last.raw_probabilities
check("the deployed member agrees with the live classifier itself",
      np.allclose(step.deployed, deployed_live, atol=1e-6),
      f"{np.round(step.deployed, 4)} vs {np.round(deployed_live, 4)}")

section("the live fusion is the mean, and it is not the ensemble re-run")
check("the live step does not call the stored ensemble",
      step.fused is None)
check("its fused value is the mean of the members",
      np.allclose(step.fused_from_mean,
                  np.mean(np.asarray(step.per_model), axis=0), atol=1e-12))
check("agreement counts the members that match the vote",
      step.agreement == sum(
          1 for p in step.per_model
          if int(np.argmax(p)) == int(np.argmax(step.fused_from_mean))),
      f"{step.agreement}/{len(step.names)}")

section("the recorded step keeps the ensemble's own output")
recorded = panel.recorded_step(0)
check("a recorded window reads back", recorded is not None)
check("it carries the ensemble's stored probabilities",
      recorded.fused is not None)
check("which equal the mean of its members",
      np.allclose(recorded.fused, recorded.fused_from_mean, atol=1e-5),
      f"{np.round(recorded.fused, 5)} vs "
      f"{np.round(recorded.fused_from_mean, 5)}")
check("it says which fold it was held out in",
      "fold" in recorded.note, recorded.note)
check("it is labelled as recorded, not live", recorded.source == "recorded")
check("the stored channel levels come back",
      panel.recorded_channels(0) is not None
      and panel.recorded_channels(0).shape == (8,))
check("out-of-range rows return nothing, rather than wrapping",
      panel.recorded_step(panel.trail_length()) is None
      and panel.recorded_step(-1) is None)

section("disagreements")
splits = panel.split_windows()
check("split windows are found on a dataset with a weak class",
      len(splits) > 0, f"{len(splits)} of {panel.trail_length()}")
if splits:
    row = splits[0]
    s = panel.recorded_step(row)
    picks = {s.label_of(p) for p in s.per_model}
    check("a split window really has the candidates disagreeing",
          len(picks) > 1, str(sorted(picks)))
    check("its recording index is reported",
          0 <= panel.recording_index_of(row) < panel.n_recordings)
unanimous = [r for r in range(panel.trail_length()) if r not in set(splits)]
if unanimous:
    s = panel.recorded_step(unanimous[0])
    check("a non-split window has them unanimous",
          len({s.label_of(p) for p in s.per_model}) == 1)

# ==============================================================================
#  Refusals -- the part that keeps it honest
# ==============================================================================
section("a mismatched panel is refused, never fed")

mismatch = dict(panel_dict)
mismatch["setup"] = dict(panel_dict["setup"], feature_set="classic")
MISMATCH_PATH = os.path.join(TEMP, "mismatch.joblib")
joblib.dump(mismatch, MISMATCH_PATH)
wrong = T.DecisionPanel(path=MISMATCH_PATH)
wrong.load()
check("it loads but does not match the model", wrong.ready
      and not wrong.matches(predictor))
check("and says why in words", "feature set" in wrong.stale_reason(predictor),
      wrong.stale_reason(predictor))

shortened = dict(panel_dict)
shortened["setup"] = dict(panel_dict["setup"], n_features=7)
SHORT_PATH = os.path.join(TEMP, "short.joblib")
joblib.dump(shortened, SHORT_PATH)
short = T.DecisionPanel(path=SHORT_PATH)
short.load()
check("a panel expecting a different feature count returns nothing",
      short.step_for_features(features, predictor) is None)
check("and explains itself", "features" in short.status, short.status)

section("a missing or wrong file is a state, not a crash")
absent = T.DecisionPanel(path=os.path.join(TEMP, "nope.joblib"))
check("a missing panel loads as False", absent.load() is False)
check("with a reason that names the fix", "retrain" in absent.status.lower(),
      absent.status)
check("and answers nothing", absent.step_for_features(features) is None
      and absent.recorded_step(0) is None and not absent.has_trail)

NOT_PANEL = os.path.join(TEMP, "notapanel.joblib")
joblib.dump({"model": estimator, "labels": LABELS}, NOT_PANEL)
imposter = T.DecisionPanel(path=NOT_PANEL)
check("a model file is not accepted as a panel", imposter.load() is False)
check("and is named as such", "not a decision panel" in imposter.status,
      imposter.status)

section("the panel never becomes the classifier")
#  Not an identity check: the payload is unpickled, so it is a copy by
#  construction. What matters is that the predictor holds ONE estimator of
#  the trained kind -- never the panel's member dict, and never the ensemble.
check("the live predictor holds exactly one estimator",
      type(predictor.model) is type(estimator)
      and not isinstance(predictor.model, dict),
      type(predictor.model).__name__)
check("and it is not the voting ensemble",
      type(predictor.model).__name__ != "VotingClassifier"
      and predictor.model is not panel.members.get(panel.vote_name))
payload = joblib.load(MODEL_PATH)
check("the model file holds one estimator and no members",
      "model" in payload and "members" not in payload
      and "trail" not in payload)
check("panel_path() points beside the model, not at it",
      T.panel_path(MODEL_PATH) == PANEL_PATH
      and T.panel_path(MODEL_PATH) != MODEL_PATH)

# ==============================================================================
#  The GUI
# ==============================================================================
section("the view")

from PySide6 import QtWidgets, QtGui, QtCore                       # noqa: E402

app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

dialog = T.ModelStatsDialog(predictor, None, panel=panel)
check("MODEL STATS builds with a panel", dialog.trail_view is not None)
view = dialog.trail_view
dialog.show()
app.processEvents()
check("the controls are shown when the panel is usable",
      view.controls.isVisible())
check("every recording is offered", view.rec_select.count() == 24,
      str(view.rec_select.count()))
check("the footer states the protocol",
      "StratifiedGroupKFold" in view.footer.text(), view.footer.text()[:60])
check("the footer states the fusion check",
      "max difference" in view.footer.text())
tables = [t for t in dialog.findChildren(QtWidgets.QTableWidget)
          if t.horizontalHeaderItem(0) is not None
          and t.horizontalHeaderItem(0).text() == "candidate"]
check("the candidate table lists every candidate and the ensemble",
      len(tables) == 1 and tables[0].rowCount() == 4,
      f"{len(tables)} table(s), "
      f"{tables[0].rowCount() if tables else 0} rows")
if tables:
    names = [tables[0].item(r, 0).text() for r in range(tables[0].rowCount())]
    check("the deployed model is marked in it",
          any("deployed" in n for n in names), str(names))
    check("the ensemble is marked as such",
          any("ensemble" in n for n in names), str(names))


def painted(widget):
    """Render the widget and count pixels that are not the background."""
    pixmap = QtGui.QPixmap(widget.size())
    pixmap.fill(QtGui.QColor("#000000"))
    widget.render(pixmap)
    image = pixmap.toImage()
    return sum(1 for x in range(0, image.width(), 6)
               for y in range(0, image.height(), 6)
               if image.pixelColor(x, y).name() != "#000000")


view._set_live(True)
view.refresh_live(force=True)
app.processEvents()
check("live mode shows the live step",
      view.trail.step is not None and view.trail.step.source == "live")
check("it draws something", painted(view.trail) > 200,
      str(painted(view.trail)))
check("the channel strip is drawn from the live window",
      view.trail.channels is not None)

view._set_live(False)
app.processEvents()
check("recorded mode switches source",
      view.trail.step is not None and view.trail.step.source == "recorded")
check("the window counter reads position / total",
      "/" in view.position.text(), view.position.text())
check("recorded mode enables the scrubber",
      view.rec_select.isEnabled() and view.slider.isEnabled())
check("live mode disables it",
      (view._set_live(True), not view.rec_select.isEnabled())[1])

view._set_live(False)
first_row = view.row
view._next_split()
app.processEvents()
check("NEXT DISAGREEMENT moves to a split window",
      view.row in set(splits), f"row {view.row}")
check("and the drawing follows it", painted(view.trail) > 200)
check("stepping again does not get stuck",
      (view._next_split(), view.row in set(splits))[1], f"row {view.row}")

view.slider.setValue(min(3, view.slider.maximum()))
app.processEvents()
check("the slider selects a window of the chosen recording",
      view.row in view._rows, f"row {view.row}")

section("the view refuses rather than invents")
blank_dialog = T.ModelStatsDialog(predictor, None, panel=absent)
blank = blank_dialog.trail_view
check("with no panel the controls are hidden",
      not blank.controls.isVisible())
check("the drawing says why", blank.trail.step is None
      and "No decision panel" in blank.trail.empty_reason)
check("it still paints a readable message", painted(blank.trail) > 20)


def candidate_tables(dialog):
    """Tables whose first column header is the candidate list's."""
    found = []
    for table in dialog.findChildren(QtWidgets.QTableWidget):
        item = table.horizontalHeaderItem(0)
        if item is not None and item.text() == "candidate":
            found.append(table)
    return found


check("no candidate table is added without a panel",
      not candidate_tables(blank_dialog),
      f"{len(candidate_tables(blank_dialog))} found")
blank_dialog.close()

stale_dialog = T.ModelStatsDialog(predictor, None, panel=wrong)
stale = stale_dialog.trail_view
check("a mismatched panel is not drawn", stale.trail.step is None)
check("and the reason is the mismatch",
      "does not match" in stale.trail.empty_reason, stale.trail.empty_reason[:50])
stale_dialog.close()

section("narrow layouts")
#  A standalone widget, NOT inside the dialog's layout: a resize() on a laid
#  out child is undone on the next layout pass, so testing it there would
#  measure the same width four times.
loose = T.DecisionTrailWidget()
loose.set_step(panel.recorded_step(splits[0] if splits else 0),
               channels=panel.recorded_channels(0), caption="x" * 180,
               cv=panel.cv)
for width, height in ((360, 300), (520, 340), (900, 420), (1600, 500)):
    loose.resize(width, height)
    try:
        drawn = painted(loose)
        ok = drawn > 50 and loose.width() == width
    except Exception as exc:                                 # pragma: no cover
        ok, drawn = False, f"{type(exc).__name__}: {exc}"
    check(f"it paints at {width}x{height}", ok, str(drawn))

#  And with no channel strip at all, which is what a panel built from a
#  feature set without per-channel levels looks like.
loose.set_step(panel.recorded_step(0), channels=None, caption="")
loose.resize(420, 300)
check("it paints with no channel strip", painted(loose) > 50)

dialog.close()

# ==============================================================================
#  The real panel, where one exists
# ==============================================================================
section("the real models/gesture_panel.joblib")

real_path = T.panel_path()
if not os.path.exists(real_path):
    print("  (no real panel on disk; skipped. Build one with "
          "python explain_votes.py)")
else:
    real = T.DecisionPanel()
    check("it loads", real.load(), real.status)
    live = T.LivePredictor()
    live.load()
    check("it matches the deployed model", real.matches(live),
          real.stale_reason(live))
    check("the deployed winner is a member, and the vote is not it",
          real.winner in real.names and real.winner != real.vote_name,
          f"{real.winner} vs {real.vote_name}")
    check("its fusion check passed", real.fusion.get("verified") is True,
          f"max|diff| = {real.fusion.get('max_abs_diff_vs_mean')}")
    check("it carries a per-window trail",
          real.has_trail and real.trail_length() > 0,
          str(real.trail_length()))
    worst_real = 0.0
    for row in range(0, real.trail_length(),
                     max(1, real.trail_length() // 40)):
        s = real.recorded_step(row)
        if s and s.fused is not None:
            worst_real = max(worst_real,
                             float(np.max(np.abs(np.asarray(s.fused)
                                                 - s.fused_from_mean))))
    check("the vote equals the mean across the real dataset too",
          worst_real < 1e-5, f"max|diff| = {worst_real:.2e}")

    real.warm_up()
    n = 19
    times = np.arange(n) / 50.0
    emg = np.abs(np.random.default_rng(1).normal(400.0, 50.0, (n, 8)))
    live.predict(emg, times)
    real_step = real.step_for_features(live.last_features, live)
    check("it answers on a real live window", real_step is not None)
    if real_step is not None:
        check("with one opinion per candidate in the spec",
              len(real_step.names) == len(real.names),
              f"{len(real_step.names)} of {len(real.names)}")
        check("each one a real probability distribution",
              all(abs(sum(p) - 1.0) < 1e-6 for p in real_step.per_model))

    import time as _time
    real.step_for_features(live.last_features, live)
    started = _time.perf_counter()
    for _ in range(5):
        real.step_for_features(live.last_features, live)
    per_window = (_time.perf_counter() - started) / 5 * 1000
    #  The live view refreshes once per prediction (150 ms by default). This
    #  is the measurement that justifies running it on the GUI thread at all.
    check("a live window costs well under the prediction interval",
          per_window < T.PREDICTION_INTERVAL_MS * 0.75,
          f"{per_window:.0f} ms vs {T.PREDICTION_INTERVAL_MS} ms")

shutil.rmtree(TEMP, ignore_errors=True)

print(f"\n================  {sum(OK)}/{len(OK)} checks passed  ================")
sys.exit(0 if all(OK) else 1)
