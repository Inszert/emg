"""
TEST MODEL.
"""

import math
import os
import time
from datetime import datetime
from typing import Optional, TYPE_CHECKING

import numpy as np

from emg_app import config as cfg
from emg_app.gui.tokens import (
    COL_ACCENT, COL_BAD, COL_GOOD, COL_TEXT, COL_TEXT_DIM, COL_WARN, FS_MD,
    FS_SM, FS_XL, H_ACTION)
from emg_app.utils import fmt_duration, safe_slug
from emg_app.sequence import STATE_COLORS, STATE_RECORDING
from emg_app.model_test import (
    MODEL_TEST_HOLD_S, MODEL_TEST_PREPARE_S, MODEL_TEST_ROUNDS,
    MODEL_TEST_SETTLE_S, ModelTestSession, load_model_tests, save_model_test)
from emg_app.prediction import latest_model_window
from emg_app.gui.qt import QtCore, QtWidgets
from emg_app.gui.widgets import fit_to_screen, make_label, scroll_dialog_body

if TYPE_CHECKING:                  # type hints only; avoids an import cycle
    from emg_app.gui.main_window import MainWindow


class ModelTestDialog(QtWidgets.QDialog):
    """
    TEST MODEL: cue each gesture, let the SAVED model predict live, count.

    Not modal, so the graph stays visible behind it. It drives the predictor
    itself (the panel's own PREDICT is stopped first so the two cannot share
    one smoothing history), and it never trains, saves or alters a model --
    it appends a result line to the wearer's model_tests.jsonl, nothing else.
    """

    def __init__(self, window: "MainWindow"):
        super().__init__(window)
        self.setWindowTitle("Test the model live")
        self.setModal(False)
        self.window_ = window
        self.predictor = window.predictor
        self.acq = window.acq
        self.session: Optional[ModelTestSession] = None
        self._last_predict = 0.0
        self.result = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        stats = self.predictor.stats or {}
        layout.addWidget(make_label(
            f"Live test -- {cfg.PERSON_ID} / {cfg.SESSION_ID}", FS_XL, True,
            COL_ACCENT))
        intro = make_label(
            f"Model: {stats.get('model_name', '?')}  "
            f"({', '.join(self.predictor.labels)})\n\n"
            "Each gesture is cued several times in a shuffled order. Get "
            "ready, then HOLD the gesture until the next cue. The first "
            f"{MODEL_TEST_SETTLE_S:.0f} s of each hold is not scored (your "
            "reaction and the onset). Nothing is recorded or trained -- only "
            "the score is kept.",
            FS_SM, False, COL_TEXT_DIM)
        intro.setWordWrap(True)
        layout.addWidget(intro)

        self.honesty = make_label("", FS_SM, True, COL_WARN)
        self.honesty.setWordWrap(True)
        layout.addWidget(self.honesty)
        self._check_honesty()

        row = QtWidgets.QHBoxLayout()
        row.addWidget(make_label("Rounds", FS_SM, False, COL_TEXT_DIM))
        self.rounds_spin = QtWidgets.QSpinBox()
        self.rounds_spin.setRange(1, 10)
        self.rounds_spin.setValue(MODEL_TEST_ROUNDS)
        self.rounds_spin.setToolTip("How many times each gesture is cued.")
        row.addWidget(self.rounds_spin)
        row.addSpacing(12)
        row.addWidget(make_label("Hold (s)", FS_SM, False, COL_TEXT_DIM))
        self.hold_spin = QtWidgets.QDoubleSpinBox()
        self.hold_spin.setRange(2.0, 8.0)
        self.hold_spin.setSingleStep(0.5)
        self.hold_spin.setValue(MODEL_TEST_HOLD_S)
        row.addWidget(self.hold_spin)
        row.addStretch(1)
        self.plan_label = make_label("", FS_SM, False, COL_TEXT_DIM)
        row.addWidget(self.plan_label)
        layout.addLayout(row)
        self.rounds_spin.valueChanged.connect(self._update_plan_label)
        self.hold_spin.valueChanged.connect(self._update_plan_label)
        self._update_plan_label()

        self.cue = make_label("press START", 26, True, COL_TEXT_DIM,
                              QtCore.Qt.AlignCenter)
        self.cue.setStyleSheet(
            f"color: {COL_TEXT_DIM}; background: transparent; "
            f"font-size: 30px; font-weight: 800;")
        layout.addWidget(self.cue)
        self.sub = make_label("", FS_MD, False, COL_TEXT_DIM,
                              QtCore.Qt.AlignCenter)
        layout.addWidget(self.sub)
        self.live = make_label("", FS_MD, True, COL_TEXT_DIM,
                               QtCore.Qt.AlignCenter)
        layout.addWidget(self.live)
        self.bar = QtWidgets.QProgressBar()
        self.bar.setRange(0, 1000)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(8)
        layout.addWidget(self.bar)

        self.results = QtWidgets.QTextBrowser()
        self.results.setMinimumHeight(220)
        layout.addWidget(self.results, 1)
        self._show_history()

        buttons = QtWidgets.QHBoxLayout()
        self.start_button = QtWidgets.QPushButton("START TEST")
        self.start_button.setObjectName("primary")
        self.start_button.setMinimumHeight(H_ACTION)
        self.start_button.clicked.connect(self.toggle)
        buttons.addWidget(self.start_button, 2)
        close = QtWidgets.QPushButton("CLOSE")
        close.setMinimumHeight(H_ACTION)
        close.clicked.connect(self.close)
        buttons.addWidget(close, 1)
        layout.addLayout(buttons)

        scroll_dialog_body(self, keep_last=1)
        fit_to_screen(self, 760, 860, min_width=480, min_height=420)

        self.timer = QtCore.QTimer(self)
        self.timer.setInterval(40)
        self.timer.timeout.connect(self._tick)

    # -- setup ----------------------------------------------------------------

    def _check_honesty(self) -> None:
        """Warn when this sitting is part of what the model trained on."""
        data = (self.predictor.payload_meta or {}).get("data") or {}
        sitting = f"{safe_slug(cfg.PERSON_ID, 'person')}/{cfg.SESSION_ID}"
        trained = set(data.get("sessions") or [])
        if sitting in trained:
            self.honesty.setText(
                f"{sitting} is IN this model's training data, so this score "
                f"will be optimistic. For an honest number: take the band "
                f"off, put it back on, and set a new Session id first.")
        elif safe_slug(cfg.PERSON_ID, "person") in set(data.get("subjects") or []):
            self.honesty.setText(
                "You are a wearer the model trained on, in a NEW sitting -- "
                "this is the 'known wearer' number.")
            self.honesty.setStyleSheet(
                f"color: {COL_GOOD}; background: transparent;")
        else:
            self.honesty.setText(
                "You are NOT in this model's training data -- this is the "
                "'new wearer' number.")
            self.honesty.setStyleSheet(
                f"color: {COL_GOOD}; background: transparent;")

    def _update_plan_label(self) -> None:
        n = self.rounds_spin.value() * len(self.predictor.labels)
        seconds = n * (MODEL_TEST_PREPARE_S + self.hold_spin.value())
        self.plan_label.setText(f"{n} trials, about {fmt_duration(seconds)}")

    def _streaming(self) -> bool:
        try:
            age = self.acq.stats().get("last_sample_age")
        except Exception:                                  # pragma: no cover
            return False
        return age is not None and age <= cfg.CALIBRATION_STALE_SECONDS

    # -- run ------------------------------------------------------------------

    def toggle(self) -> None:
        if self.session is not None and not self.session.finished:
            self.session.stop()
            self._finish(early=True)
            return
        if not self.predictor.ready:
            self.sub.setText(f"No usable model: {self.predictor.status}")
            return
        if not self._streaming():
            self.sub.setText("No samples arriving -- connect and wear the "
                             "armband first.")
            return
        self.window_.stop_prediction()
        self.session = ModelTestSession(
            None, self.predictor.labels, rounds=self.rounds_spin.value(),
            hold_s=self.hold_spin.value())
        self.session.start(time.perf_counter())
        self.predictor.reset()
        self.result = None
        self.rounds_spin.setEnabled(False)
        self.hold_spin.setEnabled(False)
        self.start_button.setText("STOP")
        self.start_button.setObjectName("danger")
        self.start_button.setStyleSheet("")
        self.timer.start()

    def _tick(self) -> None:
        s = self.session
        if s is None:
            return
        now = time.perf_counter()
        if s.tick(now):
            self.predictor.reset()       # one trial's history never votes
        if s.finished:
            self._finish()
            return
        if not self._streaming():
            s.stop()
            self._finish(early=True, why="the stream stopped")
            return

        total = len(s.plan)
        self.bar.setValue(int(1000 * (s.index + (
            0 if s.state == s.PREPARE else 0.5)) / max(1, total)))
        colour = STATE_COLORS.get(STATE_RECORDING, COL_GOOD)
        if s.state == s.PREPARE:
            self.cue.setText(f"GET READY:  {s.current}")
            self.cue.setStyleSheet(
                f"color: {COL_WARN}; background: transparent; "
                f"font-size: 30px; font-weight: 800;")
            self.sub.setText(f"trial {s.index + 1} of {total}  --  "
                             f"{cfg.GESTURE_CUES.get(s.current, '')}")
            self.live.setText("")
            return

        self.cue.setText(f"HOLD  {s.current}   {s.remaining(now):.1f} s")
        self.cue.setStyleSheet(
            f"color: {colour}; background: transparent; "
            f"font-size: 30px; font-weight: 800;")
        self.sub.setText(
            f"trial {s.index + 1} of {total}  --  "
            + ("scoring" if s.scoring(now) else "settling, not scored yet"))
        if (now - self._last_predict) * 1000.0 < cfg.PREDICTION_INTERVAL_MS:
            return
        self._last_predict = now
        emg, t = latest_model_window(self.acq,
                                     self.predictor.window_seconds())
        if emg.shape[0] < 4:
            return
        prediction = self.predictor.predict(emg, t)
        if prediction is None:
            s.stop()
            self._finish(early=True,
                         why=f"the model stopped: {self.predictor.status}")
            return
        if s.scoring(now):
            s.add(prediction.raw_label, prediction.label)
        right = prediction.label == s.current
        self.live.setText(f"the model says: {prediction.label} "
                          f"({prediction.percent})")
        self.live.setStyleSheet(
            f"color: {COL_GOOD if right else COL_BAD}; "
            f"background: transparent;")

    def _finish(self, early: bool = False, why: str = "") -> None:
        self.timer.stop()
        self.rounds_spin.setEnabled(True)
        self.hold_spin.setEnabled(True)
        self.start_button.setText("START TEST")
        self.start_button.setObjectName("primary")
        self.start_button.setStyleSheet("")
        self.cue.setText("done" if not early else "stopped")
        self.cue.setStyleSheet(
            f"color: {COL_TEXT_DIM}; background: transparent; "
            f"font-size: 30px; font-weight: 800;")
        self.live.setText("")
        self.bar.setValue(1000 if not early else self.bar.value())
        s = self.session
        summary = s.summary() if s is not None else None
        if not summary or not summary["windows"]:
            self.sub.setText("Nothing was scored" + (f": {why}" if why else
                                                     "."))
            return
        stats = self.predictor.stats or {}
        data = (self.predictor.payload_meta or {}).get("data") or {}
        self.result = {
            "unix_time": time.time(),
            "iso_time": datetime.now().isoformat(timespec="seconds"),
            "person_id": cfg.PERSON_ID,
            "session_id": cfg.SESSION_ID,
            "sitting_in_training_data":
                f"{safe_slug(cfg.PERSON_ID, 'person')}/{cfg.SESSION_ID}"
                in set(data.get("sessions") or []),
            "wearer_in_training_data":
                safe_slug(cfg.PERSON_ID, "person")
                in set(data.get("subjects") or []),
            "complete": not early,
            "stopped_because": why,
            "model": {"name": stats.get("model_name"),
                      "path": self.predictor.full_path,
                      "trained_newest": data.get("newest_stamp"),
                      "uid_digest": data.get("uid_digest"),
                      "calibration": ((self.predictor.payload_meta or {})
                                      .get("calibration") or {}).get("mode")
                      if isinstance((self.predictor.payload_meta or {})
                                    .get("calibration"), dict) else None,
                      "cv_window_accuracy": stats.get("window_accuracy"),
                      "cv_recording_vote": stats.get(
                          "recording_vote_accuracy")},
            **summary,
        }
        try:
            path = save_model_test(os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER),
                                   cfg.PERSON_ID, self.result)
            saved = f"saved to {path}"
        except OSError as exc:
            saved = f"NOT saved: {exc}"
        self.sub.setText(("stopped early -- partial result, " if early else "")
                         + saved)
        self.results.setHtml(self._result_html(self.result))
        self.window_._append_log(
            f"Model test: {summary['per_trial']['accuracy'] * 100:.0f} % per "
            f"trial, {summary['smoothed']['accuracy'] * 100:.0f} % smoothed, "
            f"{summary['per_window']['accuracy'] * 100:.0f} % per window "
            f"({summary['trials_scored']} trials).")

    # -- output ---------------------------------------------------------------

    @staticmethod
    def _pct(value) -> str:
        return ("--" if value is None or (isinstance(value, float)
                                          and not math.isfinite(value))
                else f"{value * 100:.1f} %")

    def _result_html(self, r: dict) -> str:
        labels = r["labels"]
        rows = [("per trial (one answer per hold)", r["per_trial"]),
                ("smoothed (what the panel shows)", r["smoothed"]),
                ("per window (raw, comparable with CV)", r["per_window"])]
        cell = "padding:2px 10px"
        html = [f'<div style="font-family:Segoe UI;color:{COL_TEXT}">']
        html.append('<table style="border-collapse:collapse">'
                    f'<tr><th style="{cell};text-align:left">score</th>'
                    f'<th style="{cell}">accuracy</th>'
                    f'<th style="{cell}">macro F1</th>'
                    f'<th style="{cell}">n</th></tr>')
        for name, m in rows:
            html.append(f'<tr><td style="{cell}">{name}</td>'
                        f'<td style="{cell}"><b>{self._pct(m["accuracy"])}'
                        f'</b></td><td style="{cell}">'
                        f'{self._pct(m["macro_f1"])}</td>'
                        f'<td style="{cell}">{m["n"]}</td></tr>')
        html.append("</table>")
        cv = r["model"].get("cv_window_accuracy")
        if cv is not None:
            html.append(
                f'<p style="color:{COL_TEXT_DIM}">For comparison, the '
                f"trainer's cross-validated score: {self._pct(cv)} per "
                f"window, {self._pct(r['model'].get('cv_recording_vote'))} "
                f"per recording -- on wearers it had NOT seen.</p>")
        #  The confusion matrix as ROW percentages: each row is one true
        #  gesture and sums to 100 %, so a rare gesture reads as clearly as a
        #  common one.
        m = np.asarray(r["confusion_smoothed"], dtype=float)
        html.append('<p><b>Confusion, smoothed answers</b> '
                    f'<span style="color:{COL_TEXT_DIM}">(rows = what you '
                    'did, columns = what it said; % of each row)</span></p>')
        html.append('<table style="border-collapse:collapse">')
        html.append(f'<tr><td style="{cell}"></td>' + "".join(
            f'<td style="{cell};color:{COL_TEXT_DIM}">{l[:8]}</td>'
            for l in labels) + "</tr>")
        for i, name in enumerate(labels):
            total = m[i].sum()
            html.append(f'<tr><td style="{cell}">{name}</td>')
            for j in range(len(labels)):
                share = m[i, j] / total if total else 0.0
                colour = (COL_GOOD if i == j else COL_BAD) if share >= 0.05 \
                    else COL_TEXT_DIM
                html.append(f'<td style="{cell};color:{colour};'
                            f'text-align:right">'
                            + (f"{share * 100:.0f}" if total else "")
                            + "</td>")
            html.append("</tr>")
        html.append("</table>")
        html.append(self._history_html())
        html.append("</div>")
        return "".join(html)

    def _history_html(self) -> str:
        tests = load_model_tests(os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER),
                                 cfg.PERSON_ID)[-8:]
        if not tests:
            return ""
        out = [f'<p><b>Earlier tests for {cfg.PERSON_ID}</b></p><table>']
        for t in reversed(tests):
            trial = (t.get("per_trial") or {}).get("accuracy")
            smooth = (t.get("smoothed") or {}).get("accuracy")
            tag = ("in training data" if t.get("sitting_in_training_data")
                   else "new sitting")
            out.append(
                f'<tr><td style="padding:1px 10px">{t.get("iso_time", "?")}'
                f'</td><td style="padding:1px 10px">'
                f'{(t.get("model") or {}).get("name", "?")}</td>'
                f'<td style="padding:1px 10px">{self._pct(trial)} trial</td>'
                f'<td style="padding:1px 10px">{self._pct(smooth)} smoothed'
                f'</td><td style="padding:1px 10px;color:{COL_TEXT_DIM}">'
                f'{tag}</td></tr>')
        out.append("</table>")
        return "".join(out)

    def _show_history(self) -> None:
        history = self._history_html()
        self.results.setHtml(
            f'<div style="font-family:Segoe UI;color:{COL_TEXT}">'
            + (history or f'<p style="color:{COL_TEXT_DIM}">No tests yet '
               f"for {cfg.PERSON_ID}.</p>") + "</div>")

    def closeEvent(self, event) -> None:
        if self.session is not None and not self.session.finished:
            self.session.stop()
            self._finish(early=True, why="the dialog was closed")
        self.timer.stop()
        if getattr(self.window_, "model_test_dialog", None) is self:
            self.window_.model_test_dialog = None
        super().closeEvent(event)
