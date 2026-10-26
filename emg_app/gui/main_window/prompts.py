"""
The two questions the application asks, and the adaptation statistics.

A mixin of MainWindow: every method runs on the window and uses its widgets
and state through `self`. The DECISIONS (is the last calibration offered? what
does each retrain answer train?) live in `emg_app/prompts.py`, Qt-free and
tested without a window; this file only asks and acts.

1. `on_predict_clicked` / `on_nn_clicked` -- before PREDICT or START NN: "calibrate
   the hand first?", once per application run per person (a prompt on every
   press would be a nag). "Use the last calibration" is offered only when the
   application was restarted since it was made.
2. `update_adaptation` -- hands the person's statistics, computed from the
   signal stored with their calibration, to the models that were trained with
   adaptation.
3. `maybe_ask_retrain` -- when new gestures or recordings make models out of
   date: retrain all models, or only the selected ones.
"""

import os
import time

from emg_app import adaptation as AD
from emg_app import config as cfg
from emg_app import prompts as PR
from emg_app import settings_store, zoo
from emg_app.calibration import load_calibration
from emg_app.dataset import dataset_fingerprint, dataset_summary
from emg_app.gestures import model_gesture_gaps, recorded_gestures
from emg_app.gui.qt import QtCore, QtWidgets
from emg_app.neural import load_nn_stats
from emg_app.prediction import latest_model_window


class PromptsMixin:
    """MainWindow -- the questions it asks."""

    def _init_prompts(self) -> None:
        self._app_started_ts = time.time()
        self._calibration_gate_done = set()
        self._retrain_asked = False

    # ======================================================================
    # 1. calibrate the hand first?
    # ======================================================================

    def on_predict_clicked(self) -> None:
        """The PREDICT button: ask about calibration, then start or stop."""
        starting = not (self.prediction_active or self.one_shot_deadline)
        if starting and self.predictor.ready \
                and not self.calibration_gate("PREDICT"):
            return
        self.toggle_prediction()
        if self.prediction_active or self.one_shot_deadline:
            self.update_adaptation()

    def on_nn_clicked(self) -> None:
        """The START NN button: ask about calibration, then start or stop."""
        engine = self.nn_engine
        if engine is None:
            return
        if not engine.running and not self.calibration_gate("START NN"):
            return
        self.toggle_neural()
        if engine.running:
            self.update_adaptation()

    def calibration_gate(self, what: str) -> bool:
        """
        True to go on and start. The question is asked once per run of the
        application per person (and not at all if the setting is off, a
        calibration was just made, or calibration is disabled).
        """
        if not cfg.ASK_CALIBRATION_BEFORE_PREDICT or not cfg.CALIBRATION_ENABLED:
            return True
        person = self.person_id
        if person in self._calibration_gate_done:
            return True
        root = os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER)
        record = load_calibration(root, person)
        live = None
        try:
            live, _t = latest_model_window(self.acq, 1.0)
        except Exception:                                    # pragma: no cover
            live = None
        offer = PR.calibration_offer(record, self._app_started_ts,
                                     live_block=live)
        answer = self._ask_calibration_question(what, person, record, offer)
        if answer == "cancel":
            return False
        if answer == "calibrate":
            self.open_calibration()
            after = load_calibration(root, person)
            fresh = (after is not None and float(after.get("created_unix")
                                                 or 0) >= self._app_started_ts)
            if not fresh:
                self._append_log("Calibration was not saved -- not started.")
                return False
            self._calibration_gate_done.add(person)
            return True
        if answer == "reuse":
            self._append_log(
                f"Using the last calibration for {person} "
                f"({offer['created']}, {offer['age_h']:.1f} h old) -- the "
                "band was said to have stayed on the arm.")
        self._calibration_gate_done.add(person)
        self.update_adaptation()
        return True

    def _ask_calibration_question(self, what, person, record, offer) -> str:
        """`"calibrate" | "reuse" | "skip" | "cancel"`. Overridden by tests."""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Calibrate your hand?")
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setText(f"Calibrate {person}'s hand before {what}?")
        lines = ["The models were trained on a per-wearer calibrated scale. "
                 "A fresh calibration (about a minute, REST then each "
                 "gesture) puts this wearing of the band on that scale."]
        if record is None:
            lines.append("\n" + person + " has no stored calibration, so a "
                         "model that needs one will refuse to predict.")
        elif offer["available"]:
            lines.append(f"\nThe last calibration was made {offer['created']} "
                         f"({offer['age_h']:.1f} h ago). " + offer["reason"]
                         + ".")
            check = offer.get("rest_check")
            if check is not None:
                lines.append("The live signal at rest "
                             + ("looks like it did then."
                                if check["ok"] else
                                "does NOT look like it did then (channel "
                                f"{check['worst_channel'] + 1}) -- relax the "
                                "hand, or calibrate again."))
        else:
            lines.append("\nThe last calibration cannot be reused: "
                         + offer["reason"] + ".")
        box.setInformativeText("\n".join(lines))
        calibrate = box.addButton("CALIBRATE NOW",
                                  QtWidgets.QMessageBox.AcceptRole)
        reuse = None
        if offer["available"]:
            reuse = box.addButton("USE LAST CALIBRATION",
                                  QtWidgets.QMessageBox.AcceptRole)
            reuse.setToolTip("Only if the band has stayed on your arm since "
                             "the application was restarted.")
        skip = box.addButton("Skip", QtWidgets.QMessageBox.DestructiveRole)
        box.addButton(QtWidgets.QMessageBox.Cancel)
        box.setDefaultButton(reuse if reuse is not None else calibrate)
        box.exec()
        clicked = box.clickedButton()
        if clicked is calibrate:
            return "calibrate"
        if reuse is not None and clicked is reuse:
            return "reuse"
        if clicked is skip:
            return "skip"
        return "cancel"

    # ======================================================================
    # 2. adaptation statistics
    # ======================================================================

    def update_adaptation(self) -> None:
        """
        Give the models that were trained with adaptation the statistics of
        whoever wears the band, from the signal stored with their calibration.
        Models trained without adaptation are left alone; with
        `ADAPT_LIVE` off the models use their stored global statistics.
        """
        root = os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER)
        blocks, fs = AD.load_signal(root, self.person_id)
        usable = bool(blocks) and cfg.ADAPT_LIVE

        predictor = self.predictor
        if predictor.ready and predictor.adapts:
            stats = None
            if usable:
                setup = {"window_samples": predictor.window_samples,
                         "representation": predictor.representation,
                         "feature_set": predictor.feature_set,
                         "freq_features": predictor.freq_features,
                         "overlap": predictor.stats.get("overlap", 0.75)}
                stats = AD.ml_person_stats(blocks, fs, predictor.calibration,
                                           setup, model_fs=predictor.fs)
            if stats is not None:
                predictor.set_adaptation(*stats)
                self._append_log("Adaptation: the classifier is standardising "
                                 f"with {self.person_id}'s own statistics.")
            else:
                predictor.clear_adaptation()
                self._append_log(
                    "Adaptation: the classifier was trained with it, but "
                    + ("adaptation is switched off in the settings"
                       if not cfg.ADAPT_LIVE else
                       "no calibration signal is stored for "
                       f"{self.person_id} (calibrate once)")
                    + " -- using its stored global statistics.")

        engine = getattr(self, "nn_engine", None)
        if engine is not None and engine.running:
            sidecar = load_nn_stats(engine.active_model_path) or {}
            if (sidecar.get("adaptation") or {}).get("mode") == "person":
                win = int(engine.win_samples or sidecar.get("win_samples")
                          or 0)
                stats = None
                if usable and win > 0:
                    stats = AD.nn_person_stats(
                        blocks, fs, engine.calibration, win,
                        model_fs=engine.model_fs)
                if stats is not None:
                    engine.set_adaptation(*stats)
                    self._append_log("Adaptation: the neural net is "
                                     "standardising with "
                                     f"{self.person_id}'s own statistics.")
                else:
                    engine.clear_adaptation()
                    self._append_log("Adaptation: the neural net uses its "
                                     "stored global statistics.")

    # ======================================================================
    # 3. retrain all, or only the selected?
    # ======================================================================

    def retrain_state(self) -> dict:
        """What is out of date, as sentences, plus what could be retrained."""
        root = os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER)
        fingerprint = dataset_fingerprint(root)
        states = {}
        if self.predictor.ready:
            states["The ML classifier"] = self.predictor.data_status(
                fingerprint)
        if getattr(self, "nn_engine", None) is not None \
                and self.nn_engine.trained():
            states["The neural net"] = self.nn_engine.data_status(fingerprint)
        summary = dataset_summary(root)
        recs, people = recorded_gestures(summary)
        gaps = []
        if self.predictor.ready:
            for g in model_gesture_gaps(self.predictor.labels, recs, people):
                gaps.append(dict(g, model="The ML classifier"))
        if getattr(self, "nn_engine", None) is not None \
                and self.nn_engine.labels:
            for g in model_gesture_gaps(self.nn_engine.labels, recs, people):
                gaps.append(dict(g, model="The neural net"))
        reasons = PR.retrain_reasons(states, gaps)
        digest = str(fingerprint.get("uid_digest") or "")
        stale_zoo = [e for e in zoo.list_entries()
                     if e.meta.get("data_digest") != digest]
        if stale_zoo:
            reasons.append(f"{len(stale_zoo)} model(s) in the model zoo were "
                           "measured on an older dataset")
        return {"reasons": reasons, "digest": digest,
                "total_raw": summary.get("total_raw", 0)}

    def maybe_ask_retrain(self, reason: str = "startup",
                          force: bool = False) -> bool:
        """
        Ask "retrain all models or only the selected?" when something is out
        of date. Returns True when it asked. Never asks while a training or a
        recording session is running, and not at all with the setting off
        (`force` is for the settings dialog's own button).
        """
        if not force and not cfg.ASK_RETRAIN_ON_NEW_DATA:
            return False
        if self.sequence.running or self.training_process is not None \
                or self.nn_training_process is not None:
            return False
        state = self.retrain_state()
        if not state["reasons"] or state["total_raw"] < 12:
            return False
        choice, dont_ask = self._ask_retrain_question(state["reasons"], reason)
        if dont_ask:
            cfg.ASK_RETRAIN_ON_NEW_DATA = False
            try:
                settings_store.save_settings()
            except OSError:
                pass
            self._append_log("The retrain question is switched off "
                             "(settings, General tab, turns it back on).")
        self._retrain_asked = True
        targets = PR.retrain_targets(
            choice, cfg.MODEL_SELECTION_ENABLED, list(cfg.SELECTED_ML),
            list(cfg.SELECTED_NN), [e.id for e in zoo.list_entries()])
        if targets["deployed"]:
            self.retrain_everything(zoo_ids=targets["zoo_ids"] or None)
        elif targets["zoo_ids"]:
            self.retrain_zoo(targets["zoo_ids"])
        return True

    def _ask_retrain_question(self, reasons, why: str):
        """`(choice, dont_ask_again)`; choice is all | selected | none. Tests override."""
        box = QtWidgets.QMessageBox(self)
        box.setWindowTitle("Retrain?")
        box.setIcon(QtWidgets.QMessageBox.Question)
        box.setText("New gestures or recordings are not in the models yet.")
        selected = (cfg.MODEL_SELECTION_ENABLED
                    and (cfg.SELECTED_ML or cfg.SELECTED_NN))
        box.setInformativeText(
            "\n".join("  - " + r for r in reasons[:8])
            + "\n\nRetrain ALL models: the deployed classifier and net, then "
              "every model in the zoo (long).\n"
            + ("Retrain only the SELECTED: the models picked in the "
               "settings.\n" if selected else
               "Retrain only the SELECTED: with model selection off these "
               "are the deployed classifier and net.\n")
            + "Nothing is retrained automatically; a retrain runs in the "
              "background.")
        all_button = box.addButton("Retrain ALL models",
                                   QtWidgets.QMessageBox.AcceptRole)
        sel_button = box.addButton("Retrain only the selected",
                                   QtWidgets.QMessageBox.AcceptRole)
        box.addButton("Not now", QtWidgets.QMessageBox.RejectRole)
        check = QtWidgets.QCheckBox("Do not ask when the application opens")
        box.setCheckBox(check)
        box.setDefaultButton(sel_button)
        box.exec()
        clicked = box.clickedButton()
        choice = ("all" if clicked is all_button else
                  "selected" if clicked is sel_button else "none")
        return choice, check.isChecked()
