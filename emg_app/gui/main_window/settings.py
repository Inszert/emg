"""
The settings dialog (the gear icon in the header).

A mixin of MainWindow: every method runs on the window and uses its widgets
and state through `self`.
"""

from emg_app import config as cfg
from emg_app import settings_store
from emg_app.zoo import describe_selection
from emg_app.gui.dialogs.settings import SettingsDialog
from emg_app.gui.qt import QtCore, QtGui, QtWidgets


class SettingsMixin:
    """MainWindow -- Settings."""

    def _install_settings_shortcut(self) -> None:
        self._settings_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+,"), self)
        self._settings_shortcut.activated.connect(self.open_settings)

    def open_settings(self) -> None:
        """
        Settings -> Gestures: what is recorded, what the classifiers may
        answer with, and new gestures.

        While a recording session runs the plan is already built, so which
        gestures are RECORDED is locked (the same reason GESTURES... refuses);
        the classifier switches stay available, because they only mask the
        models' output and cannot disturb a recording.
        """
        dialog = SettingsDialog(self, model_labels=self._model_label_sets(),
                                selected=list(cfg.GESTURES),
                                classifier_off=list(cfg.CLASSIFIER_OFF),
                                defaults_note=self._defaults_note())
        if self.sequence.running:
            dialog.editor.lock_recording(
                "A session is running: finish or stop it to change what is "
                "recorded. The classifier switches still work.")
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        self._apply_gesture_setup(dialog.selected(), dialog.custom_gestures(),
                                  dialog.classifier_off())
        self._apply_app_settings(dialog.app_values())
        if dialog.retrain_requested:
            #  After the dialog has closed, so the confirmation the retrain
            #  asks for is not stacked on top of it.
            QtCore.QTimer.singleShot(0, self.retrain_everything)

    # -- models, training options, questions ------------------------------------

    def _defaults_note(self) -> str:
        """One line on where the defaults of the Training tab come from."""
        return ("Defaults in this build: augmentation "
                + (", ".join(settings_store.DEFAULTS["TRAIN_AUGMENT"])
                   or "none")
                + "; adaptation "
                + ("on" if settings_store.DEFAULTS["TRAIN_ADAPT"] else "off")
                + ".")

    def _apply_app_settings(self, values: dict) -> None:
        """
        Validate, apply and save the Models / Training / General settings.

        A change of model selection reloads the live classifier (cheap) and
        stops the neural net if it is running, because the worker holds the
        old checkpoints; START NN then loads the new ones.
        """
        try:
            clean = {name: settings_store._clean(name, value)
                     for name, value in values.items()
                     if name in settings_store.SETTINGS}
        except ValueError as exc:
            QtWidgets.QMessageBox.warning(self, "Settings", str(exc))
            return
        before = settings_store.current()
        settings_store.apply_settings(clean)
        try:
            settings_store.save_settings()
        except OSError as exc:
            self._append_log(f"Could not save the settings: {exc}")
        changed = [n for n in clean if clean[n] != before.get(n)]
        if not changed:
            return
        self._append_log("Settings changed: " + ", ".join(changed))
        selection = {"MODEL_SELECTION_ENABLED", "SELECTED_ML", "SELECTED_NN"}
        if selection & set(changed):
            self._reload_selected_models()
        if "ADAPT_LIVE" in changed and getattr(self, "nn_engine", None):
            self.nn_engine._send_adaptation()

    def _reload_selected_models(self) -> None:
        """The model selection moved: load what is selected, redraw."""
        if getattr(self, "prediction_active", False) or \
                getattr(self, "one_shot_deadline", 0.0):
            self.stop_prediction()
        self.predictor.load()
        if getattr(self, "nn_engine", None) is not None:
            if self.nn_engine.running:
                self.stop_neural()
        if hasattr(self, "prob_box"):
            self._rebuild_prob_bars()
        if getattr(self, "nn_engine", None) is not None:
            self._rebuild_nn_bars()
            self._refresh_neural_panel()
        self._refresh_prediction_panel()
        what = describe_selection("ml")
        what_nn = describe_selection("nn")
        self._append_log(
            "Models in use -- ML: "
            + (what or "the original deployed model")
            + "; neural net: " + (what_nn or "the original deployed net"))
