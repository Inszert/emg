"""
The settings dialog (the gear icon in the header).

A mixin of MainWindow: every method runs on the window and uses its widgets
and state through `self`.
"""

from emg_app import config as cfg
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
                                classifier_off=list(cfg.CLASSIFIER_OFF))
        if self.sequence.running:
            dialog.editor.lock_recording(
                "A session is running: finish or stop it to change what is "
                "recorded. The classifier switches still work.")
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        self._apply_gesture_setup(dialog.selected(), dialog.custom_gestures(),
                                  dialog.classifier_off())
        if dialog.retrain_requested:
            #  After the dialog has closed, so the confirmation the retrain
            #  asks for is not stacked on top of it.
            QtCore.QTimer.singleShot(0, self.retrain_everything)
