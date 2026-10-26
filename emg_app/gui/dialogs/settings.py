"""
Settings (the gear icon).

One tab so far -- Gestures -- built as a tab widget so the next setting has
somewhere to go. The gestures tab is the same `GestureEditor` the GESTURES...
dialog uses: which gestures are recorded, which the live classifiers may
answer with, and adding new ones.
"""

from emg_app import config as cfg
from emg_app.gui.dialogs.gesture_setup import GestureEditor
from emg_app.gui.qt import QtWidgets
from emg_app.gui.tokens import COL_ACCENT, COL_TEXT_DIM
from emg_app.gui.widgets import fit_to_screen, make_label, wrap_label


class SettingsDialog(QtWidgets.QDialog):
    """
    Apply with APPLY, or with APPLY + RETRAIN when a recorded gesture is
    missing from a trained model.

    Nothing is written by the dialog itself: the caller reads
    `selected()`, `custom_gestures()` and `classifier_off()`, applies them
    and saves. `retrain_requested` says the operator asked for the retrain
    chain as well -- the caller still shows the retrain's own confirmation.
    """

    def __init__(self, parent=None, model_labels=None, selected=None,
                 classifier_off=None):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        fit_to_screen(self, 700, 780, min_width=460, min_height=420)
        self.retrain_requested = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addWidget(make_label("Settings", 15, True, COL_ACCENT))

        self.tabs = QtWidgets.QTabWidget()
        self.editor = GestureEditor(self, selected, model_labels,
                                    classifier_off)
        page = QtWidgets.QWidget()
        page_layout = QtWidgets.QVBoxLayout(page)
        page_layout.setContentsMargins(8, 8, 8, 8)
        page_layout.addWidget(self._models_line(model_labels))
        page_layout.addWidget(self.editor, 1)
        self.tabs.addTab(page, "Gestures")
        layout.addWidget(self.tabs, 1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)

        self.retrain_button = QtWidgets.QPushButton("APPLY + RETRAIN...")
        self.retrain_button.setToolTip(
            "Apply the settings, then start RETRAIN ML + NEURAL NET so the "
            "gestures recorded since the last training join both models. "
            "Asks to confirm first; runs in the background.")
        self.retrain_button.setVisible(self.editor.retrain_needed()
                                       and cfg.ENABLE_NEURAL_NET)
        self.retrain_button.clicked.connect(self._apply_and_retrain)
        buttons.addWidget(self.retrain_button)

        self.apply_button = QtWidgets.QPushButton("APPLY")
        self.apply_button.setObjectName("primary")
        self.apply_button.setToolTip(
            "Apply now. Switching a gesture off or on in the classifier "
            "takes effect at once and retrains nothing.")
        self.apply_button.clicked.connect(self._apply)
        buttons.addWidget(self.apply_button)
        layout.addLayout(buttons)

    def _models_line(self, model_labels) -> QtWidgets.QLabel:
        """What each deployed model was trained on, in one line."""
        if not model_labels:
            return wrap_label("No trained model yet: record gestures, then "
                              "RETRAIN. The classifier switches below become "
                              "available once a model knows the gesture.",
                              10, False, COL_TEXT_DIM)
        parts = [f"{name}: {len(labels)} gestures"
                 for name, labels in model_labels.items() if labels]
        return wrap_label("Trained models  --  " + "   |   ".join(parts),
                          10, False, COL_TEXT_DIM)

    # -- result -----------------------------------------------------------------

    def selected(self):
        return self.editor.selected()

    def custom_gestures(self) -> dict:
        return self.editor.custom_gestures()

    def classifier_off(self):
        return self.editor.classifier_off()

    def classifier_changes(self):
        return self.editor.classifier_changes()

    def _apply(self) -> None:
        message = self.editor.validate()
        if message:
            self.editor.warning.setText(message)
            return
        self.accept()

    def _apply_and_retrain(self) -> None:
        message = self.editor.validate()
        if message:
            self.editor.warning.setText(message)
            return
        self.retrain_requested = True
        self.accept()
