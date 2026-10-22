"""
The recording-quality dialog.
"""


from emg_app.gui.tokens import COL_TEXT_DIM
from emg_app.recording import Recording
from emg_app.gui.qt import QtWidgets
from emg_app.gui.widgets import fit_to_screen, make_label


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
