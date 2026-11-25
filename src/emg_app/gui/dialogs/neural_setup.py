"""
NEURAL SETUP (nn_spec.json).
"""


from emg_app.gui.tokens import COL_TEXT_DIM, COL_WARN, FS_SM
from emg_app.neural import (
    NN_KINDS, NN_SPEC_DEFAULTS, save_nn_spec, validate_nn_spec)
from emg_app.gui.qt import QtWidgets
from emg_app.gui.widgets import fit_to_screen, make_label, scroll_dialog_body


class NeuralSetupDialog(QtWidgets.QDialog):
    """
    Which architectures and hyperparameters TRAIN NEURAL NET sweeps.

    Saved to models/nn_spec.json and passed to train_nn.py as flags. Changes
    only the NEXT neural training; the scoring protocol is not configurable
    here, so choosing settings cannot flatter the net.
    """

    def __init__(self, spec: dict, path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Neural network setup")
        self.spec_path = path
        self.saved = False
        spec, _problems = validate_nn_spec(spec)

        root = QtWidgets.QVBoxLayout(self)
        root.addWidget(make_label(
            "Architectures to compare (each is scored on the same folds):",
            FS_SM, True))
        self.kind_boxes = {}
        for kind, text in NN_KINDS.items():
            box = QtWidgets.QCheckBox(f"{kind}  --  {text}")
            box.setChecked(kind in spec["kinds"])
            self.kind_boxes[kind] = box
            root.addWidget(box)

        form = QtWidgets.QFormLayout()
        self.windows_edit = QtWidgets.QLineEdit(
            ", ".join(f"{w:g}" for w in spec["windows"]))
        self.windows_edit.setToolTip("Window lengths in seconds, comma "
                                     "separated (0.1 - 2.0).")
        form.addRow("Window lengths (s)", self.windows_edit)

        def spin(lo, hi, value, step, decimals=0):
            w = (QtWidgets.QDoubleSpinBox() if decimals
                 else QtWidgets.QSpinBox())
            if decimals:
                w.setDecimals(decimals)
            w.setRange(lo, hi)
            w.setSingleStep(step)
            w.setValue(value)
            return w
        self.overlap = spin(0.0, 0.9, spec["overlap"], 0.05, 2)
        self.epochs = spin(5, 500, spec["epochs"], 5)
        self.hidden = spin(8, 256, spec["hidden"], 8)
        self.dropout = spin(0.0, 0.8, spec["dropout"], 0.05, 2)
        self.lr = spin(0.00001, 0.1, spec["learning_rate"], 0.0005, 5)
        self.wd = spin(0.0, 0.1, spec["weight_decay"], 0.0005, 5)
        self.batch = spin(8, 512, spec["batch_size"], 8)
        self.smooth = spin(0.0, 0.5, spec["label_smoothing"], 0.01, 2)
        for label, widget in (
                ("Window overlap", self.overlap),
                ("Epochs (fixed, no early stopping)", self.epochs),
                ("Hidden size", self.hidden), ("Dropout", self.dropout),
                ("Learning rate", self.lr), ("Weight decay", self.wd),
                ("Batch size", self.batch),
                ("Label smoothing", self.smooth)):
            form.addRow(label, widget)
        root.addLayout(form)

        self.problem_label = make_label("", 10, False, COL_WARN)
        self.problem_label.setWordWrap(True)
        root.addWidget(self.problem_label)
        root.addWidget(make_label(
            f"Saved to {path} and read by train_nn.py on the next training.",
            10, False, COL_TEXT_DIM))

        buttons = QtWidgets.QHBoxLayout()
        reset = QtWidgets.QPushButton("RESET TO DEFAULTS")
        reset.clicked.connect(self._reset)
        buttons.addWidget(reset)
        buttons.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        save = QtWidgets.QPushButton("SAVE SETUP")
        save.setObjectName("primary")
        save.setDefault(True)
        save.clicked.connect(self._save)
        buttons.addWidget(save)
        root.addLayout(buttons)
        scroll_dialog_body(self, keep_last=1)
        fit_to_screen(self, 640, 620, min_width=480, min_height=420)

    def current_spec(self) -> dict:
        wins = []
        for part in self.windows_edit.text().replace(";", ",").split(","):
            part = part.strip()
            if part:
                try:
                    wins.append(float(part))
                except ValueError:
                    pass
        return {"kinds": [k for k, b in self.kind_boxes.items()
                          if b.isChecked()],
                "windows": wins, "overlap": self.overlap.value(),
                "epochs": self.epochs.value(), "hidden": self.hidden.value(),
                "dropout": self.dropout.value(),
                "learning_rate": self.lr.value(),
                "weight_decay": self.wd.value(),
                "batch_size": self.batch.value(),
                "label_smoothing": self.smooth.value()}

    def _reset(self) -> None:
        d = NN_SPEC_DEFAULTS
        for k, b in self.kind_boxes.items():
            b.setChecked(k in d["kinds"])
        self.windows_edit.setText(", ".join(f"{w:g}" for w in d["windows"]))
        self.overlap.setValue(d["overlap"])
        self.epochs.setValue(d["epochs"])
        self.hidden.setValue(d["hidden"])
        self.dropout.setValue(d["dropout"])
        self.lr.setValue(d["learning_rate"])
        self.wd.setValue(d["weight_decay"])
        self.batch.setValue(d["batch_size"])
        self.smooth.setValue(d["label_smoothing"])

    def _save(self) -> None:
        spec, problems = validate_nn_spec(self.current_spec())
        try:
            save_nn_spec(spec, self.spec_path)
        except OSError as exc:
            self.problem_label.setText(f"Not saved: {exc}")
            return
        self.spec = spec
        self.saved = True
        self.accept()
