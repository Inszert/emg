"""
GESTURES... (the label set).
"""

import os

from emg_app import config as cfg
from emg_app.gestures import gesture_name, recorded_gestures
from emg_app.gui.tokens import COL_ACCENT, COL_GOOD, COL_TEXT_DIM, COL_WARN
from emg_app.dataset import dataset_summary
from emg_app.gui.qt import QtWidgets
from emg_app.gui.widgets import fit_to_screen, make_label, wrap_label


class GestureSetupDialog(QtWidgets.QDialog):
    """
    Which gestures this session records -- and adding new ones.

    The built-in six are checked; the optional ones (`OPTIONAL_GESTURES`)
    ship unchecked, and "Add your own" creates a gesture the tool has never
    heard of, with the instruction to show while cueing it. The caller writes
    the result to the `GESTURES` / `CUSTOM_GESTURES` globals and saves it to
    `<DATA_FOLDER>/gestures.json`, so the folder layout, the plan, every label
    combo, the calibration sweep, TEST MODEL's cues and every recording's
    config snapshot follow it -- and it is still there after a restart.

    Switching a gesture OFF never touches data already on disk -- the folder
    and its recordings stay, they are simply not cued any more, and the
    trainer still finds them because discovery walks the tree rather than
    this list. That asymmetry is deliberate: a label set is a plan for what
    to record next, not a filter on what has been recorded.

    NOTHING HERE RETRAINS. Each row says whether the deployed models know
    the gesture; a new one becomes part of them only when the operator
    presses RETRAIN.
    """

    def __init__(self, parent=None, selected=None, model_labels=None):
        super().__init__(parent)
        self.setWindowTitle("Gestures to record")
        fit_to_screen(self, 620, 680, min_width=420, min_height=360)
        selected = list(selected if selected is not None else cfg.GESTURES)
        #  {"ML model": [labels], "neural net": [labels]} -- what each
        #  deployed model was trained on, so every row can say so.
        self.model_labels = {k: set(v) for k, v in (model_labels or {}).items()
                             if v}
        self.custom = dict(cfg.CUSTOM_GESTURES)
        self.boxes = {}
        self._extra = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addWidget(make_label("Gestures to record", 15, True,
                                    COL_ACCENT))
        layout.addWidget(wrap_label(
            "Checked gestures are cued by START SESSION, offered everywhere a "
            "label is chosen, and calibrated by the sweep. Unchecking one does "
            "NOT delete anything already recorded -- it just stops being "
            "cued. Nothing here retrains a model: a new gesture joins the "
            "models only when you press RETRAIN.",
            10, False, COL_TEXT_DIM))

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        body = QtWidgets.QVBoxLayout(inner)
        body.setSpacing(4)

        self.counts, self.people = self._counts()
        body.addWidget(make_label("IN USE / BUILT IN / ADDED BY YOU", 10, True,
                                  COL_TEXT_DIM))
        for name in cfg.known_gestures():
            if name in cfg.OPTIONAL_GESTURES and name not in selected:
                continue
            body.addWidget(self._row(name, name in selected))
        #  A gesture added below goes HERE -- with the ones in use, not
        #  under the OPTIONAL heading that follows.
        self._insert_at = body.count()

        optional = [n for n in cfg.OPTIONAL_GESTURES if n not in selected]
        if optional:
            body.addWidget(make_label("", 4))
            body.addWidget(make_label("OPTIONAL -- off by default", 10, True,
                                      COL_TEXT_DIM))
            body.addWidget(wrap_label(
                "Every extra class costs recording time from every future "
                "subject and makes the problem harder, so these are only "
                "worth switching on when driven by a different muscle group "
                "rather than just a different-looking hand.",
                9, False, COL_TEXT_DIM))
            for name in optional:
                body.addWidget(self._row(name, False))

        body.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        layout.addWidget(make_label("ADD YOUR OWN", 10, True, COL_TEXT_DIM))
        form = QtWidgets.QFormLayout()
        self.custom_edit = QtWidgets.QLineEdit()
        self.custom_edit.setPlaceholderText("e.g. WRIST_FLEX")
        self.custom_edit.setToolTip(
            "Upper case with underscores. It becomes a folder name, so it "
            "goes through the same slug rules as everything else.")
        self.custom_edit.returnPressed.connect(self._add_custom)
        form.addRow("Name", self.custom_edit)
        self.cue_edit = QtWidgets.QLineEdit()
        self.cue_edit.setPlaceholderText(
            "how to do it, e.g. bend the wrist down, fingers relaxed")
        self.cue_edit.setToolTip(
            "Shown under the gesture's name while it is cued -- in START "
            "SESSION, the calibration sweep and TEST MODEL.")
        self.cue_edit.returnPressed.connect(self._add_custom)
        form.addRow("Instruction", self.cue_edit)
        layout.addLayout(form)
        add_row = QtWidgets.QHBoxLayout()
        add_row.addStretch(1)
        add = QtWidgets.QPushButton("ADD GESTURE")
        add.clicked.connect(self._add_custom)
        add_row.addWidget(add)
        layout.addLayout(add_row)

        self.warning = wrap_label("", 10, True, COL_WARN)
        layout.addWidget(self.warning)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        ok = QtWidgets.QPushButton("USE THESE")
        ok.setObjectName("primary")
        ok.clicked.connect(self._accept)
        buttons.addWidget(ok)
        layout.addLayout(buttons)
        self._custom_box = body

    # -- pieces ---------------------------------------------------------------

    def _counts(self):
        """Recordings and people per gesture already on disk, for context."""
        try:
            summary = dataset_summary(os.path.join(cfg.PROJECT_DIR,
                                                   cfg.DATA_FOLDER))
        except OSError:
            return {}, {}
        return recorded_gestures(summary)

    def _knowledge(self, name: str) -> str:
        """Which deployed models know this gesture -- the retrain question."""
        if not self.model_labels:
            return ""
        knows = [m for m, labels in self.model_labels.items() if name in labels]
        if len(knows) == len(self.model_labels):
            return "the models know it"
        if knows:
            return "only the " + " and ".join(knows) + " knows it"
        return "NOT in the models yet -- RETRAIN after recording it"

    def _caption(self, name: str) -> tuple:
        bits = []
        cue = cfg.gesture_cue(name) or self.custom.get(name, "")
        if name in self.custom:
            bits.append("added by you")
        if cue:
            bits.append(cue)
        have = self.counts.get(name, 0)
        people = self.people.get(name, 0)
        bits.append(f"{have} recording(s), {people} person(s)" if have
                    else "nothing recorded yet")
        knowledge = self._knowledge(name)
        if knowledge:
            bits.append(knowledge)
        if have and people < 2:
            bits.append("record it on a second person before trusting a "
                        "retrain's score")
        unknown = bool(knowledge.startswith("NOT"))
        colour = COL_WARN if (not have or unknown) else (
            COL_GOOD if knowledge == "the models know it" else COL_TEXT_DIM)
        return "  --  ".join(bits), colour

    def _row(self, name: str, checked: bool):
        box = QtWidgets.QCheckBox(name)
        box.setChecked(checked)
        text, colour = self._caption(name)
        box.setToolTip(text)
        self.boxes[name] = box

        holder = QtWidgets.QWidget()
        line = QtWidgets.QVBoxLayout(holder)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(0)
        line.addWidget(box)
        caption = make_label("      " + text, 9, False, colour)
        caption.setWordWrap(True)
        line.addWidget(caption)
        return holder

    def _add_custom(self) -> None:
        raw = self.custom_edit.text().strip()
        if not raw:
            self.warning.setText("Type a name for the new gesture first.")
            return
        name = gesture_name(raw)
        cue = self.cue_edit.text().strip()
        if name in self.boxes:
            self.boxes[name].setChecked(True)
            if cue and (name in self.custom or not cfg.gesture_cue(name)):
                self.custom[name] = cue
            self.warning.setText("%s was already listed -- it is now checked."
                                 % name)
            self.custom_edit.clear()
            self.cue_edit.clear()
            return
        self.custom[name] = cue
        self._extra.append(name)
        self._custom_box.insertWidget(self._insert_at, self._row(name, True))
        self._insert_at += 1
        self.warning.setText(
            f"{name} added. Press USE THESE, then record it. The models will "
            "not know it until you RETRAIN -- that is your call.")
        self.custom_edit.clear()
        self.cue_edit.clear()

    def selected(self):
        """The checked gestures, in the order the dialog listed them."""
        return [name for name, box in self.boxes.items() if box.isChecked()]

    def custom_gestures(self) -> dict:
        """Every gesture added by hand (checked or not) -> its instruction."""
        return dict(self.custom)

    def _accept(self) -> None:
        chosen = self.selected()
        if len(chosen) < 2:
            self.warning.setText(
                "Pick at least two gestures -- a classifier needs something "
                "to tell apart.")
            return
        #  REST is not required, but losing it is worth saying out loud: it is
        #  the only class that says "no gesture", and without it a live
        #  classifier has to answer with a gesture at all times.
        if "REST" not in chosen:
            self.warning.setText(
                "No REST selected. Without it the classifier has no 'no "
                "gesture' answer and will always name one. Check REST, or "
                "press USE THESE again to go ahead anyway.")
            if not getattr(self, "_warned_rest", False):
                self._warned_rest = True
                return
        self.accept()
