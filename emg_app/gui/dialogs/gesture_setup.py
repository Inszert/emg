"""
GESTURES... (the label set).
"""

import os
from collections import Counter

from emg_app import config as cfg
from emg_app.gui.tokens import COL_ACCENT, COL_TEXT_DIM, COL_WARN
from emg_app.utils import safe_slug
from emg_app.dataset import dataset_summary
from emg_app.gui.qt import QtWidgets
from emg_app.gui.widgets import fit_to_screen, make_label, wrap_label


class GestureSetupDialog(QtWidgets.QDialog):
    """
    Which gestures this session records.

    The built-in six are checked; the optional ones (`OPTIONAL_GESTURES`)
    ship unchecked, and a free-text box adds a name the tool has never heard
    of. The result is written back to the `GESTURES` global by the caller, so
    the folder layout, the plan, the combo boxes and every recording's config
    snapshot follow it.

    Switching a gesture OFF never touches data already on disk -- the folder
    and its recordings stay, they are simply not cued any more, and the
    trainer still finds them because discovery walks the tree rather than
    this list. That asymmetry is deliberate: a label set is a plan for what
    to record next, not a filter on what has been recorded.
    """

    def __init__(self, parent=None, selected=None):
        super().__init__(parent)
        self.setWindowTitle("Gestures to record")
        fit_to_screen(self, 560, 620, min_width=380, min_height=320)
        selected = list(selected if selected is not None else cfg.GESTURES)
        self.boxes = {}
        self._extra = []

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addWidget(make_label("Gestures to record", 15, True,
                                    COL_ACCENT))
        layout.addWidget(wrap_label(
            "Checked gestures are cued by START SESSION and offered "
            "everywhere a label is chosen. Unchecking one does NOT delete "
            "anything already recorded -- it just stops being cued.",
            10, False, COL_TEXT_DIM))

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        body = QtWidgets.QVBoxLayout(inner)
        body.setSpacing(4)

        counts = self._counts()
        body.addWidget(make_label("IN USE / BUILT IN", 10, True, COL_TEXT_DIM))
        for name in cfg.known_gestures():
            if name in cfg.OPTIONAL_GESTURES and name not in selected:
                continue
            body.addWidget(self._row(name, name in selected, counts))

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
                body.addWidget(self._row(name, False, counts))

        body.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(make_label("Add your own", 10, False, COL_TEXT_DIM))
        self.custom_edit = QtWidgets.QLineEdit()
        self.custom_edit.setPlaceholderText("e.g. WRIST_FLEX")
        self.custom_edit.setToolTip(
            "Upper case with underscores. It becomes a folder name, so it "
            "goes through the same slug rules as everything else.")
        self.custom_edit.returnPressed.connect(self._add_custom)
        row.addWidget(self.custom_edit, 1)
        add = QtWidgets.QPushButton("ADD")
        add.clicked.connect(self._add_custom)
        row.addWidget(add)
        layout.addLayout(row)

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

    def _counts(self) -> dict:
        """How many recordings each gesture already has, for context."""
        try:
            summary = dataset_summary(os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER))
        except OSError:
            return {}
        found = Counter()
        for sessions in summary.get("persons", {}).values():
            for counts in sessions.values():
                for label, n in counts.items():
                    found[label] += n
        return dict(found)

    def _row(self, name: str, checked: bool, counts: dict):
        box = QtWidgets.QCheckBox(name)
        box.setChecked(checked)
        cue = cfg.gesture_cue(name)
        have = counts.get(name, 0)
        bits = []
        if cue:
            bits.append(cue)
        bits.append("%d recording(s) on disk" % have if have
                    else "nothing recorded yet")
        box.setToolTip("  --  ".join(bits))
        self.boxes[name] = box

        holder = QtWidgets.QWidget()
        line = QtWidgets.QVBoxLayout(holder)
        line.setContentsMargins(0, 0, 0, 0)
        line.setSpacing(0)
        line.addWidget(box)
        caption = make_label("      " + "  --  ".join(bits), 9, False,
                             COL_TEXT_DIM if have else COL_WARN)
        caption.setWordWrap(True)
        line.addWidget(caption)
        return holder

    def _add_custom(self) -> None:
        raw = self.custom_edit.text().strip()
        if not raw:
            return
        name = safe_slug(raw, "GESTURE").upper()
        if name in self.boxes:
            self.boxes[name].setChecked(True)
            self.warning.setText("%s was already listed -- it is now checked."
                                 % name)
            self.custom_edit.clear()
            return
        box = QtWidgets.QCheckBox(name)
        box.setChecked(True)
        self.boxes[name] = box
        self._custom_box.insertWidget(self._custom_box.count() - 1, box)
        self._extra.append(name)
        self.warning.setText("")
        self.custom_edit.clear()

    def selected(self):
        """The checked gestures, in the order the dialog listed them."""
        return [name for name, box in self.boxes.items() if box.isChecked()]

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
