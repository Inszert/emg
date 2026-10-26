"""
The gesture set: what is recorded, what the live classifiers may answer with,
and adding new gestures.

`GestureEditor` is the whole editor as one widget. It lives in two places --
the settings dialog (the gear icon) and the older GESTURES... dialog, which
is now a thin shell around it -- so the two cannot drift apart.
"""

import os

from emg_app import config as cfg
from emg_app.class_mask import MIN_ACTIVE, validate_off
from emg_app.gestures import (
    describe_gaps, gesture_name, model_gesture_gaps, recorded_gestures)
from emg_app.gui.tokens import COL_ACCENT, COL_GOOD, COL_TEXT_DIM, COL_WARN
from emg_app.dataset import dataset_summary
from emg_app.gui.qt import QtWidgets
from emg_app.gui.widgets import fit_to_screen, make_label, wrap_label


class GestureEditor(QtWidgets.QWidget):
    """
    Two independent questions per gesture, and adding new ones.

    RECORD (the checkbox with the name). Which gestures this session cues,
    offers in every label combo and calibrates. The built-in six are checked;
    the optional ones (`OPTIONAL_GESTURES`) ship unchecked, and "Add your own"
    creates a gesture the tool has never heard of, with the instruction to
    show while cueing it. Switching one OFF never touches data already on
    disk -- the folder and its recordings stay, they are simply not cued any
    more, and the trainer still finds them because discovery walks the tree
    rather than this list.

    CLASSIFIER (the second checkbox). Whether the LIVE classifiers -- the ML
    model and the neural net alike -- may answer with this gesture. Unticking
    it masks it out of their output (emg_app/class_mask.py): NOTHING is
    retrained, so ticking it again brings it back at once. It is only
    available for a gesture a trained model already knows; a gesture no model
    was trained on has no output to switch, and needs RECORD, then RETRAIN.

    The caller applies the result (`selected()`, `custom_gestures()`,
    `classifier_off()`) to the config and saves it to
    `<DATA_FOLDER>/gestures.json`, so the folder layout, the plan, every
    label combo, the calibration sweep, TEST MODEL and every recording's
    config snapshot follow it -- and it is still there after a restart.
    """

    def __init__(self, parent=None, selected=None, model_labels=None,
                 classifier_off=None):
        super().__init__(parent)
        selected = list(selected if selected is not None else cfg.GESTURES)
        #  {"ML model": [labels], "neural net": [labels]} -- what each
        #  deployed model was trained on, so every row can say so.
        self.model_labels = {k: set(v) for k, v in (model_labels or {}).items()
                             if v}
        self.known_to_models = set().union(*self.model_labels.values()) \
            if self.model_labels else set()
        self.off0 = set(cfg.CLASSIFIER_OFF if classifier_off is None
                        else classifier_off)
        self.custom = dict(cfg.CUSTOM_GESTURES)
        self.boxes = {}                # RECORD checkboxes, by gesture
        self.cls_boxes = {}            # CLASSIFIER checkboxes, by gesture
        self._extra = []
        self._warned_rest = False
        self._warned_rest_off = False

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addWidget(wrap_label(
            "RECORD: checked gestures are cued by START SESSION, offered "
            "everywhere a label is chosen, and calibrated by the sweep. "
            "Unchecking one does NOT delete anything already recorded.\n"
            "CLASSIFIER: untick to switch a gesture OFF in the live "
            "classifiers (ML model and neural net). Nothing is retrained, so "
            "ticking it again brings it back at once. A gesture no model has "
            "been trained on cannot be switched -- record it, then RETRAIN.",
            10, False, COL_TEXT_DIM))

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        body = QtWidgets.QVBoxLayout(inner)
        body.setSpacing(4)

        self.counts, self.people = self._counts()
        head = QtWidgets.QHBoxLayout()
        head.addWidget(make_label("RECORD  --  IN USE / BUILT IN / ADDED BY "
                                  "YOU", 10, True, COL_TEXT_DIM))
        head.addStretch(1)
        head.addWidget(make_label("USE IN CLASSIFIER", 10, True, COL_TEXT_DIM))
        body.addLayout(head)
        #  Every gesture the tool knows, plus any label a trained model has
        #  that the tool no longer lists (so its switch is never missing).
        listed = list(cfg.known_gestures())
        for name in sorted(self.known_to_models):
            if name not in listed:
                listed.append(name)
        for name in listed:
            if name in cfg.OPTIONAL_GESTURES and name not in selected \
                    and name not in self.known_to_models:
                continue
            body.addWidget(self._row(name, name in selected))
        #  A gesture added below goes HERE -- with the ones in use, not
        #  under the OPTIONAL heading that follows.
        self._insert_at = body.count()

        optional = [n for n in cfg.OPTIONAL_GESTURES
                    if n not in selected and n not in self.boxes]
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
        self.add_button = QtWidgets.QPushButton("ADD GESTURE")
        self.add_button.clicked.connect(self._add_custom)
        add_row.addWidget(self.add_button)
        layout.addLayout(add_row)

        #  What a retrain would pick up, said before anyone has to ask.
        self.retrain_note = wrap_label("", 10, True, COL_WARN)
        layout.addWidget(self.retrain_note)
        self.warning = wrap_label("", 10, True, COL_WARN)
        layout.addWidget(self.warning)
        self._custom_box = body
        self._refresh_retrain_note()

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

    def _classifier_box(self, name: str) -> QtWidgets.QCheckBox:
        """The switch for one gesture in the live classifiers."""
        box = QtWidgets.QCheckBox("on")
        known = name in self.known_to_models
        box.setEnabled(known)
        box.setChecked(known and name not in self.off0)
        if known:
            box.setToolTip(
                f"Untick to switch {name} OFF in the live classifiers (the "
                "ML model and the neural net). It is masked out of their "
                "answer: nothing is retrained and nothing is deleted, so "
                "ticking it again brings it back at once.")
        elif self.model_labels:
            box.setText("n/a")
            box.setToolTip(
                f"No trained model knows {name} yet, so there is nothing to "
                "switch. Record it, then RETRAIN.")
        else:
            box.setText("n/a")
            box.setToolTip("No model is trained yet.")
        self.cls_boxes[name] = box
        return box

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
        top = QtWidgets.QHBoxLayout()
        top.addWidget(box)
        top.addStretch(1)
        top.addWidget(self._classifier_box(name))
        line.addLayout(top)
        caption = make_label("      " + text, 9, False, colour)
        caption.setWordWrap(True)
        line.addWidget(caption)
        return holder

    def _gaps(self) -> dict:
        """{"ML model": [gap, ...], ...}: recorded but not in that model."""
        out = {}
        for model, labels in self.model_labels.items():
            gaps = model_gesture_gaps(labels, self.counts, self.people)
            if gaps:
                out[model] = gaps
        return out

    def retrain_needed(self) -> bool:
        """True when a recorded gesture is missing from a trained model."""
        return bool(self._gaps())

    def _refresh_retrain_note(self) -> None:
        gaps = self._gaps()
        if gaps:
            self.retrain_note.setText("\n".join(
                describe_gaps(g, model.upper()) for model, g in gaps.items()))
            return
        fresh = [n for n in self._extra if not self.counts.get(n)]
        if fresh:
            self.retrain_note.setText(
                f"{', '.join(fresh)} added. Record it, then RETRAIN -- the "
                "models cannot use a gesture they have never seen.")
        else:
            self.retrain_note.setText("")

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
        #  What to do next is said once, by the retrain note above.
        self.warning.setText("")
        self._refresh_retrain_note()
        self.custom_edit.clear()
        self.cue_edit.clear()

    def lock_recording(self, reason: str) -> None:
        """
        Freeze everything that changes what is RECORDED, say why, and leave
        the classifier switches working: they mask the models' output and
        cannot disturb a recording in progress.
        """
        for box in self.boxes.values():
            box.setEnabled(False)
        for widget in (self.custom_edit, self.cue_edit, self.add_button):
            widget.setEnabled(False)
        self.warning.setText(reason)

    # -- result ---------------------------------------------------------------

    def selected(self):
        """The checked gestures, in the order the editor listed them."""
        return [name for name, box in self.boxes.items() if box.isChecked()]

    def custom_gestures(self) -> dict:
        """Every gesture added by hand (checked or not) -> its instruction."""
        return dict(self.custom)

    def classifier_off(self):
        """
        The gestures switched OFF in the classifiers.

        Only switches that can act count: an unticked box on a gesture no
        model knows is "n/a", not "off". A gesture that was off before and is
        no longer listed keeps its state, so editing the list can never
        silently switch it back on.
        """
        off = [name for name, box in self.cls_boxes.items()
               if box.isEnabled() and not box.isChecked()]
        off += [name for name in sorted(self.off0)
                if name not in self.cls_boxes and name not in off]
        return off

    def classifier_changes(self):
        """(switched_off, switched_on) relative to what was in force."""
        now = set(self.classifier_off())
        return sorted(now - self.off0), sorted(self.off0 - now)

    def validate(self) -> str:
        """
        "" when the editor may be applied, otherwise what to tell the user.

        Mirrors the old dialog: too few gestures is refused; no REST is
        warned about once, and the second press goes ahead.
        """
        chosen = self.selected()
        if len(chosen) < 2:
            return ("Pick at least two gestures -- a classifier needs "
                    "something to tell apart.")
        #  REST is not required, but losing it is worth saying out loud: it is
        #  the only class that says "no gesture", and without it a live
        #  classifier has to answer with a gesture at all times.
        if "REST" not in chosen and not self._warned_rest:
            self._warned_rest = True
            return ("No REST selected. Without it the classifier has no 'no "
                    "gesture' answer and will always name one. Check REST, "
                    "or press apply again to go ahead anyway.")
        off = self.classifier_off()
        _kept, problems = validate_off(
            off, {m: sorted(l) for m, l in self.model_labels.items()})
        if problems:
            return (problems[0] + f" Keep at least {MIN_ACTIVE} gestures "
                    "on for every model.")
        if "REST" in off and not self._warned_rest_off:
            self._warned_rest_off = True
            return ("REST is switched off in the classifier. Without it the "
                    "classifiers have no 'no gesture' answer and will always "
                    "name a gesture. Tick it, or press apply again to go "
                    "ahead anyway.")
        return ""


class GestureSetupDialog(QtWidgets.QDialog):
    """
    GESTURES... -- the editor on its own, as a dialog.

    Kept as its own class (and with the attribute names it always had) because
    the plan panel's button, the tests and `edit_gestures()` use it; the
    settings dialog hosts the same `GestureEditor` as one tab.
    """

    def __init__(self, parent=None, selected=None, model_labels=None,
                 classifier_off=None):
        super().__init__(parent)
        self.setWindowTitle("Gestures")
        fit_to_screen(self, 640, 720, min_width=420, min_height=400)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addWidget(make_label("Gestures", 15, True, COL_ACCENT))
        self.editor = GestureEditor(self, selected, model_labels,
                                    classifier_off)
        layout.addWidget(self.editor, 1)

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

    # The names the old single-class dialog exposed, now served by the editor.
    @property
    def boxes(self):
        return self.editor.boxes

    @property
    def custom_edit(self):
        return self.editor.custom_edit

    @property
    def cue_edit(self):
        return self.editor.cue_edit

    @property
    def warning(self):
        return self.editor.warning

    def _add_custom(self) -> None:
        self.editor._add_custom()

    def _caption(self, name: str) -> tuple:
        return self.editor._caption(name)

    def selected(self):
        return self.editor.selected()

    def custom_gestures(self) -> dict:
        return self.editor.custom_gestures()

    def classifier_off(self):
        return self.editor.classifier_off()

    def _accept(self) -> None:
        message = self.editor.validate()
        if message:
            self.editor.warning.setText(message)
            return
        self.accept()
