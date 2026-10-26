"""
The tabs of the settings dialog beyond Gestures:

    Models     pick the ML classifiers and the neural nets from the zoo, see
               the stars, the size of each and of the selection, soft voting
    Training   augmentation and adaptation to the person
    General    the two questions the application asks (calibrate the hand?
               retrain on new data?)

Each tab only collects values (`values()`); nothing here writes settings or
touches a model. `MainWindow._apply_app_settings` validates, applies and saves.
"""

from typing import List

from emg_app import config as cfg
from emg_app import zoo
from emg_app.gui.qt import QtCore, QtWidgets
from emg_app.gui.tokens import COL_ACCENT, COL_BAD, COL_TEXT_DIM, COL_WARN
from emg_app.gui.widgets import make_label, wrap_label

STAR = "★"

AUG_LABELS = {
    "gain": ("Gain", "Random overall and per-channel gain: electrode pressure "
                     "and skin drift."),
    "noise": ("Noise", "Multiplicative jitter on every sample: sensor noise."),
    "timewarp": ("Time warp", "Speed +-15 % and a small shift: how long and "
                              "how fast a person holds a gesture."),
    "rotate": ("Band rotation", "The band slid one pod round the arm "
                                "(circular channel roll) -- the one thing "
                                "calibration cannot fix."),
    "chandrop": ("Dead electrode", "One electrode reads zero in 30 % of the "
                                   "windows: a pod that lost contact."),
}


def _pct(v) -> str:
    return "--" if v is None else f"{v * 100:.1f} %"


class ModelsTab(QtWidgets.QWidget):
    """Choose the models the live classifiers use."""

    COLS = ["", "", "model", "per window", "macro-F1", "size", "parameters",
            "ms / window", "trained with"]

    def __init__(self, parent=None, entries=None, ensembles=None):
        super().__init__(parent)
        self.entries = list(entries if entries is not None
                            else zoo.list_entries())
        self.ensembles = (ensembles if ensembles is not None
                          else zoo.read_ensembles())
        self.stars = zoo.compute_stars(self.entries, self.ensembles)
        self.tables = {}
        self.summaries = {}
        self.vote_notes = {}
        self._building = True

        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        self.enable_box = QtWidgets.QCheckBox(
            "Use the models selected below (otherwise: the original "
            "deployed models)")
        self.enable_box.setChecked(bool(cfg.MODEL_SELECTION_ENABLED))
        self.enable_box.setToolTip(
            "Off: the ML classifier is MODEL_PATH and the neural net is "
            "NN_MODEL_PATH, exactly as before. On: the models you tick here. "
            "One model = that model; several = their soft vote (the mean of "
            "their probabilities).")
        self.enable_box.toggled.connect(self._refresh)
        layout.addWidget(self.enable_box)

        if not self.entries:
            layout.addWidget(wrap_label(
                "The model zoo is empty. Train it once with:\n"
                "    python training/train_zoo.py all\n"
                "(about an hour on this machine; it never touches the "
                "original models).", 11, False, COL_WARN))
        layout.addWidget(wrap_label(
            f"{STAR} marks the best models: the top {zoo.STAR_TOP_N} of each "
            "family by leave-one-person-out accuracy per window (ties by "
            "macro-F1), and a soft-vote combination when it beats every "
            "single model. Recomputed after every training. ML models and "
            "neural nets are never voted together.", 10, False, COL_TEXT_DIM))

        for family, title in (("ml", "ML classifiers"),
                              ("nn", "Neural nets")):
            layout.addWidget(self._family_box(family, title), 1)
        self._building = False
        self._refresh()

    # -- building -------------------------------------------------------------

    def _family_box(self, family: str, title: str) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        v = QtWidgets.QVBoxLayout(box)
        rows = zoo.rank([e for e in self.entries if e.family == family])
        table = QtWidgets.QTableWidget(len(rows), len(self.COLS))
        table.setHorizontalHeaderLabels(self.COLS)
        table.verticalHeader().setVisible(False)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setMinimumHeight(110)
        head = table.horizontalHeader()
        head.setStretchLastSection(True)
        head.setSectionResizeMode(2, QtWidgets.QHeaderView.Stretch)
        selected = list(cfg.SELECTED_ML if family == "ml"
                        else cfg.SELECTED_NN)
        for r, e in enumerate(rows):
            use = QtWidgets.QTableWidgetItem()
            use.setFlags(QtCore.Qt.ItemIsUserCheckable | QtCore.Qt.ItemIsEnabled)
            use.setCheckState(QtCore.Qt.Checked if e.id in selected
                              else QtCore.Qt.Unchecked)
            use.setData(QtCore.Qt.UserRole, e.id)
            use.setToolTip("Use this model in the live classifier")
            if not e.exists:
                use.setFlags(QtCore.Qt.NoItemFlags)
                use.setToolTip("the model file is missing")
            table.setItem(r, 0, use)
            star = QtWidgets.QTableWidgetItem(
                STAR if e.id in self.stars["models"] else "")
            star.setForeground(QtCore.Qt.darkYellow)
            if e.id in self.stars["models"]:
                star.setToolTip(self.stars["models"][e.id])
            table.setItem(r, 1, star)
            lat = e.meta.get("latency_ms")
            tw = e.trained_with
            cells = [e.name, _pct(e.window_acc), _pct(e.macro_f1),
                     zoo.human_size(e.size_bytes),
                     "--" if e.params is None else f"{e.params:,}".replace(
                         ",", " "),
                     "--" if lat is None else f"{lat:.1f}",
                     ", ".join(filter(None, [
                         "aug " + "+".join(tw.get("augment") or [])
                         if tw.get("augment") else "",
                         "adaptation" if tw.get("adapt") else ""])) or "plain"]
            for c, text in enumerate(cells, start=2):
                item = QtWidgets.QTableWidgetItem(text)
                item.setToolTip(e.meta.get("description", ""))
                if c >= 3:
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        table.itemChanged.connect(self._on_item_changed)
        self.tables[family] = table
        v.addWidget(table)

        self.summaries[family] = wrap_label("", 10, False, COL_TEXT_DIM)
        v.addWidget(self.summaries[family])
        self.vote_notes[family] = wrap_label("", 10, False, COL_TEXT_DIM)
        v.addWidget(self.vote_notes[family])

        buttons = QtWidgets.QHBoxLayout()
        top = QtWidgets.QPushButton(f"Select {STAR} models")
        top.setToolTip("Tick exactly the starred models of this family")
        top.clicked.connect(lambda: self._select_stars(family))
        best = QtWidgets.QPushButton(f"Select best combination {STAR}")
        best.setToolTip("Tick the members of the best soft-voting subset "
                        "(only offered when it beats every single model)")
        best.setEnabled(family in self.stars["ensembles"])
        if family in self.stars["ensembles"]:
            best.setToolTip(self.stars["ensembles"][family])
        best.clicked.connect(lambda: self._select_combo(family))
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(lambda: self._set_checked(family, []))
        for b in (top, best, clear):
            buttons.addWidget(b)
        buttons.addStretch(1)
        v.addLayout(buttons)
        return box

    # -- selection ------------------------------------------------------------

    def _set_checked(self, family: str, ids: List[str]) -> None:
        table = self.tables[family]
        self._building = True
        for r in range(table.rowCount()):
            item = table.item(r, 0)
            if item.flags() & QtCore.Qt.ItemIsUserCheckable:
                item.setCheckState(QtCore.Qt.Checked
                                   if item.data(QtCore.Qt.UserRole) in ids
                                   else QtCore.Qt.Unchecked)
        self._building = False
        self._refresh()

    def _select_stars(self, family: str) -> None:
        ids = [e.id for e in self.entries
               if e.family == family and e.id in self.stars["models"]]
        self._set_checked(family, ids)

    def _select_combo(self, family: str) -> None:
        combo = self.ensembles.get(family) or {}
        self._set_checked(family, list(combo.get("members") or []))

    def _on_item_changed(self, _item) -> None:
        if not self._building:
            self._refresh()

    def chosen(self, family: str) -> List[str]:
        """Ids ticked in a family, in the table's (rank) order."""
        table = self.tables[family]
        out = []
        for r in range(table.rowCount()):
            item = table.item(r, 0)
            if item.checkState() == QtCore.Qt.Checked:
                out.append(item.data(QtCore.Qt.UserRole))
        return out

    def _entries_of(self, ids):
        by_id = {e.id: e for e in self.entries}
        return [by_id[i] for i in ids if i in by_id]

    def problems(self) -> List[str]:
        """Why the current selection cannot be applied (empty = fine)."""
        if not self.enable_box.isChecked():
            return []
        out = []
        for family in ("ml", "nn"):
            ids = self.chosen(family)
            if len(ids) >= 2:
                ok, why = zoo.check_vote(self._entries_of(ids))
                if not ok:
                    out.append(f"{family.upper()}: {why}")
        return out

    def _refresh(self, *_args) -> None:
        if self._building:
            return
        on = self.enable_box.isChecked()
        for family in ("ml", "nn"):
            self.tables[family].setEnabled(on)
            ids = self.chosen(family)
            sel = self._entries_of(ids)
            s = zoo.selection_summary(sel)
            if not sel:
                text = ("Nothing ticked: the original deployed model is used "
                        "for this family.")
            else:
                text = (f"{s['n']} selected  |  total size "
                        f"{zoo.human_size(s['size_bytes'])}"
                        + ("" if s["size_known"] else " (some unknown)")
                        + (f"  |  {s['params']:,} parameters".replace(",", " ")
                           if s["params"] is not None else ""))
            self.summaries[family].setText(text)
            note = self.vote_notes[family]
            if len(sel) >= 2:
                ok, why = zoo.check_vote(sel)
                if ok:
                    note.setText("Soft vote: the answer is the mean of "
                                 "these models' probabilities.")
                    note.setStyleSheet(f"color: {COL_ACCENT};")
                else:
                    note.setText("Cannot vote: " + why)
                    note.setStyleSheet(f"color: {COL_BAD};")
            else:
                note.setText("")

    # -- result -----------------------------------------------------------------

    def values(self) -> dict:
        return {"MODEL_SELECTION_ENABLED": self.enable_box.isChecked(),
                "SELECTED_ML": self.chosen("ml"),
                "SELECTED_NN": self.chosen("nn")}


class TrainingTab(QtWidgets.QWidget):
    """Augmentation and adaptation, per family: what the TRAINERS do next time."""

    def __init__(self, parent=None, defaults_note: str = ""):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        layout.addWidget(wrap_label(
            "These switches change how models are TRAINED. A model that is "
            "already trained keeps what it learned; train again for a "
            "switch to take effect. Defaults come from the measured ablation "
            "study (docs/REPORT.md) -- a switch is on by default only if it "
            "helped, and the neural nets and the ML classifiers were "
            "measured separately because they do not agree.", 10, False,
            COL_TEXT_DIM))
        if defaults_note:
            layout.addWidget(wrap_label(defaults_note, 10, False,
                                        COL_TEXT_DIM))

        self.aug_boxes = {"NN": {}, "ML": {}}
        self.adapt_boxes = {}
        columns = QtWidgets.QHBoxLayout()
        for fam, title, current_aug, current_adapt in (
                ("NN", "Neural nets", cfg.TRAIN_AUGMENT_NN,
                 cfg.TRAIN_ADAPT_NN),
                ("ML", "ML classifiers (model zoo)", cfg.TRAIN_AUGMENT_ML,
                 cfg.TRAIN_ADAPT_ML)):
            box = QtWidgets.QGroupBox(title)
            v = QtWidgets.QVBoxLayout(box)
            v.addWidget(make_label("Augmentation", 10, True, COL_TEXT_DIM))
            for name, (label, tip) in AUG_LABELS.items():
                check = QtWidgets.QCheckBox(label)
                check.setToolTip(tip)
                check.setChecked(name in current_aug)
                self.aug_boxes[fam][name] = check
                v.addWidget(check)
            v.addWidget(make_label("Adaptation to the person", 10, True,
                                   COL_TEXT_DIM))
            adapt = QtWidgets.QCheckBox(
                "Train with adaptation (each person standardised with "
                "their own statistics)")
            adapt.setChecked(bool(current_adapt))
            adapt.setToolTip(
                "Each person's signal is standardised with statistics from "
                "about 9 s of their own unlabelled signal -- the length of a "
                "calibration sweep.")
            self.adapt_boxes[fam] = adapt
            v.addWidget(adapt)
            v.addStretch(1)
            columns.addWidget(box, 1)
        layout.addLayout(columns)

        self.adapt_live = QtWidgets.QCheckBox(
            "Use adaptation live (only models that were trained with it; off "
            "= they use their stored global statistics)")
        self.adapt_live.setChecked(bool(cfg.ADAPT_LIVE))
        layout.addWidget(self.adapt_live)
        layout.addWidget(wrap_label(
            "The person's statistics come from the calibration: the signal "
            "captured while the hand is calibrated is stored with it. The "
            "ML switches apply to models trained by the model zoo "
            "(training/train_zoo.py); the classic RETRAIN sweep trains "
            "plain models.", 10, False, COL_TEXT_DIM))
        layout.addStretch(1)

    def values(self) -> dict:
        return {
            "TRAIN_AUGMENT_NN": [n for n, b in self.aug_boxes["NN"].items()
                                 if b.isChecked()],
            "TRAIN_AUGMENT_ML": [n for n, b in self.aug_boxes["ML"].items()
                                 if b.isChecked()],
            "TRAIN_ADAPT_NN": self.adapt_boxes["NN"].isChecked(),
            "TRAIN_ADAPT_ML": self.adapt_boxes["ML"].isChecked(),
            "ADAPT_LIVE": self.adapt_live.isChecked()}


class GeneralTab(QtWidgets.QWidget):
    """The two questions the application asks."""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)
        box = QtWidgets.QGroupBox("Questions")
        v = QtWidgets.QVBoxLayout(box)
        self.ask_calibration = QtWidgets.QCheckBox(
            "Before PREDICT / START NN, ask whether to calibrate the hand")
        self.ask_calibration.setChecked(bool(cfg.ASK_CALIBRATION_BEFORE_PREDICT))
        self.ask_calibration.setToolTip(
            "The question offers to reuse the last calibration only when the "
            "application was restarted while the band stayed on the arm.")
        v.addWidget(self.ask_calibration)
        self.ask_retrain = QtWidgets.QCheckBox(
            "When new gestures or new recordings make models out of date, "
            "ask whether to retrain (all models or only the selected)")
        self.ask_retrain.setChecked(bool(cfg.ASK_RETRAIN_ON_NEW_DATA))
        self.ask_retrain.setToolTip(
            "Shown when the application opens; turn it off to stop being "
            "asked. RETRAIN stays available in the TRAINING panel.")
        v.addWidget(self.ask_retrain)
        layout.addWidget(box)
        layout.addStretch(1)

    def values(self) -> dict:
        return {"ASK_CALIBRATION_BEFORE_PREDICT":
                self.ask_calibration.isChecked(),
                "ASK_RETRAIN_ON_NEW_DATA": self.ask_retrain.isChecked()}
