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
from emg_app.gui.qt import QtCore, QtGui, QtWidgets
from emg_app.gui.tokens import (
    COL_ACCENT, COL_BAD, COL_TEXT, COL_TEXT_DIM, COL_WARN)
from emg_app.gui.widgets import fit_table_height, make_label, wrap_label

STAR = "★"
BLUE_STAR = "#2a78d6"       # the project's ML blue
YELLOW_STAR = "#c9a227"     # the default star
EXISTS_ROLE = QtCore.Qt.UserRole + 1     # does the model file exist

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


def training_recommendations() -> dict:
    """
    What the ablation study recommends for each family's training switches:
    {"nn": {"augment": [...], "adapt": bool, "reasons": [...]}, "ml": {...}}.

    The values come from the measured decisions in results/experiments.json
    (tools/collect_results.py builds it) so the reasons are the real ones; if
    the file is missing the shipped defaults are used and the reason says so.
    """
    from emg_app import settings_store
    out = {}
    for fam in ("nn", "ml"):
        out[fam] = {
            "augment": list(settings_store.DEFAULTS[f"TRAIN_AUGMENT_{fam.upper()}"]),
            "adapt": bool(settings_store.DEFAULTS[f"TRAIN_ADAPT_{fam.upper()}"]),
            "reasons": [], "measured": False}
    try:
        import json
        import os
        with open(os.path.join(cfg.PROJECT_DIR, "results", "experiments.json"),
                  encoding="utf-8") as fh:
            doc = json.load(fh)
        for d in doc.get("defaults", []):
            rec = out.get(d.get("family"))
            if rec is None:
                continue
            if d.get("setting") == "augmentation":
                rec["augment"] = list(d.get("value") or [])
            elif d.get("setting") == "adaptation":
                rec["adapt"] = bool(d.get("value"))
            rec["reasons"].append(f"{d['setting']}: {d.get('reason', '')}")
            rec["measured"] = True
    except (OSError, ValueError, KeyError, TypeError):
        for rec in out.values():
            rec["reasons"] = ["the shipped defaults (results/experiments.json "
                              "was not found, so the reasons are not shown)"]
    return out


class ContainedTable(QtWidgets.QTableWidget):
    """
    A table inside a scrolling page that keeps the mouse wheel for itself.

    Qt hands a wheel event to the parent as soon as the table is at the end of
    its range (or not scrollable), so the whole settings page moved when the
    list was only meant to. This consumes the wheel whenever the table has
    something to scroll; a table that fits entirely lets the page have it.
    """

    def wheelEvent(self, event) -> None:
        if self.verticalScrollBar().maximum() > 0:
            super().wheelEvent(event)
            event.accept()
        else:
            event.ignore()


class ModelsTab(QtWidgets.QWidget):
    """Choose the models the live classifiers use."""

    VISIBLE_ROWS = 8          # rows a model list shows before it scrolls itself

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
        layout.addWidget(wrap_label(
            f"A blue {STAR} marks a model trained WITHOUT somebody that scores "
            f"HIGHER than its yellow {STAR} on the same held-out people. It is "
            "worked out live from the scores, so it follows every retraining. "
            "Yellow stays the default; the hover text says by how much (often "
            "a fraction of a point). The models' own headline scores cover "
            "fewer people and are not what is compared. "
            "See docs/EXCLUDING_PEOPLE.md.", 10, False, COL_TEXT_DIM))

        for family, title in (("ml", "ML classifiers"),
                              ("nn", "Neural nets")):
            layout.addWidget(self._family_box(family, title), 1)
        self._building = False
        self._refresh()

    # -- building -------------------------------------------------------------

    def _family_box(self, family: str, title: str) -> QtWidgets.QGroupBox:
        box = QtWidgets.QGroupBox(title)
        v = QtWidgets.QVBoxLayout(box)
        #  Ordinary models first. A model trained without someone is scored on
        #  fewer (and easier) people, so its number must not sort it to the top.
        mine = [e for e in self.entries if e.family == family]
        rows = (zoo.rank([e for e in mine if not e.excluded])
                + zoo.rank([e for e in mine if e.excluded]))
        table = ContainedTable(len(rows), len(self.COLS))
        table.setHorizontalHeaderLabels(self.COLS)
        table.verticalHeader().setVisible(False)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        #  A fixed height, so the list scrolls inside itself (and the page
        #  around it has a predictable length) instead of stretching.
        fit_table_height(table, min(max(len(rows), 1), self.VISIBLE_ROWS))
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
            use.setData(EXISTS_ROLE, bool(e.exists))
            use.setToolTip("Use this model in the live classifier")
            if not e.exists:
                use.setFlags(QtCore.Qt.NoItemFlags)
                use.setToolTip("the model file is missing")
            table.setItem(r, 0, use)
            blue = e.id in self.stars.get("variants", {})
            star = QtWidgets.QTableWidgetItem(
                STAR if (e.id in self.stars["models"] or blue) else "")
            star.setForeground(QtGui.QColor(BLUE_STAR if blue
                                            else YELLOW_STAR))
            if e.id in self.stars["models"]:
                star.setToolTip(self.stars["models"][e.id])
            elif blue:
                star.setToolTip(self.stars["variants"][e.id])
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
                         "adaptation" if tw.get("adapt") else "",
                         "without " + ", ".join(e.excluded)
                         if e.excluded else ""])) or "plain"]
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
        yellow = self._star_button(
            f"Yellow {STAR}", YELLOW_STAR,
            "Tick exactly the yellow-starred models of this family (the top "
            f"{zoo.STAR_TOP_N} by accuracy with a person held out).",
            lambda: self._select_stars(family))
        n_blue = len(self._blue_ids(family))
        blue = self._star_button(
            f"Blue {STAR}", BLUE_STAR,
            "Tick every model of this family that scores higher than its "
            "yellow star on the same held-out people (worked out live; see "
            "the hover text of a blue star)." if n_blue else
            "No model of this family has a blue star yet: none trained "
            "without somebody scores higher than its yellow star.",
            lambda: self._select_blue(family), enabled=bool(n_blue))
        both = self._star_button(
            f"Yellow + blue {STAR}", None,
            "Tick the yellow-starred models and the blue-starred ones.",
            lambda: self._select_both(family), enabled=bool(n_blue))
        best = QtWidgets.QPushButton(f"Best combination {STAR}")
        best.setToolTip("Tick the members of the best soft-voting subset "
                        "(only offered when it beats every single model)")
        best.setEnabled(family in self.stars["ensembles"])
        if family in self.stars["ensembles"]:
            best.setToolTip(self.stars["ensembles"][family])
        best.clicked.connect(lambda: self._select_combo(family))
        clear = QtWidgets.QPushButton("Clear")
        clear.clicked.connect(lambda: self._set_checked(family, []))
        for b in (yellow, blue, both, best, clear):
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
            if item.data(EXISTS_ROLE):
                item.setCheckState(QtCore.Qt.Checked
                                   if item.data(QtCore.Qt.UserRole) in ids
                                   else QtCore.Qt.Unchecked)
        self._building = False
        self._refresh()

    @staticmethod
    def _star_button(text, colour, tip, slot, enabled=True):
        """A quick-selector button whose text has the colour of its star."""
        button = QtWidgets.QPushButton(text)
        if colour:
            button.setStyleSheet(f"QPushButton {{ color: {colour}; "
                                 "font-weight: bold; }"
                                 "QPushButton:disabled { color: gray; }")
        button.setToolTip(tip)
        button.setEnabled(enabled)
        button.clicked.connect(lambda _=False: slot())
        return button

    def _blue_ids(self, family: str) -> List[str]:
        """Ids of the blue-starred models of a family, from the live stars."""
        by_id = {e.id: e for e in self.entries}
        return [i for i in self.stars.get("variants", {})
                if i in by_id and by_id[i].family == family
                and by_id[i].exists]

    def _yellow_ids(self, family: str) -> List[str]:
        return [e.id for e in self.entries
                if e.family == family and e.id in self.stars["models"]]

    def _quick(self, family: str, ids: List[str]) -> None:
        """Tick `ids` exactly; choosing models means to use them, so the
        'use the selected models' switch goes on (APPLY would ignore the
        ticks otherwise)."""
        if ids and not self.enable_box.isChecked():
            self.enable_box.setChecked(True)
        self._set_checked(family, ids)

    def _select_stars(self, family: str) -> None:
        """The yellow stars."""
        self._quick(family, self._yellow_ids(family))

    def _select_blue(self, family: str) -> None:
        self._quick(family, self._blue_ids(family))

    def _select_both(self, family: str) -> None:
        self._quick(family, self._yellow_ids(family) + self._blue_ids(family))

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

    def _lock(self, family: str, on: bool) -> None:
        """
        Switch the tick boxes of a family on or off WITHOUT disabling the
        table: a disabled widget gets no wheel events, so the list could not
        be scrolled while the selection was off (the default) and the mouse
        wheel moved the whole page instead.
        """
        table = self.tables[family]
        #  Changing a flag emits itemChanged, which would call _refresh again.
        was, self._building = self._building, True
        try:
            for r in range(table.rowCount()):
                item = table.item(r, 0)
                if not item.data(EXISTS_ROLE):
                    continue
                flags = QtCore.Qt.ItemIsEnabled
                if on:
                    flags |= QtCore.Qt.ItemIsUserCheckable
                item.setFlags(flags)
        finally:
            self._building = was
        table.setStyleSheet("" if on else f"QTableWidget {{ color: "
                                          f"{COL_TEXT_DIM}; }}")

    def _refresh(self, *_args) -> None:
        if self._building:
            return
        on = self.enable_box.isChecked()
        for family in ("ml", "nn"):
            self._lock(family, on)
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
        self.recommendations = training_recommendations()
        self.recommend_buttons = {}
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
            adapt = QtWidgets.QCheckBox("Train with adaptation")
            adapt.setChecked(bool(current_adapt))
            adapt.setToolTip(
                "Each person's signal is standardised with statistics from "
                "about 9 s of their own unlabelled signal -- the length of a "
                "calibration sweep.")
            self.adapt_boxes[fam] = adapt
            v.addWidget(adapt)

            rec = self.recommendations[fam.lower()]
            v.addWidget(make_label("Recommended", 10, True, COL_ACCENT))
            v.addWidget(wrap_label(
                f"augmentation {', '.join(rec['augment']) or 'none'}; "
                f"adaptation {'on' if rec['adapt'] else 'off'}", 10, True,
                COL_TEXT))
            v.addWidget(wrap_label("\n".join(rec["reasons"]), 9, False,
                                   COL_TEXT_DIM))
            use = QtWidgets.QPushButton("Use recommended")
            use.setToolTip(
                f"Set the {title} switches above to the measured "
                "recommendation. Applies on APPLY; train again for it to "
                "take effect.")
            use.clicked.connect(lambda _=False, f=fam: self.use_recommended(f))
            self.recommend_buttons[fam] = use
            v.addWidget(use)
            v.addStretch(1)
            columns.addWidget(box, 1)
        layout.addLayout(columns)

        layout.addWidget(self._exclusion_box())

        self.adapt_live = QtWidgets.QCheckBox("Use adaptation live")
        self.adapt_live.setChecked(bool(cfg.ADAPT_LIVE))
        self.adapt_live.setToolTip(
            "Only models that were trained with adaptation; off = they use "
            "their stored global statistics.")
        layout.addWidget(self.adapt_live)
        layout.addWidget(wrap_label(
            "Only models that were trained with adaptation use it; off = "
            "they use their stored global statistics.", 10, False,
            COL_TEXT_DIM))
        layout.addWidget(wrap_label(
            "The person's statistics come from the calibration: the signal "
            "captured while the hand is calibrated is stored with it. The "
            "ML switches apply to models trained by the model zoo "
            "(training/train_zoo.py); the classic RETRAIN sweep trains "
            "plain models.", 10, False, COL_TEXT_DIM))
        layout.addStretch(1)

    # -- train without some people ---------------------------------------------

    @staticmethod
    def _held_out_scores() -> dict:
        """`{family: {person: window accuracy}}` of the best ordinary model of
        each family: how well a model that never saw that person reads them."""
        out = {}
        entries = [e for e in zoo.list_entries() if not e.excluded]
        for fam in zoo.FAMILIES:
            ranked = [e for e in zoo.rank([e for e in entries
                                           if e.family == fam and e.exists])
                      if e.window_acc is not None]
            if ranked:
                out[fam] = dict(ranked[0].scores.get("per_person") or {})
        return out

    def _exclusion_box(self) -> QtWidgets.QGroupBox:
        from emg_app import settings_store
        try:
            people = settings_store.people_on_disk()
        except OSError:
            people = []
        box = QtWidgets.QGroupBox("Train without these people")
        v = QtWidgets.QVBoxLayout(box)
        v.addWidget(wrap_label(
            "Leaves a person's recordings out of the NEXT training (RETRAIN, "
            "TRAIN NEURAL NET and the model zoo). Nothing on disk changes and "
            "a trained model keeps what it learned. The zoo saves these as "
            "separate '<model> (without <name>)' entries beside the ordinary "
            "ones, so both can be compared. The percentages are how well the "
            "best ordinary ML model / neural net reads that person when they "
            "were held out of training -- a low number means the other people "
            "do not cover how that person wears the band, so including them "
            "may cost the rest. Measured: docs/EXCLUDING_PEOPLE.md.",
            10, False, COL_TEXT_DIM))
        scores = self._held_out_scores()
        self.exclude_boxes = {}
        for person in people:
            held = ", ".join(
                f"{fam.upper()} {scores[fam][person] * 100:.0f} %"
                for fam in zoo.FAMILIES
                if person in scores.get(fam, {}))
            check = QtWidgets.QCheckBox(
                person + (f"   (held out: {held})" if held else ""))
            check.setChecked(person in cfg.TRAIN_EXCLUDE_PERSONS)
            check.setToolTip(f"Train without {person}'s recordings")
            self.exclude_boxes[person] = check
            v.addWidget(check)
        if not people:
            v.addWidget(wrap_label("No recordings found.", 10, False,
                                   COL_WARN))
        self.exclude_note = wrap_label("", 10, False, COL_BAD)
        v.addWidget(self.exclude_note)
        return box

    def problems(self) -> List[str]:
        """Why the current choice cannot be applied (empty = fine)."""
        left = [p for p, b in self.exclude_boxes.items() if not b.isChecked()]
        if self.exclude_boxes and len(left) < 2:
            return ["Train without these people: at least two people must "
                    "stay in, otherwise there is nobody to hold out."]
        return []

    def use_recommended(self, fam: str) -> None:
        """Tick one family's switches to the measured recommendation."""
        rec = self.recommendations[fam.lower()]
        for name, box in self.aug_boxes[fam].items():
            box.setChecked(name in rec["augment"])
        self.adapt_boxes[fam].setChecked(rec["adapt"])

    def values(self) -> dict:
        return {
            "TRAIN_AUGMENT_NN": [n for n, b in self.aug_boxes["NN"].items()
                                 if b.isChecked()],
            "TRAIN_AUGMENT_ML": [n for n, b in self.aug_boxes["ML"].items()
                                 if b.isChecked()],
            "TRAIN_ADAPT_NN": self.adapt_boxes["NN"].isChecked(),
            "TRAIN_ADAPT_ML": self.adapt_boxes["ML"].isChecked(),
            "TRAIN_EXCLUDE_PERSONS": [p for p, b in self.exclude_boxes.items()
                                      if b.isChecked()],
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
            "Ask to calibrate before PREDICT / START NN")
        self.ask_calibration.setChecked(bool(cfg.ASK_CALIBRATION_BEFORE_PREDICT))
        self.ask_calibration.setToolTip(
            "The question offers to reuse the last calibration only when the "
            "application was restarted while the band stayed on the arm.")
        v.addWidget(self.ask_calibration)
        v.addWidget(wrap_label(
            "Asks whether to calibrate the hand first; offers the last "
            "calibration only if the band stayed on the arm.", 10, False,
            COL_TEXT_DIM))
        self.ask_retrain = QtWidgets.QCheckBox(
            "Ask to retrain when the models are out of date")
        self.ask_retrain.setChecked(bool(cfg.ASK_RETRAIN_ON_NEW_DATA))
        self.ask_retrain.setToolTip(
            "Shown when the application opens; turn it off to stop being "
            "asked. RETRAIN stays available in the TRAINING panel.")
        v.addWidget(self.ask_retrain)
        v.addWidget(wrap_label(
            "When new gestures or recordings make models out of date: "
            "retrain all models or only the selected.", 10, False,
            COL_TEXT_DIM))
        layout.addWidget(box)
        look = QtWidgets.QGroupBox("Appearance")
        lv = QtWidgets.QVBoxLayout(look)
        self.icon_box = QtWidgets.QCheckBox("Show the application icon")
        self.icon_box.setChecked(bool(cfg.APP_ICON_ENABLED))
        self.icon_box.setToolTip(
            "Off: the toolkit's default icon. Takes effect the next time the "
            "application starts.")
        lv.addWidget(self.icon_box)
        layout.addWidget(look)
        layout.addStretch(1)

    def values(self) -> dict:
        return {"ASK_CALIBRATION_BEFORE_PREDICT":
                self.ask_calibration.isChecked(),
                "ASK_RETRAIN_ON_NEW_DATA": self.ask_retrain.isChecked(),
                "APP_ICON_ENABLED": self.icon_box.isChecked()}
