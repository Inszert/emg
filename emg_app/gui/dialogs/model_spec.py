"""
MODEL SPEC (model_spec.json).
"""

import os

from emg_app import config as cfg
from emg_app.gui.tokens import (
    COL_ACCENT, COL_BAD, COL_GOOD, COL_TEXT, COL_TEXT_DIM, COL_WARN, FS_LG,
    FS_XL, H_CONTROL)
from emg_app.calibration import CALIBRATION_MODES
from emg_app.model_spec import (
    DATA_CALIBRATION_DEFAULT, default_data_selection, default_model_spec,
    expand_auto_spec, model_param_default, model_search_space,
    save_model_spec, validate_model_spec)
from emg_app.dataset import dataset_person_sessions
from emg_app.gui.qt import QtCore, QtGui, QtWidgets
from emg_app.gui.widgets import fit_to_screen, make_label, scroll_dialog_body


class ModelSpecDialog(QtWidgets.QDialog):
    """
    Which classifiers the sweep trains, with what hyperparameters, and whether
    they also vote as an ensemble.

    The editor is generated from `MODEL_KINDS`, so adding a model kind or a
    hyperparameter to that schema is all it takes to make it appear here --
    there is no second list of widgets to keep in step.

    This dialog changes what gets TRAINED, never what gets reported. Every
    score still comes from the same cross-validation in train_model.py, so
    switching a model on cannot flatter it.
    """

    def __init__(self, spec: dict, spec_path: str, parent=None):
        super().__init__(parent)
        self.spec_path = spec_path
        self.spec, problems = validate_model_spec(spec)
        self.saved = False
        self._loading = False
        self._param_widgets = {}

        self.setWindowTitle("Model specification")
        fit_to_screen(self, 1180, 860, min_width=760, min_height=520)

        root = QtWidgets.QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)

        root.addWidget(make_label(
            "Which classifiers the sweep trains, and how they vote",
            FS_XL, True, COL_ACCENT))
        intro = make_label(
            "Every enabled model is trained and scored with the same honest "
            "protocol, and the best one is saved. Turning a model on does not "
            "make it good -- the cross-validation decides that.",
            11, False, COL_TEXT_DIM)
        intro.setWordWrap(True)
        root.addWidget(intro)

        body = QtWidgets.QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body, 1)

        # ---- left: the roster ------------------------------------------------
        left = QtWidgets.QVBoxLayout()
        left.addWidget(make_label("MODELS", 10, True, COL_TEXT_DIM))

        self.model_list = QtWidgets.QListWidget()
        self.model_list.setMinimumWidth(320)
        # Deliberately no minimum height: a minimum taller than the space the
        # column actually has makes the list paint straight over the buttons
        # underneath it. It takes the stretch instead, so it is as tall as the
        # dialog allows and no taller.
        self.model_list.setSizePolicy(QtWidgets.QSizePolicy.Preferred,
                                      QtWidgets.QSizePolicy.Expanding)
        # A row per model, not four rows and a scrollbar: choosing what to
        # train means comparing the list, which means seeing the list.
        self.model_list.setHorizontalScrollBarPolicy(
            QtCore.Qt.ScrollBarAlwaysOff)
        self.model_list.setTextElideMode(QtCore.Qt.ElideRight)
        self.model_list.currentRowChanged.connect(self._on_row_changed)
        self.model_list.itemChanged.connect(self._on_item_changed)
        left.addWidget(self.model_list, 1)

        self.roster_info = make_label("", 10, False, COL_TEXT_DIM)
        self.roster_info.setWordWrap(True)
        left.addWidget(self.roster_info)

        row = QtWidgets.QHBoxLayout()
        all_on = QtWidgets.QPushButton("ALL ON")
        all_on.setMinimumHeight(H_CONTROL)
        all_on.setToolTip("Train every model. Slower, and the sweep still "
                          "picks one winner.")
        all_on.clicked.connect(lambda: self._set_all(True))
        row.addWidget(all_on)
        all_off = QtWidgets.QPushButton("ALL OFF")
        all_off.setMinimumHeight(H_CONTROL)
        all_off.clicked.connect(lambda: self._set_all(False))
        row.addWidget(all_off)
        left.addLayout(row)

        row_b = QtWidgets.QHBoxLayout()
        self.add_combo = QtWidgets.QComboBox()
        for kind, schema in cfg.MODEL_KINDS.items():
            self.add_combo.addItem(schema["label"], kind)
        self.add_combo.setToolTip("Add another candidate of this kind. You can "
                                  "have several of the same kind with "
                                  "different hyperparameters.")
        row_b.addWidget(self.add_combo, 1)
        add = QtWidgets.QPushButton("ADD")
        add.clicked.connect(self.add_model)
        row_b.addWidget(add)
        left.addLayout(row_b)

        remove = QtWidgets.QPushButton("REMOVE SELECTED")
        remove.setObjectName("danger")
        remove.setMinimumHeight(H_CONTROL)
        remove.clicked.connect(self.remove_model)
        left.addWidget(remove)

        body.addLayout(left, 0)

        # ---- right: the selected model's settings ----------------------------
        right = QtWidgets.QVBoxLayout()
        self.detail_title = make_label("", FS_LG, True, COL_ACCENT)
        self.detail_title.setWordWrap(True)
        right.addWidget(self.detail_title)

        self.detail_note = make_label("", 10, False, COL_TEXT_DIM)
        self.detail_note.setWordWrap(True)
        right.addWidget(self.detail_note)

        form_box = QtWidgets.QGroupBox("HYPERPARAMETERS")
        self.form_host = QtWidgets.QVBoxLayout(form_box)
        self.form_host.setSpacing(6)
        self.form_host.setContentsMargins(10, 14, 10, 10)
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)
        scroll.setWidget(form_box)
        right.addWidget(scroll, 1)

        body.addLayout(right, 2)

        # ---- voting ----------------------------------------------------------
        vote_box = QtWidgets.QGroupBox("VOTING ENSEMBLE")
        vote_box.setMaximumHeight(150)
        vote = QtWidgets.QVBoxLayout(vote_box)
        vote.setSpacing(5)

        self.vote_check = QtWidgets.QCheckBox(
            "also build an ensemble that votes over the enabled models")
        self.vote_check.setToolTip(
            "The ensemble is added as one more candidate, so the sweep scores "
            "it against its own members.")
        self.vote_check.toggled.connect(self._on_voting_changed)
        vote.addWidget(self.vote_check)

        vrow = QtWidgets.QHBoxLayout()
        vrow.addWidget(make_label("Vote by", 11, False, COL_TEXT_DIM))
        self.vote_mode = QtWidgets.QComboBox()
        self.vote_mode.addItem("soft  -- average the predicted probabilities",
                               "soft")
        self.vote_mode.addItem("hard  -- count the votes", "hard")
        self.vote_mode.setToolTip(
            "Soft voting usually wins, but it needs predict_proba from every "
            "member; any member that cannot is left out of the ensemble.")
        self.vote_mode.currentIndexChanged.connect(self._on_voting_changed)
        vrow.addWidget(self.vote_mode, 1)
        vote.addLayout(vrow)

        self.vote_always = QtWidgets.QCheckBox(
            "always save the ensemble, even if a single model scored higher")
        self.vote_always.setToolTip(
            "Off (recommended): the ensemble competes and the best score "
            "wins. On: the ensemble is used regardless -- a deliberate "
            "choice, not an improvement.")
        self.vote_always.toggled.connect(self._on_voting_changed)
        vote.addWidget(self.vote_always)

        self.vote_info = make_label("", 10, False, COL_TEXT_DIM)
        self.vote_info.setWordWrap(True)
        vote.addWidget(self.vote_info)
        root.addWidget(vote_box)

        # ---- auto mode -------------------------------------------------------
        auto_box = QtWidgets.QGroupBox("AUTO MODE  (try combinations)")
        auto_box.setMaximumHeight(160)
        auto = QtWidgets.QVBoxLayout(auto_box)
        auto.setSpacing(5)

        self.auto_check = QtWidgets.QCheckBox(
            "try hyperparameter combinations for each model and keep the best")
        self.auto_check.setToolTip(
            "Every combination becomes its own candidate and is scored by the "
            "same cross-validation, so they all appear in the results table. "
            "This multiplies the training time."
        )
        self.auto_check.toggled.connect(self._on_auto_changed)
        auto.addWidget(self.auto_check)

        arow = QtWidgets.QHBoxLayout()
        arow.addWidget(make_label("Max combinations per model", 11, False,
                                  COL_TEXT_DIM))
        self.auto_spin = QtWidgets.QSpinBox()
        self.auto_spin.setRange(1, 64)
        self.auto_spin.setToolTip(
            "The full grid is sampled down to this many, seeded so a rerun "
            "gives the same candidates. Your configured values are always "
            "combination #1, so AUTO can never lose the setting you had."
        )
        self.auto_spin.valueChanged.connect(self._on_auto_changed)
        arow.addWidget(self.auto_spin, 1)
        auto.addLayout(arow)

        self.auto_info = make_label("", 10, False, COL_TEXT_DIM)
        self.auto_info.setWordWrap(True)
        auto.addWidget(self.auto_info)

        self.auto_warning = make_label("", 10, True, COL_WARN)
        self.auto_warning.setWordWrap(True)
        auto.addWidget(self.auto_warning)
        root.addWidget(auto_box)

        # ---- training data ---------------------------------------------------
        data_box = QtWidgets.QGroupBox("TRAINING DATA")
        data = QtWidgets.QVBoxLayout(data_box)
        data.setSpacing(5)

        self.data_list = QtWidgets.QListWidget()
        self.data_list.setMaximumHeight(120)
        self.data_list.setToolTip(
            "Tick the subject/session folders to train on. Ticking all of "
            "them, or none, means everything on disk -- including sessions "
            "recorded later.")
        self.data_list.itemChanged.connect(self._on_data_changed)
        data.addWidget(self.data_list)

        drow = QtWidgets.QHBoxLayout()
        drow.addWidget(make_label("Calibration for training", 11, False,
                                  COL_TEXT_DIM))
        self.cal_combo = QtWidgets.QComboBox()
        self.cal_combo.addItem(
            f"follow the app setting ({cfg.CALIBRATION_MODE})",
            DATA_CALIBRATION_DEFAULT)
        for mode in CALIBRATION_MODES:
            self.cal_combo.addItem(
                "off  -- no calibration; any wearer can use the model"
                if mode == "off" else mode, mode)
        self.cal_combo.setToolTip(
            "Whether each wearer's stored calibration is applied to their "
            "training data. A calibrated model refuses to run for a wearer "
            "who has not calibrated; 'off' trains on raw device counts and "
            "has no such requirement, at the cost of cross-wearer accuracy.")
        self.cal_combo.currentIndexChanged.connect(self._on_data_changed)
        drow.addWidget(self.cal_combo, 1)
        data.addLayout(drow)

        self.data_info = make_label("", 10, False, COL_TEXT_DIM)
        self.data_info.setWordWrap(True)
        data.addWidget(self.data_info)
        root.addWidget(data_box)

        # ---- footer ----------------------------------------------------------
        self.problem_label = make_label("", 10, False, COL_WARN)
        self.problem_label.setWordWrap(True)
        root.addWidget(self.problem_label)

        where = make_label(
            f"Saved to {spec_path}  --  read by train_model.py on the next "
            f"retrain, and copied into the model it produces.",
            10, False, COL_TEXT_DIM)
        where.setWordWrap(True)
        root.addWidget(where)

        buttons = QtWidgets.QHBoxLayout()
        reset = QtWidgets.QPushButton("RESET TO DEFAULTS")
        reset.clicked.connect(self.reset_defaults)
        buttons.addWidget(reset)
        buttons.addStretch(1)
        cancel = QtWidgets.QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        save = QtWidgets.QPushButton("SAVE SPEC")
        save.setObjectName("primary")
        save.setDefault(True)
        save.clicked.connect(self.save_spec)
        buttons.addWidget(save)
        root.addLayout(buttons)

        #  ~700 px of minimum content: more than a 1080p screen has at 150 %
        #  scaling. Scroll the body, keep SAVE pinned, and size again.
        scroll_dialog_body(self, keep_last=1)
        fit_to_screen(self, 1180, 860, min_width=760, min_height=520)

        self._reload_all(problems)

    # -- population -----------------------------------------------------------

    def _reload_all(self, problems=None) -> None:
        self._loading = True
        try:
            self.model_list.clear()
            for entry in self.spec["models"]:
                item = QtWidgets.QListWidgetItem(self._row_text(entry))
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked if entry["enabled"]
                                   else QtCore.Qt.Unchecked)
                item.setForeground(QtGui.QColor(
                    COL_TEXT if entry["enabled"] else COL_TEXT_DIM))
                self.model_list.addItem(item)

            vote = self.spec["voting"]
            self.vote_check.setChecked(bool(vote["enabled"]))
            index = self.vote_mode.findData(vote["mode"])
            self.vote_mode.setCurrentIndex(max(0, index))
            self.vote_always.setChecked(bool(vote["always"]))

            auto = self.spec.get("auto") or {}
            self.auto_check.setChecked(bool(auto.get("enabled")))
            self.auto_spin.setValue(int(auto.get(
                "max_per_model", cfg.MODEL_AUTO_MAX_PER_MODEL)))

            selection = self.spec.get("data") or default_data_selection()
            chosen = set(selection["include"])
            try:
                available = dataset_person_sessions(
                    os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER))
            except OSError:
                available = {}
            # A pair named in the spec but gone from disk stays visible, so it
            # is not silently dropped on the next save.
            for key in sorted(chosen - set(available)):
                available[key] = 0
            self.data_list.clear()
            for key, count in sorted(available.items()):
                item = QtWidgets.QListWidgetItem(
                    f"{key}      {count} recording(s)" if count
                    else f"{key}      not on disk")
                item.setData(QtCore.Qt.UserRole, key)
                item.setFlags(item.flags() | QtCore.Qt.ItemIsUserCheckable)
                item.setCheckState(QtCore.Qt.Checked
                                   if not chosen or key in chosen
                                   else QtCore.Qt.Unchecked)
                self.data_list.addItem(item)
            self.cal_combo.setCurrentIndex(max(0, self.cal_combo.findData(
                selection["calibration"])))
        finally:
            self._loading = False

        if self.model_list.count():
            self.model_list.setCurrentRow(0)
        self._refresh_summary(problems)

    @staticmethod
    def _row_text(entry: dict) -> str:
        kind = cfg.MODEL_KINDS.get(entry["kind"], {})
        scaler = entry.get("scaler", "none")
        return (f"{entry['name']}      [{kind.get('label', entry['kind'])}"
                + (f", {scaler} scaler]" if scaler != "none" else "]"))

    def _refresh_summary(self, problems=None) -> None:
        on = [m for m in self.spec["models"] if m["enabled"]]
        self.roster_info.setText(
            f"{len(on)} of {len(self.spec['models'])} will be trained"
            + ("   --   nothing enabled!" if not on else ""))
        self.roster_info.setStyleSheet(
            f"color: {COL_BAD if not on else COL_GOOD}; background: transparent;")

        vote = self.spec["voting"]
        if not vote["enabled"]:
            self.vote_info.setText("no ensemble -- each model stands alone")
        elif len(on) < 2:
            self.vote_info.setText(
                "at least two enabled models are needed for a vote; "
                "the ensemble will be skipped")
        else:
            self.vote_info.setText(
                f"Vote-{vote['mode']}-{len(on)} over "
                + ", ".join(m["name"] for m in on))

        self._refresh_auto_info()

        selection = self.spec.get("data") or default_data_selection()
        include, cal = selection["include"], selection["calibration"]
        who = (f"{len(include)} subject/session folder(s)" if include
               else "everything on disk")
        if cal == "off":
            cal_text = ("calibration OFF -- the model needs no calibration "
                        "to run for any wearer")
        elif cal == DATA_CALIBRATION_DEFAULT:
            cal_text = f"calibration follows the app setting ({cfg.CALIBRATION_MODE})"
        else:
            cal_text = (f"calibration '{cal}' -- every wearer must calibrate "
                        "before the model will run for them")
        self.data_info.setText(f"Training on {who}; {cal_text}.")

        if problems:
            self.problem_label.setText("spec was corrected: "
                                       + "; ".join(problems[:4])
                                       + (" ..." if len(problems) > 4 else ""))
        else:
            self.problem_label.setText("")

    def _refresh_auto_info(self) -> None:
        """
        What AUTO mode will actually produce, and what that costs.

        Shown as a count rather than a promise: the number of candidates is
        the number of full cross-validations the next retrain will run, which
        is the thing that decides whether it takes a minute or an hour.
        """
        auto = self.spec.get("auto") or {}
        if not auto.get("enabled"):
            self.auto_info.setText(
                "off -- each model is trained once, with the values set above")
            self.auto_warning.setText("")
            return

        expanded, note = expand_auto_spec(self.spec, seed=0)
        candidates = [m for m in expanded["models"] if m["enabled"]]
        plain = [m for m in self.spec["models"] if m["enabled"]]
        tuned = sorted({m.get("auto_from") for m in candidates
                        if m.get("auto_from")})

        self.auto_info.setText(
            f"{len(candidates)} candidates from {len(plain)} enabled model(s)"
            + (f"; tuning {', '.join(tuned)}" if tuned else "")
            + f"   --   roughly {len(candidates) / max(1, len(plain)):.1f}x "
              "the training time of one pass"
        )
        self.auto_warning.setText(
            "Honest-result warning: picking the best of "
            f"{len(candidates)} candidates on the same folds makes the "
            "WINNER's score optimistic -- the more combinations, the more "
            "optimistic. Trust the leave-one-subject-out number in MODEL "
            "STATS, not the sweep's best row."
        )

    # -- the per-model form ---------------------------------------------------

    def _on_row_changed(self, row: int) -> None:
        while self.form_host.count():
            item = self.form_host.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
        self._param_widgets = {}

        if not (0 <= row < len(self.spec["models"])):
            self.detail_title.setText("")
            self.detail_note.setText("")
            return

        entry = self.spec["models"][row]
        schema = cfg.MODEL_KINDS.get(entry["kind"], {})
        self.detail_title.setText(
            f"{entry['name']}   --   {schema.get('label', entry['kind'])}")
        self.detail_note.setText(schema.get("note", ""))

        grid = QtWidgets.QWidget()
        form = QtWidgets.QFormLayout(grid)
        form.setLabelAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        form.setFormAlignment(QtCore.Qt.AlignLeft | QtCore.Qt.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(7)
        form.setContentsMargins(4, 4, 4, 4)
        # Let a long hint wrap instead of forcing the dialog wider.
        form.setRowWrapPolicy(QtWidgets.QFormLayout.DontWrapRows)
        form.setFieldGrowthPolicy(QtWidgets.QFormLayout.ExpandingFieldsGrow)

        name_edit = QtWidgets.QLineEdit(entry["name"])
        name_edit.setToolTip("How this candidate is named in the sweep "
                             "results and the saved report.")
        name_edit.editingFinished.connect(
            lambda: self._set_name(row, name_edit.text()))
        form.addRow(self._form_label("name"), name_edit)

        scaler_combo = QtWidgets.QComboBox()
        scaler_combo.addItems(list(cfg.MODEL_SCALERS))
        scaler_combo.setCurrentText(entry.get("scaler", "none"))
        scaler_combo.setToolTip(
            "Feature scaling before the classifier. Tree models do not need "
            "it; SVM, kNN, logistic regression and MLP very much do.")
        scaler_combo.currentTextChanged.connect(
            lambda text: self._set_scaler(row, text))
        form.addRow(self._form_label("scaler"), scaler_combo)

        space = model_search_space(entry["kind"])
        auto_box = QtWidgets.QCheckBox(
            f"include in AUTO mode ({len(space)} parameter(s) to try)"
            if space else "no search space for this kind")
        auto_box.setChecked(bool(entry.get("auto", True)) and bool(space))
        auto_box.setEnabled(bool(space))
        auto_box.setToolTip(
            "When AUTO MODE is on, generate combinations for this model. "
            "Switch it off to pin this one to the values below while the "
            "others are tuned."
            if space else
            "AUTO mode has nothing to vary for this kind.")
        auto_box.toggled.connect(
            lambda on: self._set_auto_for(row, on))
        form.addRow(self._form_label("auto-tune"), auto_box)
        if space:
            tried = make_label(
                "tries " + ";  ".join(
                    f"{param} = {', '.join(str(v) for v in values)}"
                    for param, values in sorted(space.items())),
                9, False, COL_TEXT_DIM)
            tried.setWordWrap(True)
            form.addRow("", tried)

        for param, rule in schema.get("params", {}).items():
            widget = self._build_param_widget(row, param, rule,
                                              entry["params"].get(param))
            form.addRow(self._form_label(param), widget)
            # The explanation goes on its own line. Putting it in the label --
            # which is what this did at first -- makes the label column as
            # wide as the longest sentence, and every row then collides with
            # the next one.
            if rule.get("note"):
                hint = make_label(rule["note"], 9, False, COL_TEXT_DIM)
                hint.setWordWrap(True)
                form.addRow("", hint)
            self._param_widgets[param] = widget

        self.form_host.addWidget(grid)
        self.form_host.addStretch(1)

    @staticmethod
    def _form_label(text: str):
        """A fixed-width field label, so every row lines up."""
        label = make_label(text, 11, False, COL_TEXT)
        label.setAlignment(QtCore.Qt.AlignRight | QtCore.Qt.AlignVCenter)
        label.setMinimumWidth(130)
        return label

    def _build_param_widget(self, row: int, param: str, rule: dict, value):
        """One editor, chosen by the schema's type."""
        kind = rule.get("type")

        if kind == "choice":
            widget = QtWidgets.QComboBox()
            widget.addItems([str(v) for v in rule.get("values", [])])
            widget.setCurrentText(str(value))
            widget.currentTextChanged.connect(
                lambda text: self._set_param(row, param, text))
            return widget

        if kind == "int":
            widget = QtWidgets.QSpinBox()
            widget.setRange(int(rule.get("min", 0)), int(rule.get("max", 10**6)))
            widget.setSingleStep(int(rule.get("step", 1)))
            widget.setValue(int(value if value is not None else 0))
            widget.valueChanged.connect(
                lambda number: self._set_param(row, param, int(number)))
            return widget

        if kind == "float":
            widget = QtWidgets.QDoubleSpinBox()
            widget.setDecimals(int(rule.get("decimals", 3)))
            widget.setRange(float(rule.get("min", 0.0)),
                            float(rule.get("max", 10**6)))
            widget.setSingleStep(float(rule.get("step", 0.1)))
            widget.setValue(float(value if value is not None else 0.0))
            widget.valueChanged.connect(
                lambda number: self._set_param(row, param, float(number)))
            return widget

        if kind == "ints":
            widget = QtWidgets.QLineEdit(
                ", ".join(str(int(n)) for n in (value or [])))
            widget.setPlaceholderText("e.g. 256, 128")
            widget.setToolTip(
                f"One number per layer, comma separated. At most "
                f"{rule.get('max_items', 4)} layers, each "
                f"{rule.get('min', 1)}-{rule.get('max', 1024)} units.")
            widget.editingFinished.connect(
                lambda: self._set_param(row, param, widget.text()))
            return widget

        widget = QtWidgets.QLineEdit(str(value))             # pragma: no cover
        widget.editingFinished.connect(
            lambda: self._set_param(row, param, widget.text()))
        return widget

    # -- edits ----------------------------------------------------------------

    def _revalidate(self, refresh_form: bool = False) -> None:
        """
        Re-validate after every edit, so the dialog can never hold a spec the
        trainer would refuse. The current row is kept.
        """
        row = self.model_list.currentRow()
        self.spec, problems = validate_model_spec(self.spec)
        self._loading = True
        try:
            for index, entry in enumerate(self.spec["models"]):
                if index < self.model_list.count():
                    item = self.model_list.item(index)
                    item.setText(self._row_text(entry))
                    item.setCheckState(QtCore.Qt.Checked if entry["enabled"]
                                       else QtCore.Qt.Unchecked)
                    item.setForeground(QtGui.QColor(
                        COL_TEXT if entry["enabled"] else COL_TEXT_DIM))
        finally:
            self._loading = False
        self._refresh_summary(problems)
        if refresh_form:
            self._on_row_changed(row)

    def _set_param(self, row: int, param: str, value) -> None:
        if self._loading or not (0 <= row < len(self.spec["models"])):
            return
        self.spec["models"][row]["params"][param] = value
        self._revalidate()

    def _set_name(self, row: int, text: str) -> None:
        if self._loading or not (0 <= row < len(self.spec["models"])):
            return
        text = text.strip()
        if not text or text == self.spec["models"][row]["name"]:
            return
        self.spec["models"][row]["name"] = text
        self._revalidate(refresh_form=True)

    def _set_scaler(self, row: int, text: str) -> None:
        if self._loading or not (0 <= row < len(self.spec["models"])):
            return
        self.spec["models"][row]["scaler"] = text
        self._revalidate()

    def _on_item_changed(self, item) -> None:
        if self._loading:
            return
        row = self.model_list.row(item)
        if 0 <= row < len(self.spec["models"]):
            self.spec["models"][row]["enabled"] = \
                item.checkState() == QtCore.Qt.Checked
            self._revalidate()

    def _on_auto_changed(self, *_args) -> None:
        if self._loading:
            return
        self.spec["auto"] = {
            "enabled": self.auto_check.isChecked(),
            "max_per_model": int(self.auto_spin.value()),
        }
        self._revalidate()

    def _on_data_changed(self, *_args) -> None:
        if self._loading:
            return
        keys = [self.data_list.item(i).data(QtCore.Qt.UserRole)
                for i in range(self.data_list.count())]
        ticked = [self.data_list.item(i).data(QtCore.Qt.UserRole)
                  for i in range(self.data_list.count())
                  if self.data_list.item(i).checkState() == QtCore.Qt.Checked]
        # All or none ticked is "everything", stored as an empty list so a
        # session recorded tomorrow is included without opening this dialog.
        include = [] if len(ticked) in (0, len(keys)) else ticked
        self.spec["data"] = {
            "include": include,
            "calibration": self.cal_combo.currentData()
            or DATA_CALIBRATION_DEFAULT,
        }
        self._revalidate()

    def _set_auto_for(self, row: int, enabled: bool) -> None:
        if self._loading or not (0 <= row < len(self.spec["models"])):
            return
        self.spec["models"][row]["auto"] = bool(enabled)
        self._revalidate()

    def _on_voting_changed(self, *_args) -> None:
        if self._loading:
            return
        self.spec["voting"] = {
            "enabled": self.vote_check.isChecked(),
            "mode": self.vote_mode.currentData() or "soft",
            "always": self.vote_always.isChecked(),
        }
        self._revalidate()

    def _set_all(self, enabled: bool) -> None:
        for entry in self.spec["models"]:
            entry["enabled"] = bool(enabled)
        self._reload_all()

    def add_model(self) -> None:
        kind = self.add_combo.currentData()
        schema = cfg.MODEL_KINDS[kind]
        existing = {m["name"] for m in self.spec["models"]}
        name = kind
        suffix = 2
        while name in existing:
            name = f"{kind}-{suffix}"
            suffix += 1
        self.spec["models"].append({
            "name": name, "kind": kind, "enabled": True,
            "scaler": schema.get("scaler", "none"),
            "params": {p: model_param_default(kind, p)
                       for p in schema["params"]},
        })
        self._reload_all()
        self.model_list.setCurrentRow(self.model_list.count() - 1)

    def remove_model(self) -> None:
        row = self.model_list.currentRow()
        if not (0 <= row < len(self.spec["models"])):
            return
        if len(self.spec["models"]) == 1:
            self.problem_label.setText(
                "the last model cannot be removed -- there would be nothing "
                "to train")
            return
        del self.spec["models"][row]
        self._reload_all()

    def reset_defaults(self) -> None:
        answer = QtWidgets.QMessageBox.question(
            self, "Reset model spec",
            "Go back to the built-in roster?\n\nThis discards the models and "
            "hyperparameters you have set here.",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            return
        self.spec = default_model_spec()
        self._reload_all()

    def save_spec(self) -> None:
        try:
            save_model_spec(self.spec_path, self.spec)
        except OSError as exc:
            QtWidgets.QMessageBox.critical(
                self, "Model spec", f"Could not write the spec:\n{exc}")
            return
        self.saved = True
        self.accept()
