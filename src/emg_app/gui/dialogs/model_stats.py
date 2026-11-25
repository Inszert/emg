"""
MODEL STATS.
"""

from typing import TYPE_CHECKING

import os

import numpy as np

from emg_app import config as cfg
from emg_app.gui.tokens import (
    COL_ACCENT, COL_BAD, COL_BG, COL_BORDER, COL_GOOD, COL_PANEL,
    COL_TEXT_DIM, COL_WARN, RAD_SM, SP_SM)
from emg_app.model_spec import model_report_html
from emg_app.dataset import dataset_fingerprint
from emg_app.decision_panel import DecisionPanel
from emg_app.gui.qt import QtCore, QtGui, QtWidgets
from emg_app.gui.widgets import fit_to_screen, make_label
from emg_app.gui.decision_trail import DecisionTrailView, FeatureSpaceView

if TYPE_CHECKING:                  # type hints only; avoids an import cycle
    from emg_app.prediction import LivePredictor


class ModelStatsDialog(QtWidgets.QDialog):
    @staticmethod
    def _members_table(members):
        table = QtWidgets.QTableWidget(len(members), 5)
        table.setHorizontalHeaderLabels(
            ["model", "per window", "macro-F1", "per gesture", "size"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        table.setSelectionMode(QtWidgets.QAbstractItemView.NoSelection)

        def pct(v):
            return "--" if v is None else f"{v * 100:.1f} %"
        from emg_app import zoo
        for r, m in enumerate(members):
            cells = [m["name"], pct(m["window_acc"]), pct(m["macro_f1"]),
                     pct(m["vote_acc"]), zoo.human_size(m["size_bytes"])]
            for c, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if c:
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                table.setItem(r, c, item)
        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(True)
        from emg_app.gui.widgets import fit_table_height
        fit_table_height(table, len(members))
        return table

    """
    What the trained model actually is and how well it really does.

    Everything here comes from the cross-validation that train_model.py ran;
    nothing is recomputed or rounded up. The validation protocol and its
    limitations are shown as prominently as the accuracy, because an accuracy
    without its protocol is not a result.
    """

    def __init__(self, predictor: "LivePredictor", parent=None,
                 panel: "DecisionPanel" = None):
        super().__init__(parent)
        self.setWindowTitle("Model statistics")
        fit_to_screen(self, 940, 820, min_width=480, min_height=360)
        self.predictor = predictor
        #  Self-sufficient when nobody hands one in: the dialog is opened
        #  from the tests and from --summary paths too, and a missing panel
        #  is a normal state with its own message, not an error.
        if panel is None:
            panel = DecisionPanel(model_path=predictor.path)
            panel.load()
        self.panel = panel
        self.trail_view = None
        self.space_view = None
        self.trail_timer = None

        stats = dict(predictor.stats or {})
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        if not predictor.ready and not stats:
            layout.addWidget(make_label("No model loaded.", 14, True))
            layout.addWidget(make_label(predictor.status, 11, False, COL_TEXT_DIM))
            layout.addWidget(make_label(
                "Train one with:    python training/train_model.py", 11, False, COL_ACCENT))
            close = QtWidgets.QPushButton("Close")
            close.clicked.connect(self.accept)
            layout.addStretch(1)
            layout.addWidget(close)
            return

        layout.addWidget(make_label(
            stats.get("model_name", "model"), 18, True, COL_ACCENT))
        layout.addWidget(make_label(
            f"trained {stats.get('created_iso', '?')}", 10, False, COL_TEXT_DIM))

        # ---- headline numbers ------------------------------------------------
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        body = QtWidgets.QVBoxLayout(inner)
        body.setSpacing(10)

        window_acc = stats.get("window_accuracy")
        vote_acc = stats.get("recording_vote_accuracy")
        tiles = QtWidgets.QHBoxLayout()
        for caption, value, note in (
            ("PER GESTURE", vote_acc,
             f"majority vote, {stats.get('n_recordings_voted', '?')} recordings"),
            ("PER WINDOW", window_acc,
             f"single {stats.get('window_s', 0) * 1000:.0f} ms slice, no smoothing"),
            ("MACRO F1", stats.get("window_macro_f1"), "per window, class-balanced"),
        ):
            box = QtWidgets.QFrame()
            box.setObjectName("card")
            box.setStyleSheet(
                f"QFrame#card {{ background: {COL_PANEL}; border: 1px solid "
                f"{COL_BORDER}; border-radius: 8px; }}")
            col = QtWidgets.QVBoxLayout(box)
            col.setSpacing(1)
            col.addWidget(make_label(caption, 9, False, COL_TEXT_DIM))
            text = "--" if value is None else f"{value * 100:.1f} %"
            colour = COL_TEXT_DIM
            if value is not None:
                colour = (COL_GOOD if value >= 0.85
                          else COL_WARN if value >= 0.6 else COL_BAD)
            col.addWidget(make_label(text, 22, True, colour))
            note_label = make_label(note, 9, False, COL_TEXT_DIM)
            note_label.setWordWrap(True)
            col.addWidget(note_label)
            tiles.addWidget(box)
        body.addLayout(tiles)

        #  A soft vote: the members' OWN numbers, so the vote's tiles above
        #  are never mistaken for a member's.
        if stats.get("members"):
            body.addWidget(make_label(
                "THE %d MODELS IN THIS VOTE" % len(stats["members"]), 11,
                True, COL_TEXT_DIM))
            body.addWidget(self._members_table(stats["members"]))

        chance = 1.0 / max(1, len(predictor.labels or stats.get("labels", []) or [1]))
        body.addWidget(make_label(
            f"Chance level for {len(predictor.labels)} gestures is "
            f"{chance * 100:.1f} %.", 11, False, COL_TEXT_DIM))

        # ---- decision trail --------------------------------------------------
        #  How every candidate judged this window and how the seven become
        #  one. Placed above the rest because it is the only part of this
        #  dialog that moves while you use the armband.
        body.addWidget(make_label("DECISION TRAIL", 11, True, COL_TEXT_DIM))
        note = make_label(
            "Every classifier in the spec, on one window, fusing into a "
            "single soft vote. The row marked * is the one that actually "
            "predicts; the vote is shown for comparison and is not what the "
            "app loads.", 10, False, COL_TEXT_DIM)
        note.setWordWrap(True)
        body.addWidget(note)

        if getattr(predictor, "using_selection", False) and self.panel.ready and                 set(self.panel.names) != set(predictor.members or []):
            warn = make_label(
                "The trail below is the one saved with the DEPLOYED model ("
                + ", ".join(self.panel.names) + "), not your Settings "
                "selection (" + stats.get("model_name", "?") + "). Its rows "
                "and the * marker do not follow the selection; how the "
                "selected models voted is shown under the probability bars "
                "of the LIVE CLASSIFIER panel.", 10, False, COL_WARN)
            warn.setWordWrap(True)
            body.addWidget(warn)

        self.trail_view = DecisionTrailView(self.panel, predictor, self)
        trail_frame = QtWidgets.QFrame()
        trail_frame.setObjectName("card")
        trail_frame.setStyleSheet(
            f"QFrame#card {{ background: {COL_PANEL}; border: 1px solid "
            f"{COL_BORDER}; border-radius: {RAD_SM}px; }}")
        trail_box = QtWidgets.QVBoxLayout(trail_frame)
        trail_box.setContentsMargins(SP_SM, SP_SM, SP_SM, SP_SM)
        trail_box.addWidget(self.trail_view)
        body.addWidget(trail_frame)

        # ---- feature space ---------------------------------------------------
        #  The same windows the numbers above come from, as a picture. Placed
        #  under the trail because it answers a different question: not "what
        #  did the models say about this window" but "where does this window
        #  sit among everything they were trained on".
        body.addWidget(make_label("FEATURE SPACE", 11, True, COL_TEXT_DIM))
        self.space_view = FeatureSpaceView(self.panel, predictor, self)
        space_frame = QtWidgets.QFrame()
        space_frame.setObjectName("card")
        space_frame.setStyleSheet(
            f"QFrame#card {{ background: {COL_PANEL}; border: 1px solid "
            f"{COL_BORDER}; border-radius: {RAD_SM}px; }}")
        space_box = QtWidgets.QVBoxLayout(space_frame)
        space_box.setContentsMargins(SP_SM, SP_SM, SP_SM, SP_SM)
        space_box.addWidget(self.space_view)
        body.addWidget(space_frame)

        if self.panel is not None and self.panel.ready \
                and not self.panel.stale_reason(predictor):
            self._add_candidate_table(body, predictor)

        #  Repaint on the classifier's own cadence rather than faster: a
        #  prediction arrives a few times a second and there is nothing new
        #  to draw in between.
        self.trail_timer = QtCore.QTimer(self)
        self.trail_timer.setInterval(120)
        self.trail_timer.timeout.connect(self._tick_trail)
        self.trail_timer.start()

        # ---- what the model actually is --------------------------------------
        # Read off the loaded estimator rather than off the spec: this is the
        # object that will run on the next prediction.
        report = QtWidgets.QTextBrowser()
        report.setOpenExternalLinks(False)
        report.setMinimumHeight(360)
        report.setStyleSheet(
            f"QTextBrowser {{ background: {COL_BG}; border: 1px solid "
            f"{COL_BORDER}; border-radius: {RAD_SM}px; padding: {SP_SM}px; }}")
        report.setHtml(model_report_html(predictor, stats))
        report.setToolTip(
            "Read off the fitted estimator on disk, not off the spec that "
            "was meant to produce it. If the two ever disagreed, this is the "
            "one that predicts.")
        body.addWidget(report, 1)

        # ---- how it was validated -------------------------------------------
        body.addWidget(make_label("HOW THIS WAS MEASURED", 11, True, COL_TEXT_DIM))
        protocol = QtWidgets.QPlainTextEdit()
        protocol.setReadOnly(True)
        protocol.setMaximumHeight(130)
        folds = stats.get("fold_accuracies") or []
        subject_independent = stats.get("subject_independent")
        lines = [
            f"Protocol      : {stats.get('validation_protocol', '?')}",
            f"Folds         : {stats.get('n_splits', '?')}"
            + (f"   ({', '.join(f'{a * 100:.0f}%' for a in folds)})" if folds else ""),
            f"Windows       : {stats.get('n_windows', '?')} "
            f"x {stats.get('n_features', '?')} features",
            f"Recordings    : {stats.get('n_recordings', '?')} used "
            f"of {stats.get('n_recordings_total', '?')} "
            f"(filter: {stats.get('excluded', '?')})",
            f"Subjects      : {', '.join(stats.get('subjects', []) or ['?'])}"
            f"   sessions: {stats.get('n_sessions', '?')}",
            "",
            "A whole recording is either in train or in test, never both --"
            " windows overlap, so a random split would leak and inflate this.",
        ]
        if subject_independent:
            lines.append(
                f"Subject-independent: {subject_independent['mean'] * 100:.1f} % "
                f"+- {subject_independent['std'] * 100:.1f}")
        else:
            lines.append(
                "Subject-independent: NOT MEASURABLE -- one subject only. This "
                "model is specific to the person who recorded it.")
        protocol.setPlainText("\n".join(lines))
        body.addWidget(protocol)

        # ---- per class -------------------------------------------------------
        cm = stats.get("confusion_matrix")
        labels = stats.get("labels") or predictor.labels
        if cm and labels:
            matrix = np.array(cm, dtype=float)
            body.addWidget(make_label(
                "PER GESTURE (per window, summed over folds)", 11, True,
                COL_TEXT_DIM))
            table = QtWidgets.QTableWidget(len(labels), 4)
            table.setHorizontalHeaderLabels(
                ["gesture", "recall", "precision", "windows"])
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
            table.horizontalHeader().setSectionResizeMode(
                0, QtWidgets.QHeaderView.Stretch)
            for i, name in enumerate(labels):
                support = matrix[i].sum()
                recall = matrix[i, i] / support if support else 0.0
                column = matrix[:, i].sum()
                precision = matrix[i, i] / column if column else 0.0
                for col, text in enumerate(
                        [name, f"{recall * 100:.1f} %",
                         f"{precision * 100:.1f} %", f"{int(support)}"]):
                    item = QtWidgets.QTableWidgetItem(text)
                    if col == 1:
                        item.setForeground(QtGui.QColor(
                            COL_GOOD if recall >= 0.8 else
                            COL_WARN if recall >= 0.5 else COL_BAD))
                    table.setItem(i, col, item)
            table.setMinimumHeight(min(260, 30 + 28 * len(labels)))
            body.addWidget(table)

            body.addWidget(make_label(
                "CONFUSION MATRIX  (rows = truth, columns = predicted)",
                11, True, COL_TEXT_DIM))
            grid = QtWidgets.QTableWidget(len(labels), len(labels))
            grid.setHorizontalHeaderLabels([l[:7] for l in labels])
            grid.setVerticalHeaderLabels([l[:9] for l in labels])
            grid.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
            peak = max(1.0, matrix.max())
            for i in range(len(labels)):
                row_total = matrix[i].sum()
                for j in range(len(labels)):
                    share = matrix[i, j] / row_total if row_total else 0.0
                    item = QtWidgets.QTableWidgetItem(f"{int(matrix[i, j])}")
                    item.setTextAlignment(QtCore.Qt.AlignCenter)
                    item.setToolTip(f"{labels[i]} predicted as {labels[j]}: "
                                    f"{share * 100:.1f} %")
                    if matrix[i, j]:
                        strength = matrix[i, j] / peak
                        if i == j:
                            item.setBackground(QtGui.QColor(
                                34, 197, 94, int(40 + 150 * strength)))
                        else:
                            item.setBackground(QtGui.QColor(
                                239, 68, 68, int(30 + 140 * strength)))
                    grid.setItem(i, j, item)
            grid.resizeColumnsToContents()
            grid.setMinimumHeight(min(300, 40 + 30 * len(labels)))
            body.addWidget(grid)

        # ---- setup -----------------------------------------------------------
        body.addWidget(make_label("PREPROCESSING", 11, True, COL_TEXT_DIM))
        setup = QtWidgets.QPlainTextEdit()
        setup.setReadOnly(True)
        setup.setMaximumHeight(120)
        setup.setPlainText("\n".join([
            f"window         : {stats.get('window_s', predictor.window_s)} s "
            f"({predictor.window_samples} samples at {predictor.fs:.0f} Hz)",
            f"overlap        : {stats.get('overlap', '?')}",
            f"representation : {predictor.representation}",
            f"feature set    : {predictor.feature_set} "
            f"({stats.get('n_features', '?')} features)",
            f"EMG mode       : {predictor.payload_meta.get('emg_mode', '?')}",
            f"report         : {stats.get('report_file', '-')}",
        ]))
        body.addWidget(setup)

        fit = stats.get("train_fit_accuracy")
        if fit is not None:
            body.addWidget(make_label(
                f"Training-set fit: {fit * 100:.1f} %. This is NOT a "
                "performance estimate -- it only shows the model can represent "
                "the data. Judge it by the cross-validated numbers above.",
                10, False, COL_WARN))

        # ---- provenance ------------------------------------------------------
        #  Which recordings this model saw, and whether that is still the
        #  dataset. Asked constantly, and unanswerable from the scores alone.
        body.addWidget(make_label("TRAINING DATA", 11, True, COL_TEXT_DIM))
        data = predictor.payload_meta.get("data") or {}
        status = predictor.data_status(dataset_fingerprint(
            os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER)))
        provenance = QtWidgets.QPlainTextEdit()
        provenance.setReadOnly(True)
        provenance.setMaximumHeight(170)
        lines = [
            f"status         : {status['headline']}",
            f"trained        : {predictor.payload_meta.get('created_iso', '?')}",
            f"recordings     : {data.get('n_recordings', stats.get('n_recordings', '?'))}"
            + (f" of {stats['n_recordings_total']} on disk at the time"
               if stats.get("n_recordings_total") else ""),
            f"per gesture    : " + (", ".join(
                f"{name} {count}"
                for name, count in sorted((data.get('labels') or {}).items()))
                or ", ".join(stats.get("labels") or predictor.labels)),
            f"subjects       : " + (", ".join(
                data.get("subjects") or stats.get("subjects") or []) or "?"),
            f"sessions       : " + (", ".join(data.get("sessions") or [])
                                    or str(stats.get("n_sessions", "?"))),
            f"newest in set  : " + (data.get("newest_stamp") or "not recorded"),
            f"dataset id     : " + (data.get("uid_digest") or "not recorded"),
            f"excluded       : {stats.get('excluded', '?')}",
        ]
        provenance.setPlainText("\n".join(lines))
        body.addWidget(provenance)
        body.addWidget(make_label(
            status["detail"], 10, False,
            {"good": COL_GOOD, "warn": COL_WARN}.get(status["severity"],
                                                     COL_TEXT_DIM)))

        body.addStretch(1)
        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        reload_button = QtWidgets.QPushButton("Reload model")
        reload_button.clicked.connect(lambda: self._reload(predictor))
        close = QtWidgets.QPushButton("Close")
        close.setObjectName("primary")
        close.setDefault(True)
        close.clicked.connect(self.accept)
        buttons.addWidget(reload_button)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    def _tick_trail(self) -> None:
        if not self.isVisible():
            return
        if self.trail_view is not None:
            self.trail_view.refresh_live()
        #  Shares the trail's timer rather than adding a second one: both
        #  draw the same live window, and it only changes when a prediction
        #  arrives.
        if self.space_view is not None:
            self.space_view.refresh_live()

    def _add_candidate_table(self, body, predictor) -> None:
        """
        Every candidate's out-of-fold score on the winner's setup.

        Not the same thing as `best_per_model` in the sweep report: there,
        each model got to pick its own preprocessing. Here they all run on
        one setup, which is what makes the trail above a fair comparison and
        usually makes these numbers lower. Said in the dialog rather than
        left for someone to discover.
        """
        panel = self.panel
        body.addWidget(make_label(
            "EVERY CANDIDATE, SAME SETUP (out-of-fold)", 11, True,
            COL_TEXT_DIM))
        explain = make_label(
            f"All of them on the winning setup -- "
            f"{panel.setup.get('window_s', '?')} s windows, overlap "
            f"{panel.setup.get('overlap', '?')}, "
            f"{panel.setup.get('representation', '?')}/"
            f"{panel.setup.get('feature_set', '?')}, "
            f"{panel.setup.get('n_features', '?')} features. The sweep report "
            f"lets each model choose its own preprocessing instead, so those "
            f"numbers are higher and are not comparable with these.",
            10, False, COL_TEXT_DIM)
        explain.setWordWrap(True)
        body.addWidget(explain)

        order = list(panel.names) + ([panel.vote_name] if panel.vote_name
                                     and panel.vote_name in panel.cv else [])
        table = QtWidgets.QTableWidget(len(order), 5)
        table.setHorizontalHeaderLabels(
            ["candidate", "per window", "per recording", "mean conf.",
             "windows wrong"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        table.horizontalHeader().setSectionResizeMode(
            0, QtWidgets.QHeaderView.Stretch)
        for row, name in enumerate(order):
            entry = panel.cv.get(name, {})
            window = entry.get("window_accuracy")
            recording = entry.get("recording_accuracy")
            shown = name
            if name == panel.winner:
                shown += "   * deployed"
            elif name == panel.vote_name:
                shown += "   (ensemble)"
            cells = [
                shown,
                "--" if window is None else f"{window * 100:.2f} %",
                "--" if recording is None else f"{recording * 100:.2f} %",
                "--" if entry.get("mean_confidence") is None
                else f"{entry['mean_confidence']:.3f}",
                str(entry.get("n_windows_wrong", "--")),
            ]
            for column, text in enumerate(cells):
                item = QtWidgets.QTableWidgetItem(text)
                if column in (1, 2):
                    item.setTextAlignment(QtCore.Qt.AlignRight
                                          | QtCore.Qt.AlignVCenter)
                    value = window if column == 1 else recording
                    if value is not None:
                        item.setForeground(QtGui.QColor(
                            COL_GOOD if value >= 0.85 else
                            COL_WARN if value >= 0.6 else COL_BAD))
                if name == panel.winner:
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                table.setItem(row, column, item)
        table.setMinimumHeight(min(300, 34 + 28 * len(order)))
        body.addWidget(table)

    def closeEvent(self, event):
        if self.trail_timer is not None:
            self.trail_timer.stop()
        super().closeEvent(event)

    def _reload(self, predictor) -> None:
        ok = predictor.load()
        QtWidgets.QMessageBox.information(
            self, "Reload model",
            "Model reloaded." if ok else f"Could not load: {predictor.status}")
        self.accept()
