"""
NN STATS.
"""

from typing import TYPE_CHECKING

import math

import numpy as np

from emg_app import config as cfg
from emg_app.gui.tokens import (
    COL_ACCENT, COL_BAD, COL_GOOD, COL_PANEL, COL_TEXT_DIM, COL_WARN, RAD_SM,
    SP_SM)
from emg_app.neural import load_nn_stats, nn_stats_path
from emg_app.model_spec import estimator_size, file_size_text
from emg_app.gui.qt import QtCore, QtGui, QtWidgets
from emg_app.gui.widgets import (
    _stat_tile, fit_table_height, fit_to_screen, make_label, wrap_label)

if TYPE_CHECKING:                  # type hints only; avoids an import cycle
    from emg_app.neural import NeuralEngine
    from emg_app.prediction import LivePredictor


class NeuralStatsDialog(QtWidgets.QDialog):
    """
    The NEURAL NETWORK's own numbers -- and only its own.

    A separate dialog rather than a tab inside MODEL STATS, for the same
    reason the net is a separate file with a separate button: the two models
    are independently validated and must be read independently. Where they
    are compared, at the bottom, both protocols are named and the comparison
    says in words that neither number is an ensemble.

    It reads the plain-JSON sidecar `train_nn.py` writes beside the
    checkpoint, NOT the checkpoint itself -- the application cannot import
    torch to unpickle a `.pt`. That is also what makes this dialog work with
    the worker stopped, with no GPU, and on a machine where torch was never
    installed.
    """

    def __init__(self, engine: "NeuralEngine", parent=None,
                 predictor: "LivePredictor" = None):
        super().__init__(parent)
        self.setWindowTitle("Neural network statistics")
        fit_to_screen(self, 900, 800, min_width=460, min_height=340)
        self.engine = engine
        self.predictor = predictor

        sidecar = load_nn_stats(engine.model_path) or {}
        stats = dict(sidecar.get("stats") or {})
        self.sidecar = sidecar
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(8)

        if not sidecar:
            layout.addWidget(make_label("No neural network trained.", 14, True))
            layout.addWidget(wrap_label(
                f"Nothing at {nn_stats_path(engine.model_path)}.\n"
                "The sidecar is written by the trainer alongside the "
                "checkpoint; without it the application cannot read a "
                "checkpoint's metrics, because a .pt file is a torch pickle "
                "and the recorder does not import torch.",
                10, False, COL_TEXT_DIM))
            layout.addWidget(make_label(
                "Train one with:    python training/train_nn.py", 11, False, COL_ACCENT))
            close = QtWidgets.QPushButton("Close")
            close.clicked.connect(self.accept)
            layout.addStretch(1)
            layout.addWidget(close)
            return

        kind = sidecar.get("kind", "?")
        layout.addWidget(make_label(f"NN-{kind}", 18, True, COL_ACCENT))
        layout.addWidget(wrap_label(
            f"{sidecar.get('architecture', '')}", 10, False, COL_TEXT_DIM
        ))
        layout.addWidget(make_label(
            f"trained {sidecar.get('created_iso', '?')}   |   "
            f"{stats.get('n_parameters', '?')} parameters   |   "
            f"{stats.get('epochs', '?')} epochs", 10, False, COL_TEXT_DIM))

        #  The separation rule, at the top, where it cannot be scrolled past.
        banner = QtWidgets.QLabel(
            "This model is NOT part of the app's answer. The LIVE CLASSIFIER "
            "panel is produced entirely by the scikit-learn model; this net "
            "has its own START button and its output is never averaged into "
            "that prediction or into the DECISION TRAIL's soft vote.")
        banner.setWordWrap(True)
        banner.setStyleSheet(
            f"QLabel {{ color: {COL_WARN}; background: {COL_PANEL}; "
            f"border: 1px solid {COL_WARN}; border-radius: {RAD_SM}px; "
            f"padding: {SP_SM}px; }}")
        layout.addWidget(banner)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        inner = QtWidgets.QWidget()
        body = QtWidgets.QVBoxLayout(inner)
        body.setSpacing(10)

        grouping = stats.get("grouping", "?")
        tiles = QtWidgets.QHBoxLayout()
        for caption, value, note in (
            ("PER GESTURE", stats.get("recording_vote_accuracy"),
             f"majority vote, {stats.get('n_recordings_voted', '?')} "
             f"recordings"),
            ("PER WINDOW", stats.get("window_accuracy"),
             f"single {float(stats.get('window_s') or 0) * 1000:.0f} ms "
             f"slice, no smoothing"),
            ("MACRO F1", stats.get("window_macro_f1"),
             "per window, class-balanced"),
        ):
            tiles.addWidget(_stat_tile(caption, value, note))
        body.addLayout(tiles)

        baseline = stats.get("majority_class_baseline")
        body.addWidget(wrap_label(
            f"Scored {grouping}-grouped, against a majority-class baseline of "
            + (f"{baseline * 100:.1f} %." if baseline else "?")
            + "  Quote the baseline next to the number: on a six-class "
              "problem it is what the number has to beat, not 1/n.",
            10, False, COL_TEXT_DIM))

        # ---- what the net saw, and what it bought -----------------------
        body.addWidget(make_label("WHAT THE TEMPORAL STRUCTURE IS WORTH",
                                  11, True, COL_TEXT_DIM))
        control = stats.get("dc_only_control")
        if control:
            gain = ((stats.get("window_accuracy") or 0)
                    - control["window_accuracy"]) * 100
            row = QtWidgets.QHBoxLayout()
            row.addWidget(_stat_tile(
                "DC ONLY", control.get("window_accuracy"),
                "the same net, every window flattened to its 8 channel "
                "means -- all time removed", neutral=True))
            row.addWidget(_stat_tile(
                "FULL INPUT", stats.get("window_accuracy"),
                "the real window, time intact"))
            body.addLayout(row)
            verdict = make_label(
                f"The temporal structure is worth {gain:+.1f} points."
                + ("  That is thin: with the 8 channel means alone this net "
                   "scores nearly the same, so nothing here is evidence that "
                   "a sequence model was needed."
                   if gain < 5 else
                   "  So the net is reading how the signal changes, not only "
                   "how loud it is."),
                11, True, COL_BAD if gain < 5 else COL_GOOD)
            verdict.setWordWrap(True)
            body.addWidget(verdict)
            if (stats.get("calibration_mode") or
                    (sidecar.get("calibration") or {}).get("mode")) != "off":
                hint = make_label(
                    "Reminder: a RISING DC-only score is not good news. "
                    "Per-wearer calibration makes the eight channel means "
                    "transfer between people, which is not the same as "
                    "learning a gesture.", 9, False, COL_TEXT_DIM)
                hint.setWordWrap(True)
                body.addWidget(hint)
        else:
            body.addWidget(make_label(
                "Not measured on this run (--no-control).", 10, False,
                COL_TEXT_DIM))

        # ---- protocol ----------------------------------------------------
        body.addWidget(make_label("HOW THIS WAS MEASURED", 11, True,
                                  COL_TEXT_DIM))
        protocol = QtWidgets.QPlainTextEdit()
        protocol.setReadOnly(True)
        protocol.setMaximumHeight(190)
        folds = stats.get("fold_accuracies") or []
        contrast = stats.get("grouping_contrast")
        subject = stats.get("subject_independent")
        calibration = sidecar.get("calibration") or {}
        lines = [
            f"Protocol      : {stats.get('validation_protocol', '?')}",
            f"Folds         : {stats.get('n_splits', '?')}"
            + (f"   ({', '.join(f'{a * 100:.0f}%' for a in folds)})"
               if folds else ""),
            f"Windows       : {stats.get('n_windows', '?')} x "
            f"{stats.get('win_samples', '?')} samples x 8 channels"
            + ("   (deduped by source timeline)"
               if stats.get("deduped") else "   (NOT deduped -- leaks)"),
            f"Recordings    : {stats.get('n_recordings', '?')} used of "
            f"{stats.get('n_recordings_total', '?')} "
            f"(filter: {stats.get('excluded', '?')})",
            f"Subjects      : {', '.join(stats.get('subjects', []) or ['?'])}"
            f"   sessions: {stats.get('n_sessions', '?')}",
            f"Calibration   : {calibration.get('mode', 'off')}"
            + (f"   (requested {calibration.get('requested')})"
               if calibration.get("requested") !=
                  calibration.get("mode") else "")
            + (f"   wearers: {', '.join(calibration.get('subjects') or [])}"
               if calibration.get("subjects") else ""),
            f"Training      : {stats.get('epochs', '?')} epochs, batch "
            f"{stats.get('batch_size', '?')}, lr "
            f"{stats.get('learning_rate', '?')}, weight decay "
            f"{stats.get('weight_decay', '?')}, label smoothing "
            f"{stats.get('label_smoothing', '?')}",
            "Early stopping: NO -- stopping when the held-out fold looks "
            "best is fitting the test set.",
            "Scaling       : log1p + per-channel standardise, fitted on the "
            "TRAINING FOLD only; the fitted numbers ship inside the "
            "checkpoint.",
        ]
        if contrast:
            gap = ((stats.get("window_accuracy") or 0)
                   - contrast["window_accuracy"]) * 100
            lines.append(
                f"Other protocol: {contrast.get('grouping', '?')}-grouped "
                f"{contrast['window_accuracy'] * 100:.1f} % per window "
                f"({gap:+.1f} points) -- a contrast, not the selection key")
        if subject:
            lines.append(
                f"Per wearer    : " + ", ".join(
                    f"{p} {s * 100:.1f} %"
                    for p, s in sorted(subject["per_subject"].items()))
                + f"   mean {subject['mean'] * 100:.1f} % "
                  f"+- {subject['std'] * 100:.1f}")
        else:
            lines.append(
                "Per wearer    : NOT MEASURABLE -- one subject only. This "
                "net is specific to the person who recorded it.")
        if calibration.get("mode", "off") != "off":
            lines.append(
                "NOTE          : a calibrated subject-grouped score is NOT "
                "zero-shot. The held-out wearer's own unlabelled signal set "
                "their gain.")
        if stats.get("selection_bias_note"):
            lines.append(f"Selection     : {stats['selection_bias_note']}")
        protocol.setPlainText("\n".join(lines))
        body.addWidget(protocol)

        # ---- every candidate ---------------------------------------------
        candidates = stats.get("candidates") or []
        if candidates:
            body.addWidget(make_label(
                f"ALL {len(candidates)} CANDIDATES  (every architecture x "
                f"window, on the same folds)", 11, True, COL_TEXT_DIM))
            table = QtWidgets.QTableWidget(len(candidates), 6)
            table.setHorizontalHeaderLabels(
                ["architecture", "window", "per window", "per gesture",
                 "macro F1", "parameters"])
            table.verticalHeader().setVisible(False)
            table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
            table.horizontalHeader().setSectionResizeMode(
                0, QtWidgets.QHeaderView.Stretch)
            chosen = (stats.get("kind"), stats.get("window_s"))
            for row, entry in enumerate(candidates):
                is_best = (entry.get("kind"), entry.get("window_s")) == chosen
                cells = [
                    ("* " if is_best else "") + str(entry.get("kind", "?")),
                    f"{float(entry.get('window_s') or 0):.2f} s",
                    f"{float(entry.get('window_accuracy') or 0) * 100:.1f} %",
                    f"{float(entry.get('recording_vote_accuracy') or 0) * 100:.1f} %",
                    f"{float(entry.get('window_macro_f1') or 0) * 100:.1f} %",
                    str(entry.get("n_parameters", "?")),
                ]
                for column, text in enumerate(cells):
                    item = QtWidgets.QTableWidgetItem(text)
                    if is_best:
                        font = item.font()
                        font.setBold(True)
                        item.setFont(font)
                        item.setForeground(QtGui.QColor(COL_ACCENT))
                    table.setItem(row, column, item)
            table.setMinimumHeight(min(280, 34 + 26 * len(candidates)))
            body.addWidget(table)
            body.addWidget(wrap_label(
                "* the one that was saved. Choosing the best of these on the "
                "same folds makes its score optimistic -- the per-wearer "
                "figures above did not pick their own folds.",
                9, False, COL_TEXT_DIM))

        # ---- confusion ----------------------------------------------------
        matrix = stats.get("confusion_matrix")
        labels = stats.get("labels") or sidecar.get("labels") or []
        if matrix and labels:
            self._add_confusion(body, np.array(matrix, dtype=float), labels)

        # ---- against the deployed model ----------------------------------
        if self.predictor is not None:
            self._add_comparison(body, stats)

        # ---- the live worker ----------------------------------------------
        body.addWidget(make_label("THE INFERENCE WORKER", 11, True,
                                  COL_TEXT_DIM))
        worker = QtWidgets.QPlainTextEdit()
        worker.setReadOnly(True)
        worker.setMaximumHeight(140)
        summary = engine.stats_summary()
        device = (engine.meta.get("device") or {}) if engine.running else {}
        trained_on = sidecar.get("device_trained_on") or {}
        worker_lines = [
            f"state         : {summary['state']}  --  {summary['status']}",
            f"process       : {engine.script}  (a separate process: the "
            f"recorder never imports torch)",
            f"running on    : "
            + (f"{device.get('device', '?')} {device.get('gpu', '')}"
               if device else "not running"),
            f"trained on    : {trained_on.get('device', '?')} "
            f"{trained_on.get('gpu', '')}  torch {trained_on.get('torch', '?')}",
            f"windows       : {summary['answered']} answered, "
            f"{summary['sent']} sent, {summary['dropped']} dropped "
            f"(the previous one had not come back), {summary['errors']} "
            f"errors",
            f"latency       : "
            + (f"{summary['latency_ms']:.1f} ms round trip, "
               f"{summary['infer_ms']:.2f} ms forward pass"
               if not math.isnan(summary["latency_ms"]) else "not measured yet")
            + f"   (budget: {cfg.NN_INTERVAL_MS} ms per window)",
        ]
        noise = engine.worker_output(6)
        if noise:
            worker_lines.append("")
            worker_lines.append("the worker's own output:")
            worker_lines.extend(f"    {line}" for line in noise)
        worker.setPlainText("\n".join(worker_lines))
        body.addWidget(worker)

        scroll.setWidget(inner)
        layout.addWidget(scroll, 1)

        buttons = QtWidgets.QHBoxLayout()
        buttons.addStretch(1)
        close = QtWidgets.QPushButton("Close")
        close.clicked.connect(self.accept)
        buttons.addWidget(close)
        layout.addLayout(buttons)

    # -- pieces ---------------------------------------------------------------

    def _add_confusion(self, body, matrix, labels) -> None:
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
            column_total = matrix[:, i].sum()
            precision = matrix[i, i] / column_total if column_total else 0.0
            for column, text in enumerate(
                    [name, f"{recall * 100:.1f} %",
                     f"{precision * 100:.1f} %", f"{int(support)}"]):
                item = QtWidgets.QTableWidgetItem(text)
                if column == 1:
                    item.setForeground(QtGui.QColor(
                        COL_GOOD if recall >= 0.8 else
                        COL_WARN if recall >= 0.5 else COL_BAD))
                table.setItem(i, column, item)
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
                    item.setBackground(QtGui.QColor(
                        34, 197, 94, int(40 + 150 * strength)) if i == j
                        else QtGui.QColor(239, 68, 68,
                                          int(30 + 140 * strength)))
                grid.setItem(i, j, item)
        grid.resizeColumnsToContents()
        grid.setMinimumHeight(min(300, 40 + 30 * len(labels)))
        body.addWidget(grid)

    def _size_rows(self, stats) -> list:
        """
        How big each model is, measured rather than quoted.

        File sizes are read off disk and the ML numbers off the fitted
        estimator (estimator_size), so they follow whatever spec was trained
        last. A soft vote gets one row per member, because that is where the
        size is: the forests dwarf the SVM and the LDA.
        """
        size = estimator_size(getattr(self.predictor, "model", None))
        net_params = int(stats.get("n_parameters") or 0)
        sidecar = getattr(self, "sidecar", {}) or {}
        parts = size["parts"]
        rows = [
            ("size on disk", file_size_text(self.predictor.full_path),
             file_size_text(self.engine.full_model_path)),
            ("learned numbers",
             f"{size['total']:,}" if size["total"] else "?",
             f"{net_params:,} parameters" if net_params else "?"),
        ]
        if size["total"] and net_params:
            rows.append(("relative size",
                         f"{size['total'] / net_params:,.0f}x the net",
                         "1x"))
        rows.append((
            "structure",
            (f"soft vote of {len(parts)} models" if len(parts) > 1
             else (parts[0][3] if parts else "?")),
            f"{sidecar.get('kind', stats.get('kind', '?'))} @ "
            f"{sidecar.get('window_s', '?')} s, hidden "
            f"{sidecar.get('hidden', '?')}"))
        if len(parts) > 1:
            for name, _cls, numbers, detail in parts:
                rows.append((f"  {name}", f"{numbers:,}  --  {detail}", ""))
        rows.append(("runs on", "CPU, scikit-learn",
                     "GPU or CPU, needs torch (in a subprocess)"))
        return rows

    def _add_comparison(self, body, stats) -> None:
        """
        The two models side by side, with both protocols named.

        This is the only place in the application where the net's number and
        the deployed model's number appear together, and it is deliberately a
        TABLE of two independent results rather than a combined figure: there
        is no ensemble, so there is no third number to show. Where the two
        were scored on different protocols the table says so instead of
        subtracting them.
        """
        deployed = dict(self.predictor.stats or {})
        if not deployed:
            return
        body.addWidget(make_label("AGAINST THE DEPLOYED MODEL", 11, True,
                                  COL_TEXT_DIM))
        #  The sklearn payload calls it `selection_grouping` and the net's
        #  sidecar calls it `grouping`. Resolved with `or` rather than
        #  `.get(key, fallback)` because a key PRESENT and set to None would
        #  slip through a default and make the comparison below claim the two
        #  protocols differ when they do not -- a warning nobody could act on.
        deployed_grouping = str(deployed.get("grouping")
                                or deployed.get("selection_grouping")
                                or "?")
        net_grouping = str(stats.get("grouping") or "?")
        rows = [
            ("model", str(deployed.get("model_name", "sklearn model")),
             f"NN-{stats.get('kind', '?')}"),
            ("grouping", deployed_grouping, net_grouping),
            ("per window",
             f"{(deployed.get('window_accuracy') or 0) * 100:.1f} %",
             f"{(stats.get('window_accuracy') or 0) * 100:.1f} %"),
            ("per gesture",
             f"{(deployed.get('recording_vote_accuracy') or 0) * 100:.1f} %",
             f"{(stats.get('recording_vote_accuracy') or 0) * 100:.1f} %"),
            ("macro F1",
             f"{(deployed.get('window_macro_f1') or 0) * 100:.1f} %",
             f"{(stats.get('window_macro_f1') or 0) * 100:.1f} %"),
            ("windows", str(deployed.get("n_windows", "?")),
             str(stats.get("n_windows", "?"))),
            ("input", f"{deployed.get('n_features', '?')} features "
                      f"({self.predictor.feature_set})",
             f"{stats.get('win_samples', '?')} x 8 raw samples"),
        ] + self._size_rows(stats)
        table = QtWidgets.QTableWidget(len(rows), 3)
        self.compare_table = table
        table.setHorizontalHeaderLabels(
            ["", "LIVE CLASSIFIER (deployed)", "NEURAL NET (separate)"])
        table.verticalHeader().setVisible(False)
        table.setEditTriggers(QtWidgets.QTableWidget.NoEditTriggers)
        for column in (1, 2):
            table.horizontalHeader().setSectionResizeMode(
                column, QtWidgets.QHeaderView.Stretch)
        for row, (caption, left, right) in enumerate(rows):
            for column, text in enumerate((caption, left, right)):
                item = QtWidgets.QTableWidgetItem(text)
                if column == 0:
                    item.setForeground(QtGui.QColor(COL_TEXT_DIM))
                table.setItem(row, column, item)
        fit_table_height(table, len(rows))
        body.addWidget(table)

        same = deployed_grouping == net_grouping != "?"
        caption = make_label(
            ("Both columns were scored on the same protocol, so the two "
             "accuracies are comparable."
             if same else
             "CAREFUL: the two columns were scored on DIFFERENT protocols, "
             "so these accuracies are not comparable. Retrain both with the "
             "same dataset on disk before reading the difference.")
            + "  Either way there is no third number: the two models are "
              "never combined, and the app's prediction comes from the left "
              "column alone.",
            10, False, COL_TEXT_DIM if same else COL_WARN)
        caption.setWordWrap(True)
        body.addWidget(caption)
