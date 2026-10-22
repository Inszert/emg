"""
MainWindow: the window itself -- layout, timers, the control state machine,
connection status and closing. Each feature lives in its own mixin in this
package; MainWindow combines them.
"""

import os
import time
from datetime import datetime

import numpy as np

from emg_app import config as cfg
from emg_app.acquisition import (
    CONN_CONNECTED, CONN_CONNECTING, CONN_DISCONNECTED, CONN_ERROR,
    CONN_RECONNECTING, CONN_STOPPED, MyoAcquisition)
from emg_app.compute import blank_compute_info
from emg_app.decision_panel import DecisionPanel
from emg_app.gui.plot import EmgPlotWidget
from emg_app.gui.qt import QtCore, QtGui, QtWidgets
from emg_app.gui.tokens import (
    COL_ACCENT, COL_BAD, COL_BORDER, COL_GOOD, COL_PANEL, COL_TEXT,
    COL_TEXT_DIM, COL_WARN, FS_SM, SP_SM)
from emg_app.gui.widgets import (
    AcquisitionBridge, StatusBanner, available_screen_size, collapsible,
    fit_to_screen, make_label)
from emg_app.model_spec import load_model_spec, model_spec_path
from emg_app.neural import NeuralEngine
from emg_app.prediction import LivePredictor
from emg_app.recording import DatasetWriter
from emg_app.sequence import (
    STATE_COUNTDOWN, STATE_PREPARE, STATE_RECONNECTING, STATE_RECORDING,
    SequenceController)
from emg_app.utils import emg_range, fmt_duration, nominal_rate

from emg_app.gui.main_window.session import SessionMixin
from emg_app.gui.main_window.recordings import RecordingsMixin
from emg_app.gui.main_window.labeling import LabelingMixin
from emg_app.gui.main_window.takes import TakesMixin
from emg_app.gui.main_window.markers import MarkersMixin
from emg_app.gui.main_window.prediction import PredictionMixin
from emg_app.gui.main_window.neural import NeuralMixin
from emg_app.gui.main_window.training import TrainingMixin

# ------------------------------------------------------------------------------
# Main window
# ------------------------------------------------------------------------------


class MainWindow(SessionMixin, RecordingsMixin, LabelingMixin, TakesMixin,
                 MarkersMixin, PredictionMixin, NeuralMixin, TrainingMixin,
                 QtWidgets.QMainWindow):
    """Everything wired together."""


    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{cfg.APP_NAME}  v{cfg.APP_VERSION}")
        # Fit the display rather than assuming a big monitor. The minimum is
        # deliberately small: a minimum bigger than the screen is a window the
        # operator cannot resize into view, which is worse than a cramped one.
        self._fitted_size = fit_to_screen(
            self, cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT, min_width=900, min_height=600)

        self.person_id = cfg.PERSON_ID
        self.session_id = cfg.SESSION_ID
        self.recordings = []              # List[Recording] collected this run
        self.session_markers = []         # List[Marker] logged this session

        # FREE TAKE state. Deliberately NOT routed through SequenceController:
        # a take has no plan, no countdown and no review step, and the
        # acquisition thread has only one capture slot, so the two must never
        # run at the same time (enforced in _refresh_controls).
        self.free_capture = None          # the Capture while a take runs
        self.free_take_label = ""         # intended gesture for that take
        self.free_take_number = 0         # position in the session
        self.export_worker = None
        self.review_recording = None      # the take awaiting a verdict
        self.prediction_active = False    # continuous mode running
        self.one_shot_capture_at = 0.0    # when the classified window starts
        self.one_shot_deadline = 0.0      # when it ends
        self.one_shot_clear_at = 0.0      # when to clear a held one-shot result
        #  The background job running train_model.py (a TrainingJobWatcher,
        #  not a QProcess: the job outlives the window -- see SECTION 5b).
        self.training_process = None
        self._job = None                  # the watcher of the job running
        self._chain_job = None            # set while an ML + NN chain runs
        self._job_step = ""               # "ml" / "nn": which step logs now
        self._job_steps_ended = set()
        # Which classifiers the next retrain sweeps. Loaded from disk so an
        # edit survives a restart; the defaults reproduce the roster
        # train_model.py has always used.
        self.model_spec, self._spec_problems = load_model_spec(
            model_spec_path(cfg.MODEL_PATH))
        self.training_output = []         # its captured stdout
        self.training_started_at = 0.0    # for the elapsed/ETA readout
        self.training_phase_text = ""
        # ETA state, see _estimate_training_remaining(). The 1 s clock keeps
        # elapsed and the countdown moving between trainer events, which can
        # be minutes apart while one expensive model is being fitted.
        self._train_stages = []           # [(name, units)] from the trainer
        self._train_stage_clock = {}      # name -> [t_first, done_first, t_last, done_last]
        self._train_eta_snapshot = None   # (monotonic time, seconds left, lower bound)
        self._train_live = False
        self._train_clock = QtCore.QTimer(self)
        self._train_clock.setInterval(1000)
        self._train_clock.timeout.connect(self._tick_training_clock)
        self._train_clock.start()
        # GPU or CPU. Unknown until the subprocess probe answers, and shown
        # as unknown until then rather than assumed either way.
        self.compute_info = blank_compute_info()
        self.compute_process = None       # QProcess running the probe
        self._link_paused = False         # session paused by a dropped link
        self._last_banner = (None, None, None)

        self.writer = DatasetWriter(
            os.path.join(cfg.PROJECT_DIR, cfg.DATA_FOLDER), self.person_id, self.session_id
        )

        self.bridge = AcquisitionBridge()
        self.bridge.state_changed.connect(self._on_connection_state)
        self.bridge.log_line.connect(self._append_log)

        self.acq = MyoAcquisition(
            port=cfg.MYO_PORT,
            mode=cfg.EMG_MODE,
            on_state=lambda state, msg: self.bridge.state_changed.emit(state, msg),
            on_log=lambda text: self.bridge.log_line.emit(text),
        )

        self.sequence = SequenceController(
            on_capture_start=self._on_capture_start,
            on_capture_stop=self._on_capture_stop,
            on_state=self._on_sequence_state,
            on_finished=self._on_sequence_finished,
            on_capture_discard=self._on_capture_discard,
        )

        self.predictor = LivePredictor()
        self.predictor.load()
        # The side file behind MODEL STATS -> DECISION TRAIL. Optional by
        # design: a failure to load it must not touch the classifier, so it
        # is a separate object with its own status string.
        #  Loaded lazily, when MODEL STATS is first opened: it unpickles every
        #  candidate (two 400-tree forests among them) and nothing outside
        #  that dialog reads it, so there is no reason to spend the time at
        #  startup.
        self.decision_panel = DecisionPanel(model_path=cfg.MODEL_PATH)
        self.stats_dialog = None

        #  The neural net: a SEPARATE engine, a separate process, a separate
        #  button. Constructed but not started -- nothing happens until START
        #  NN is pressed, so a machine with no torch and no checkpoint behaves
        #  exactly as it did before this existed. It is never consulted by
        #  `self.predictor` and never feeds `self.decision_panel`; see
        #  section 8L.
        self.nn_engine = NeuralEngine() if cfg.ENABLE_NEURAL_NET else None
        self.nn_training_process = None    # the job's watcher, during train_nn
        self.nn_stats_dialog = None
        self.nn_prob_bars = {}

        self._build_ui()
        self._load_session_markers()
        self._refresh_take_history()
        self._refresh_spec_summary()
        self._refresh_gesture_counts()
        self._refresh_label_counts()
        self._start_timers()

        self.acq.start()
        self._append_log(f"{cfg.APP_NAME} {cfg.APP_VERSION} started. Mode: {cfg.EMG_MODE}.")
        self._refresh_plan_info()

    # ======================================================================
    # UI construction
    # ======================================================================

    def _build_ui(self) -> None:
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(10, 8, 10, 8)
        outer.setSpacing(8)

        outer.addWidget(self._build_header())

        self.banner = StatusBanner()
        outer.addWidget(self.banner)

        splitter = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        splitter.addWidget(self._build_plot_area())
        splitter.addWidget(self._build_side_panel())
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)
        splitter.setSizes([1180, 440])
        outer.addWidget(splitter, 1)

        outer.addWidget(self._build_button_bar())

        self.status = self.statusBar()
        self.status_conn = QtWidgets.QLabel("--")
        self.status_rate = QtWidgets.QLabel("--")
        self.status_buffer = QtWidgets.QLabel("--")
        self.status_counts = QtWidgets.QLabel("--")
        # GPU or CPU, always on screen. The full badge lives in the LIVE
        # CLASSIFIER panel, which folds away; this one cannot be folded, so
        # the answer is never more than a glance away.
        self.status_compute = QtWidgets.QLabel("GPU/CPU ?")
        for widget in (self.status_conn, self.status_rate,
                       self.status_buffer, self.status_counts,
                       self.status_compute):
            widget.setStyleSheet(f"color: {COL_TEXT_DIM}; padding: 0 10px;")
            self.status.addPermanentWidget(widget)
        self.status.showMessage("Ready.")

    def _build_header(self):
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {COL_PANEL}; border: 1px solid {COL_BORDER};"
            f" border-radius: 8px; }}"
        )
        row = QtWidgets.QHBoxLayout(frame)
        row.setContentsMargins(14, 8, 14, 8)

        row.addWidget(make_label("MYO EMG DATA COLLECTION", 15, True, COL_TEXT))
        row.addSpacing(18)

        self.conn_pill = QtWidgets.QLabel(CONN_DISCONNECTED)
        self.conn_pill.setAlignment(QtCore.Qt.AlignCenter)
        self.conn_pill.setMinimumWidth(150)
        self._set_pill(CONN_DISCONNECTED, COL_TEXT_DIM)
        row.addWidget(self.conn_pill)

        row.addStretch(1)

        self.header_person = make_label("--", 12, True, COL_ACCENT)
        self.header_gesture = make_label("--", 12, True, "#ffffff")
        self.header_count = make_label("0", 12, True, COL_GOOD)
        for caption, widget in (("PERSON", self.header_person),
                                ("GESTURE", self.header_gesture),
                                ("RECORDED", self.header_count)):
            box = QtWidgets.QVBoxLayout()
            box.setSpacing(0)
            box.addWidget(make_label(caption, 8, False, COL_TEXT_DIM))
            box.addWidget(widget)
            container = QtWidgets.QWidget()
            container.setLayout(box)
            container.setStyleSheet("background: transparent;")
            row.addWidget(container)
            row.addSpacing(16)

        return frame

    def _build_plot_area(self):
        panel = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # ---- graph toolbar ---------------------------------------------------
        bar = QtWidgets.QHBoxLayout()
        bar.setSpacing(8)

        bar.addWidget(make_label("Window", 10, False, COL_TEXT_DIM))
        self.window_spin = QtWidgets.QDoubleSpinBox()
        # Up to the whole ring buffer: the point of a long buffer is being
        # able to actually look at it.
        self.window_spin.setRange(0.25, float(cfg.RING_BUFFER_SECONDS))
        self.window_spin.setSingleStep(0.5)
        self.window_spin.setDecimals(2)
        self.window_spin.setSuffix(" s")
        self.window_spin.setValue(cfg.PLOT_WINDOW_S)
        self.window_spin.setToolTip("Visible time window")
        bar.addWidget(self.window_spin)

        self.autoscale_check = QtWidgets.QCheckBox("Autoscale")
        self.autoscale_check.setChecked(cfg.PLOT_AUTOSCALE)
        self.autoscale_check.setToolTip(
            "Grows instantly on a peak and shrinks slowly, so a spike is never "
            "clipped out of view"
        )
        bar.addWidget(self.autoscale_check)

        # The sensitivity slider is only meaningful with Autoscale OFF -- its
        # own tooltip said so -- yet it permanently occupied ~190 px of the
        # toolbar, which is what squeezed everything else into "Win" / "Sen:"
        # / "Cha" on a narrow window. It now appears only when it does
        # something.
        self.sensitivity_label = make_label("Sensitivity", FS_SM, False,
                                            COL_TEXT_DIM)
        bar.addWidget(self.sensitivity_label)
        self.scale_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.scale_slider.setRange(10, 1000)
        lo, hi = emg_range()
        self.scale_slider.setValue(int(np.clip((hi - lo) * 0.5, 10, 1000)))
        self.scale_slider.setFixedWidth(130)
        self.scale_slider.setToolTip(
            "Manual vertical scale. Only used when Autoscale is off.")
        bar.addWidget(self.scale_slider)
        self.sensitivity_label.setVisible(not cfg.PLOT_AUTOSCALE)
        self.scale_slider.setVisible(not cfg.PLOT_AUTOSCALE)

        self.envelope_check = QtWidgets.QCheckBox("RMS envelope")
        self.envelope_check.setChecked(cfg.SHOW_ENVELOPE)
        bar.addWidget(self.envelope_check)

        self.freeze_button = QtWidgets.QPushButton("Freeze graph")
        self.freeze_button.setCheckable(True)
        self.freeze_button.setToolTip(
            "Stop updating the plot. Recording and saving are unaffected."
        )
        bar.addWidget(self.freeze_button)

        reset = QtWidgets.QPushButton("Reset view")
        reset.setToolTip("Undo mouse zoom/pan")
        bar.addWidget(reset)

        bar.addStretch(1)
        layout.addLayout(bar)

        # Channels on their own row. Eight checkboxes plus a label is 220 px
        # that the first row cannot spare on a laptop screen.
        chan_bar = QtWidgets.QHBoxLayout()
        chan_bar.setSpacing(SP_SM)
        chan_bar.addWidget(make_label("Channels", FS_SM, False, COL_TEXT_DIM))
        self.channel_checks = []
        for i in range(cfg.N_CHANNELS):
            check = QtWidgets.QCheckBox(str(i + 1))
            check.setChecked(True)
            check.setStyleSheet(
                f"QCheckBox {{ color: {cfg.CHANNEL_COLORS[i]}; font-weight: 700; }}"
                f"QCheckBox::indicator:checked {{ background: {cfg.CHANNEL_COLORS[i]};"
                f" border-color: {cfg.CHANNEL_COLORS[i]}; }}"
            )
            check.toggled.connect(
                lambda checked, index=i: self.plot.set_channel_visible(index, checked)
            )
            self.channel_checks.append(check)
            chan_bar.addWidget(check)

        chan_bar.addStretch(1)
        layout.addLayout(chan_bar)

        self.plot = EmgPlotWidget()
        layout.addWidget(self.plot, 1)

        # Review read-out: what the green band currently keeps.
        self.review_info = make_label("", 11, True, "#22c55e")
        self.review_info.setWordWrap(True)
        layout.addWidget(self.review_info)
        self.plot.trim_changed.connect(self._update_review_info)
        self.plot.selection_changed.connect(self._on_selection_changed)

        self.window_spin.valueChanged.connect(self.plot.set_window_seconds)
        self.autoscale_check.toggled.connect(self.plot.set_autoscale)
        self.autoscale_check.toggled.connect(
            lambda on: (self.sensitivity_label.setVisible(not on),
                        self.scale_slider.setVisible(not on)))
        self.scale_slider.valueChanged.connect(
            lambda value: self.plot.set_manual_scale(float(value))
        )
        self.envelope_check.toggled.connect(self.plot.set_envelope_visible)
        self.freeze_button.toggled.connect(self._on_freeze_toggled)
        reset.clicked.connect(self.plot.reset_view)
        self.plot.set_manual_scale(float(self.scale_slider.value()))

        return panel

    def _build_side_panel(self):
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOff)

        inner = QtWidgets.QWidget()
        column = QtWidgets.QVBoxLayout(inner)
        column.setContentsMargins(4, 0, 4, 0)
        column.setSpacing(8)

        # Every group folds. The ones open by default are the ones you need
        # to start recording; the rest are one click away and out of the way
        # until then.
        self.sections = {}
        for key, builder, start_open in (
            ("subject", self._group_subject, True),
            ("free_take", self._group_free_take if cfg.FREE_TAKE_ENABLED else None,
             True),
            ("label", self._group_label_on_graph, True),
            ("markers", self._group_markers if cfg.MARKERS_ENABLED else None, False),
            ("plan", self._group_plan, False),
            ("channels", self._group_channels, False),
            ("recordings", self._group_recordings, True),
            ("prediction", self._group_prediction, False),
            #  Below LIVE CLASSIFIER, not inside it. The reading order is the
            #  separation: the deployed answer first, then a clearly
            #  separate second opinion that does not contribute to it.
            ("neural", self._group_neural if cfg.ENABLE_NEURAL_NET else None,
             False),
            ("training", self._group_training, False),
            ("log", self._group_log, False),
        ):
            if builder is None:
                continue
            section = collapsible(builder(), start_open and cfg.SIDE_PANELS_START_OPEN)
            section.toggled.connect(
                lambda opened, sec=section: self._on_section_toggled(sec, opened))
            self.sections[key] = section
            column.addWidget(section)
            if key in cfg.SIDE_PANELS_HIDDEN:
                section.setVisible(False)
        column.addStretch(1)

        scroll.setWidget(inner)
        self._scroll_area = scroll
        # Wide enough for the controls to read properly, but clamped on a
        # small screen so the panel cannot push the graph off the window.
        avail_w, _ = available_screen_size(self)
        scroll.setMinimumWidth(min(410, max(280, int(avail_w * 0.32))))
        scroll.setMaximumWidth(max(410, int(avail_w * 0.45)))
        return scroll

    def _group_channels(self):
        group = QtWidgets.QGroupBox("CHANNELS")
        grid = QtWidgets.QGridLayout(group)
        grid.setSpacing(3)
        grid.setColumnStretch(4, 1)

        headers = ["", "value", "min", "max", "rms"]
        for col, text in enumerate(headers):
            grid.addWidget(make_label(text, 9, False, COL_TEXT_DIM), 0, col)

        self.channel_labels = []
        for i in range(cfg.N_CHANNELS):
            name = make_label(cfg.CHANNEL_NAMES[i], 10, True, cfg.CHANNEL_COLORS[i])
            grid.addWidget(name, i + 1, 0)
            row = []
            for col in range(1, 5):
                value = make_label("0", 10, False, COL_TEXT)
                value.setFont(QtGui.QFont("Consolas", 9))
                grid.addWidget(value, i + 1, col)
                row.append(value)
            self.channel_labels.append(row)

        return group

    def _group_log(self):
        group = QtWidgets.QGroupBox("LOG")
        layout = QtWidgets.QVBoxLayout(group)
        self.log_box = QtWidgets.QPlainTextEdit()
        self.log_box.setReadOnly(True)
        self.log_box.setMaximumBlockCount(500)
        self.log_box.setMinimumHeight(110)
        self.log_box.setFont(QtGui.QFont("Consolas", 8))
        layout.addWidget(self.log_box)
        return group

    def _build_button_bar(self):
        frame = QtWidgets.QFrame()
        frame.setStyleSheet(
            f"QFrame {{ background: {COL_PANEL}; border: 1px solid {COL_BORDER};"
            f" border-radius: 8px; }}"
        )
        row = QtWidgets.QHBoxLayout(frame)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(8)

        self.start_button = QtWidgets.QPushButton("START SESSION")
        self.start_button.setObjectName("primary")
        self.start_button.setMinimumHeight(38)
        self.start_button.clicked.connect(self.start_session)

        self.pause_button = QtWidgets.QPushButton("PAUSE")
        self.pause_button.setMinimumHeight(38)
        self.pause_button.clicked.connect(self.toggle_pause)

        self.stop_button = QtWidgets.QPushButton("STOP")
        self.stop_button.setObjectName("danger")
        self.stop_button.setMinimumHeight(38)
        self.stop_button.clicked.connect(self.stop_session)

        self.skip_button = QtWidgets.QPushButton("SKIP")
        self.skip_button.setMinimumHeight(38)
        self.skip_button.clicked.connect(self.skip_current)

        self.accept_button = QtWidgets.QPushButton("ACCEPT && NEXT")
        self.accept_button.setObjectName("primary")
        self.accept_button.setMinimumHeight(38)
        self.accept_button.setToolTip(
            "Keep this take (writing the trimmed copy for the green band) and "
            "start the next repetition.  Space, Enter or Ctrl+Enter"
        )
        self.accept_button.clicked.connect(self.accept_review)

        self.redo_button = QtWidgets.QPushButton("REDO")
        self.redo_button.setObjectName("warn")
        self.redo_button.setMinimumHeight(38)
        self.redo_button.setToolTip(
            "Reject this take and record the same gesture again. The rejected "
            "file is moved to _deleted, never erased.  Ctrl+R"
        )
        self.redo_button.clicked.connect(self.redo_review)

        self.record_button = QtWidgets.QPushButton("RECORD (manual)")
        self.record_button.setMinimumHeight(38)
        self.record_button.setCheckable(True)
        self.record_button.setToolTip(
            "Record the gesture selected above until pressed again"
        )
        self.record_button.clicked.connect(self.toggle_manual_record)

        self.graph_button = QtWidgets.QPushButton("SAVE GRAPH")
        self.graph_button.setMinimumHeight(38)
        self.graph_button.clicked.connect(self.save_graph)

        self.export_button = QtWidgets.QPushButton("EXPORT DATASET")
        self.export_button.setMinimumHeight(38)
        self.export_button.setToolTip(
            "Build a combined windows + features bundle from everything on disk"
        )
        self.export_button.clicked.connect(self.export_dataset)

        for button in (self.start_button, self.accept_button, self.redo_button,
                       self.pause_button, self.stop_button, self.skip_button,
                       self.record_button):
            row.addWidget(button)
        row.addStretch(1)
        for button in (self.graph_button, self.export_button):
            row.addWidget(button)

        # Accept is driven by giving the button focus when the review starts,
        # so Space/Enter activate it the normal Qt way. The explicit shortcuts
        # use a modifier on purpose: a bare "Space" or "R" window shortcut
        # would swallow those characters when you are typing in Notes.
        self._accept_shortcut = QtGui.QShortcut(
            QtGui.QKeySequence("Ctrl+Return"), self)
        self._accept_shortcut.activated.connect(self.accept_review)
        self._redo_shortcut = QtGui.QShortcut(QtGui.QKeySequence("Ctrl+R"), self)
        self._redo_shortcut.activated.connect(self.redo_review)

        return frame

    # ======================================================================
    # Timers
    # ======================================================================

    def _start_timers(self) -> None:
        self.plot_timer = QtCore.QTimer(self)
        self.plot_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.plot_timer.timeout.connect(self._tick_plot)
        self.plot_timer.start(max(8, int(1000 / max(1, cfg.GUI_FPS))))

        self.seq_timer = QtCore.QTimer(self)
        self.seq_timer.setTimerType(QtCore.Qt.PreciseTimer)
        self.seq_timer.timeout.connect(self._tick_sequence)
        self.seq_timer.start(20)

        self.ui_timer = QtCore.QTimer(self)
        self.ui_timer.timeout.connect(self._tick_slow_ui)
        self.ui_timer.start(100)

        # Prediction runs on its own timer so its rate is independent of the
        # graph frame rate and of the recording state machine.
        self.pred_timer = QtCore.QTimer(self)
        self.pred_timer.timeout.connect(self._tick_prediction)

        # The neural net gets its OWN timer, not a share of the one above.
        # Two reasons: the two models are meant to be runnable independently
        # or together, and the net's cadence is a different question from the
        # classifier's -- its round trip crosses a process boundary, so a
        # slow worker must slow only itself down. It is created here and
        # started by START NN.
        self.nn_timer = QtCore.QTimer(self)
        self.nn_timer.timeout.connect(self._tick_neural)

        # Ask what hardware the training will run on. Deferred so the window
        # is already on screen and streaming before the probe starts.
        QtCore.QTimer.singleShot(400, self.start_compute_probe)
        #  A training run may have outlived the previous window.
        QtCore.QTimer.singleShot(600, self._resume_training_job)

    def _tick_plot(self) -> None:
        if self.plot.review_mode:
            return          # the graph is showing a finished take, not the stream

        n = int(self.plot.window_s * max(1.0, self.acq.stats()["rate"]
                                         or nominal_rate())) + 8
        emg, t = self.acq.ring.snapshot(n)
        if emg.shape[0] < 2:
            return
        self.plot.update_data(emg, t)

        # Either kind of capture gets the recording band: a free take is
        # every bit as much "this is being written down" as a cued take.
        capture = self.sequence.active_capture or self.free_capture
        if capture is not None and len(capture) > 1:
            span = len(capture) / max(1.0, self.acq.stats()["rate"] or nominal_rate())
            self.plot.set_recording_span(-min(span, self.plot.window_s), 0.0, True)
        else:
            self.plot.set_recording_span(0, 0, False)

    def _tick_sequence(self) -> None:
        self.sequence.tick()
        self._refresh_banner()

    def _tick_slow_ui(self) -> None:
        stats = self.acq.stats()

        self.status_conn.setText(f"{self.acq.state}")
        self.status_rate.setText(f"{stats['rate']:.0f} Hz")
        self.status_buffer.setText(f"buffer {stats['ring_fill'] * 100:.0f} %")
        # Show the input backlog: this is the number that reveals a lagging
        # link, and a rate readout alone cannot (a constant delay does not
        # change the spacing between samples).
        backlog = stats.get("backlog", 0)
        lag_s = backlog / max(1.0, 25.0 * (stats["rate"] or nominal_rate()))
        self.status_counts.setText(
            f"samples {stats['samples']:,}   dropped {stats['dropped']}   "
            f"reconnects {stats['reconnects']}   "
            f"lag {lag_s:.1f}s ({backlog}B)"
        )
        self.status_counts.setStyleSheet(
            f"color: {COL_BAD if lag_s > 1.0 else COL_WARN if lag_s > 0.35 else COL_TEXT_DIM};"
            " padding: 0 10px;"
        )

        last, mins, maxs, rms = self.plot.channel_stats()
        for i in range(cfg.N_CHANNELS):
            self.channel_labels[i][0].setText(f"{int(last[i])}")
            self.channel_labels[i][1].setText(f"{int(mins[i])}")
            self.channel_labels[i][2].setText(f"{int(maxs[i])}")
            self.channel_labels[i][3].setText(f"{rms[i]:.1f}")

        self._tick_free_take()
        # Throttled inside: a recording saved from anywhere -- the sequence,
        # a label-on-graph cut, a clip out of a take -- makes the model one
        # recording more out of date, and this is what notices.
        self._refresh_model_status()

        self.header_count.setText(str(len(self.recordings)))
        item = self.sequence.current
        self.header_gesture.setText(item[0] if item else
                                    self.gesture_combo.currentText())
        self.header_person.setText(f"{self.person_id} / {self.session_id}")

        done, total = self.sequence.progress
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(done)

        self._refresh_controls()

        # Clear a held one-shot result once its time is up.
        if self.one_shot_clear_at and time.perf_counter() > self.one_shot_clear_at:
            self.one_shot_clear_at = 0.0
            self.plot.set_prediction_overlay("")
            self.prediction_conf.setText("ready")

        self._update_plot_title()

    def _refresh_controls(self) -> None:
        """
        Enable/disable every control for the current state.

        Called from the 100 ms UI timer and directly by anything that changes
        state, so a button never stays wrong for a tenth of a second after the
        thing it controls has already happened.
        """
        running = self.sequence.running and not self.sequence.manual
        reviewing = self.sequence.reviewing
        # A free take owns the acquisition thread's single capture slot, so
        # nothing else may open a capture while one is running.
        taking = self.free_capture is not None
        self.start_button.setEnabled(not self.sequence.running and not taking)
        self.pause_button.setEnabled((running or self.sequence.paused)
                                     and not reviewing)
        self.pause_button.setText("RESUME" if self.sequence.paused else "PAUSE")
        self.stop_button.setEnabled(self.sequence.running)
        self.skip_button.setEnabled(running and not reviewing)
        self.record_button.setEnabled(not running and not self.sequence.paused
                                      and not reviewing and not taking)
        self.accept_button.setEnabled(reviewing)
        self.redo_button.setEnabled(reviewing)
        if cfg.FREE_TAKE_ENABLED and hasattr(self, "free_take_button"):
            self.free_take_button.setEnabled(not self.sequence.running)
            self.take_gesture_combo.setEnabled(not taking)
            self.open_take_button.setEnabled(not taking)
            # Deleting the take that is still being recorded would pull the
            # file out from under save_take(), so it waits until it is saved.
            self.delete_take_button.setEnabled(
                not taking and self.take_list.count() > 0
                and self.take_list.currentItem() is not None
                and bool(self.take_list.currentItem().data(
                    QtCore.Qt.UserRole + 1)))
        if cfg.CALIBRATION_ENABLED and hasattr(self, "calibration_button"):
            # Calibrating reads the live signal, so it waits for whatever is
            # currently recording -- same reason DELETE TAKE(S) does.
            self.calibration_button.setEnabled(
                not self.sequence.running and not taking)
        labelling = self.label_mode_button.isChecked()
        self.label_mode_button.setEnabled(not self.sequence.running
                                          or labelling)
        self.select_button.setEnabled(labelling)
        self.add_label_button.setEnabled(
            labelling and self.plot.selection_bounds() is not None)
        self.undo_label_button.setEnabled(labelling and bool(self.plot.marks))
        for widget in (self.person_edit, self.session_edit, self.reps_spin,
                       self.order_combo):
            widget.setEnabled(not self.sequence.running)

        # ---- visibility ------------------------------------------------
        # Enabling is not enough: eight permanently visible buttons, six of
        # them greyed out, is a wall to read past every time. A control that
        # cannot be used in this mode is not shown at all, so what is on
        # screen is exactly what you can do right now.
        idle = not self.sequence.running and not taking
        self.start_button.setVisible(idle)
        self.record_button.setVisible(idle)
        self.pause_button.setVisible(running or self.sequence.paused)
        self.stop_button.setVisible(self.sequence.running)
        self.skip_button.setVisible(running and not reviewing)
        self.accept_button.setVisible(reviewing)
        self.redo_button.setVisible(reviewing)

        # The banner is for "what should I do right now". When nothing is
        # happening it says READY, which is noise occupying 130 px above the
        # graph -- so it is hidden until it has something to report.
        self.banner.setVisible(
            self.sequence.running
            or taking
            or self.acq.state in (CONN_RECONNECTING, CONN_CONNECTING,
                                  CONN_ERROR)
        )

        self.trim_button.setEnabled(self.table.currentRow() >= 0)
        self.autotrim_button.setEnabled(bool(self.recordings))
        self.delete_button.setEnabled(bool(self.recordings))
        self.clear_button.setEnabled(bool(self.recordings))
        self.export_button.setEnabled(self.export_worker is None)
        self.retrain_button.setEnabled(
            (not self.sequence.running or self.training_process is not None)
            # One trainer at a time: the two sweeps would fight over the
            # machine, and the neural one also holds a CUDA context.
            and self.nn_training_process is None)
        if hasattr(self, "retrain_all_button"):
            #  While a chain runs this is its cancel button, in either step.
            chain = getattr(self, "_chain_job", None)
            idle = (self.training_process is None
                    and self.nn_training_process is None)
            self.retrain_all_button.setEnabled(
                chain is not None or (idle and not self.sequence.running))
            self.retrain_all_button.setText(
                "CANCEL ML + NN TRAINING" if chain is not None
                else "RETRAIN ML + NEURAL NET")
        if cfg.ENABLE_NEURAL_NET and hasattr(self, "nn_start_button"):
            self._refresh_neural_panel(force=False)
        if hasattr(self, "spec_button"):
            # Editing the spec mid-training would be misleading: the running
            # sweep already has its copy.
            self.spec_button.setEnabled(self.training_process is None)

    # ======================================================================
    # Connection
    # ======================================================================

    def _set_pill(self, text: str, color: str) -> None:
        self.conn_pill.setText(text)
        self.conn_pill.setStyleSheet(
            f"QLabel {{ background: {color}22; color: {color};"
            f" border: 1px solid {color}; border-radius: 11px;"
            f" padding: 4px 12px; font-weight: 700; letter-spacing: 1px; }}"
        )

    def _on_connection_state(self, state: str, message: str) -> None:
        colors = {
            CONN_CONNECTED: COL_GOOD,
            CONN_CONNECTING: COL_WARN,
            CONN_RECONNECTING: COL_WARN,
            CONN_DISCONNECTED: COL_TEXT_DIM,
            CONN_STOPPED: COL_TEXT_DIM,
            CONN_ERROR: COL_BAD,
        }
        text = "MYO " + state
        self._set_pill(text, colors.get(state, COL_TEXT_DIM))
        self._append_log(f"{state}: {message}" if message else state)
        self.status.showMessage(message or state, 6000)

        if not cfg.PAUSE_SESSION_ON_LINK_LOSS:
            return

        # A session must not keep counting repetitions into a dead link:
        # it would quietly fill the dataset with empty recordings.
        if state in (CONN_RECONNECTING, CONN_ERROR):
            if self.sequence.running and not self.sequence.paused \
                    and not self.sequence.reviewing:
                self._append_log("Link lost -- session paused automatically.")
                self.sequence.last_message = "Myo disconnected"
                self._link_paused = True
                # Throw away the partial recording and retry the repetition
                # rather than banking an incomplete sample.
                self.sequence.pause(discard_in_flight=True)

        # ...and it should pick up again by itself once the armband is back,
        # so an unattended subject is not left staring at a frozen screen.
        elif state == CONN_CONNECTED and self._link_paused:
            self._link_paused = False
            if self.sequence.paused:
                self._append_log("Myo reconnected -- resuming the session.")
                self.sequence.resume()

    def _append_log(self, text: str) -> None:
        self.log_box.appendPlainText(f"{datetime.now().strftime('%H:%M:%S')}  {text}")

    # ======================================================================
    # Identity / plan
    # ======================================================================

    @property
    def fitted_to_screen(self) -> bool:
        """True when the preferred window size did not fit on this display."""
        want = (cfg.WINDOW_WIDTH, cfg.WINDOW_HEIGHT)
        return tuple(getattr(self, "_fitted_size", want)) != want

    # ======================================================================
    # Banner
    # ======================================================================

    def _refresh_banner(self) -> None:
        if self.acq.state in (CONN_RECONNECTING, CONN_CONNECTING) and \
                (not self.sequence.running or self._link_paused):
            # A dead link outranks everything else: whatever is "recording"
            # is recording nothing. The open take is still named, so it is
            # clear that it was not the take that failed.
            message = self.acq.state_message or "Reconnecting..."
            if self.free_capture is not None:
                message = (f"free take {self.free_take_number:03d} is open and "
                           f"receiving nothing -- {message}")
            self.banner.update_state(
                STATE_RECONNECTING, "MYO DISCONNECTED", message, None,
            )
            self.plot.set_overlay("")
            return

        # A free take owns the banner while it runs: the sequence is idle, so
        # its own banner would read READY while the armband is recording.
        if self.free_capture is not None and not self.sequence.running:
            elapsed = time.time() - self.free_capture.started_unix
            cap = max(1.0, cfg.FREE_TAKE_MAX_MINUTES * 60.0)
            self.banner.update_state(
                STATE_RECORDING, "FREE TAKE -- RECORDING",
                f"{self.free_take_label}   {fmt_duration(elapsed)} of "
                f"{fmt_duration(cap)} max   {len(self.free_capture):,} samples"
                "   --   press STOP & SAVE TAKE when you are done",
                min(1.0, elapsed / cap),
            )
            self.plot.set_overlay("REC", "rgba(239,68,68,150)")
            return

        state = self.sequence.state
        big, small = self.sequence.banner()

        fraction = None
        if self.sequence.state_duration > 0 and not self.sequence.paused:
            fraction = self.sequence.elapsed / self.sequence.state_duration
        if state == STATE_RECORDING and self.sequence.manual:
            small = f"{self.sequence.current[0]}   {self.sequence.elapsed:.1f} s"

        self.banner.update_state(state, big, small, fraction)

        if state == STATE_COUNTDOWN:
            self.plot.set_overlay(big, "rgba(251,146,60,220)")
        elif state == STATE_RECORDING:
            self.plot.set_overlay("REC", "rgba(239,68,68,150)")
        elif state == STATE_PREPARE:
            self.plot.set_overlay(self.sequence.current[0]
                                  if self.sequence.current else "",
                                  "rgba(251,191,36,180)")
        else:
            self.plot.set_overlay("")

    def _update_plot_title(self) -> None:
        item = self.sequence.current
        gesture = item[0] if item else self.gesture_combo.currentText()
        self.plot.set_title(
            f"{cfg.APP_NAME}  |  {self.person_id} / {self.session_id}  |  "
            f"gesture: {gesture}  |  mode: {cfg.EMG_MODE} "
            f"@ {self.acq.stats()['rate']:.0f} Hz  |  "
            f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

    # ======================================================================
    # Retraining
    # ======================================================================

    def _on_section_toggled(self, section, opened: bool) -> None:
        """
        Bring a section the operator just opened into view.

        Opening the eighth section used to reveal nothing, because it unfolded
        below the fold of the scroll area. Qt needs the layout to settle
        before the new geometry is known, hence the single-shot timer.
        """
        if not opened:
            return
        QtCore.QTimer.singleShot(
            0, lambda: self._scroll_area.ensureWidgetVisible(section, 0, 0)
            if getattr(self, "_scroll_area", None) else None)

    def expand_section(self, key: str) -> None:
        """
        Unfold a side-panel section.

        Folding is a convenience, but a folded section also hides whatever
        happens inside it -- start a retrain with LIVE CLASSIFIER folded and
        the progress bar reports to nobody. Anything that begins showing
        progress calls this first.
        """
        section = getattr(self, "sections", {}).get(key)
        if section is not None and not section.expanded:
            section.set_expanded(True)

    # ======================================================================
    # Shutdown
    # ======================================================================

    def _on_freeze_toggled(self, frozen: bool) -> None:
        self.plot.set_frozen(frozen)
        self.freeze_button.setText("Resume graph" if frozen else "Freeze graph")

    def closeEvent(self, event) -> None:
        if self.sequence.running:
            answer = QtWidgets.QMessageBox.question(
                self, "Quit",
                "A recording session is still running. Quit anyway?\n\n"
                "Everything already saved stays on disk.",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                event.ignore()
                return

        #  A training job is NOT owned by the window: by default it keeps
        #  going, and the next window reattaches to it. Asked rather than
        #  assumed, because "stop it" is also a reasonable thing to want.
        job = getattr(self, "_job", None)
        if job is not None and not job.done:
            box = QtWidgets.QMessageBox(self)
            box.setWindowTitle("Training is running")
            box.setIcon(QtWidgets.QMessageBox.Question)
            box.setText("A training run is still going.")
            box.setInformativeText(
                "Keep it running in the background? It will finish and save "
                "its model on its own; open the app again any time to see "
                "the progress or the result.")
            keep = box.addButton("Keep training",
                                 QtWidgets.QMessageBox.AcceptRole)
            stop = box.addButton("Stop training and quit",
                                 QtWidgets.QMessageBox.DestructiveRole)
            box.addButton(QtWidgets.QMessageBox.Cancel)
            box.setDefaultButton(keep)
            box.exec()
            clicked = box.clickedButton()
            if clicked is stop:
                job.kill()
            elif clicked is not keep:
                event.ignore()
                return
            job.detach()
            self._append_log("Window closing -- training "
                             + ("cancelled." if clicked is stop
                                else "continues in the background."))
            self._job = None
            self.training_process = None
            self.nn_training_process = None

        self.sequence.stop()
        self.stop_prediction()
        if getattr(self, "model_test_dialog", None) is not None:
            self.model_test_dialog.close()
        # The net's worker holds a CUDA context, so it is shut down
        # explicitly rather than left to a dangling pipe: a subprocess that
        # outlives the window keeps the GPU busy and the checkpoint open.
        if self.nn_engine is not None:
            self.stop_neural()
        if self.compute_process is not None:
            try:
                self.compute_process.kill()
                self.compute_process.waitForFinished(1000)
            except RuntimeError:
                pass
            self.compute_process = None
        for timer in (self.plot_timer, self.seq_timer, self.ui_timer,
                      self.pred_timer, self.nn_timer):
            timer.stop()
        if self.export_worker is not None:
            self.export_worker.wait(3000)
        self.acq.stop()
        self.acq.join(timeout=3.0)
        super().closeEvent(event)
