"""
The cued recording sequence: build_plan and SequenceController (no Qt).
"""

import math
import random
import time

import numpy as np

from emg_app import config as cfg


# ==============================================================================
# SECTION 6 -- AUTOMATIC RECORDING SEQUENCE (state machine)
#
# Pure logic, no Qt and no timers of its own: the GUI calls tick() and the
# machine reports what should be on screen. That makes the whole recording
# protocol testable without an armband.
# ==============================================================================

STATE_IDLE = "IDLE"
STATE_REST = "REST"
STATE_PREPARE = "PREPARE"
STATE_COUNTDOWN = "COUNTDOWN"
STATE_RECORDING = "RECORDING"
STATE_STOPPING = "STOPPING"
STATE_REVIEW = "REVIEW"
STATE_SAVED = "SAVED"
STATE_PAUSED = "PAUSED"
STATE_DONE = "DONE"
STATE_ERROR = "ERROR"
STATE_RECONNECTING = "RECONNECTING"

# Colour per state, used by the big status banner.
STATE_COLORS = {
    STATE_IDLE: "#64748b",
    STATE_REST: "#38bdf8",
    STATE_PREPARE: "#fbbf24",
    STATE_COUNTDOWN: "#fb923c",
    STATE_RECORDING: "#ef4444",
    STATE_STOPPING: "#f97316",
    STATE_REVIEW: "#06b6d4",
    STATE_SAVED: "#22c55e",
    STATE_PAUSED: "#a78bfa",
    STATE_DONE: "#22c55e",
    STATE_ERROR: "#ef4444",
    STATE_RECONNECTING: "#f59e0b",
}


def build_plan(gestures=None, repetitions: int = None, order: str = None,
               seed: int = None):
    """
    Build the ordered list of (gesture, repetition) the session will collect.

    "interleaved" walks one repetition of every gesture before starting the
    next round, so fatigue and electrode drift are spread evenly across the
    label set instead of being baked into whichever gesture came last.
    """
    gestures = list(gestures if gestures is not None else cfg.GESTURES)
    repetitions = int(cfg.REPETITIONS if repetitions is None else repetitions)
    order = (order or cfg.GESTURE_ORDER).lower()
    rng = random.Random(cfg.RANDOM_SEED if seed is None else seed)

    if not gestures or repetitions <= 0:
        return []

    if order == "blocked":
        plan = [(g, r + 1) for g in gestures for r in range(repetitions)]
    elif order == "random":
        plan = [(g, r + 1) for g in gestures for r in range(repetitions)]
        rng.shuffle(plan)
    else:  # interleaved
        plan = []
        for r in range(repetitions):
            round_gestures = list(gestures)
            rng.shuffle(round_gestures)
            plan.extend((g, r + 1) for g in round_gestures)

    if cfg.MAX_RECORDINGS and len(plan) > cfg.MAX_RECORDINGS:
        plan = plan[:cfg.MAX_RECORDINGS]
    return plan


class SequenceController:
    """
    The REST -> GET READY -> 3 -> 2 -> 1 -> RECORD -> pause loop.

    Driven by repeated tick() calls (the GUI uses a 20 ms timer). All timing is
    wall-clock based, so a dropped frame delays nothing: the machine simply
    notices it is already past the deadline.

    Callbacks:
        on_capture_start(label, repetition) -> Capture
        on_capture_stop(capture, label, repetition)
        on_state(state, previous_state)
        on_finished()
    """

    def __init__(self, on_capture_start, on_capture_stop,
                 on_state=None, on_finished=None, on_capture_discard=None):
        self._on_capture_start = on_capture_start
        self._on_capture_stop = on_capture_stop
        self._on_state = on_state or (lambda state, previous: None)
        self._on_finished = on_finished or (lambda: None)
        self._on_capture_discard = on_capture_discard or (lambda cap, label, rep: None)

        self.plan = []
        self.index = 0
        self.state = STATE_IDLE
        self.state_entered = time.perf_counter()
        self.state_duration = 0.0
        self.paused = False
        self.pause_started = 0.0
        self.active_capture = None
        self.manual = False               # True = free-running manual recording
        self.last_message = ""

    # -- queries --------------------------------------------------------------

    @property
    def running(self) -> bool:
        return self.state not in (STATE_IDLE, STATE_DONE, STATE_ERROR)

    @property
    def current(self):
        """(label, repetition) of the item being collected, or None."""
        if 0 <= self.index < len(self.plan):
            return self.plan[self.index]
        return None

    @property
    def next_item(self):
        if 0 <= self.index + 1 < len(self.plan):
            return self.plan[self.index + 1]
        return None

    @property
    def progress(self):
        """(completed, total) over the whole plan."""
        return (min(self.index, len(self.plan)), len(self.plan))

    @property
    def elapsed(self) -> float:
        if self.paused:
            return self.pause_started - self.state_entered
        return time.perf_counter() - self.state_entered

    @property
    def remaining(self) -> float:
        if not self.state_duration:
            return 0.0
        return max(0.0, self.state_duration - self.elapsed)

    def banner(self):
        """
        (big_text, small_text) for the status banner -- the thing the subject
        actually looks at. Deliberately blunt and high-contrast.
        """
        item = self.current
        label = item[0] if item else ""

        if self.state == STATE_IDLE:
            return ("READY", "Press START SESSION")
        if self.state == STATE_PAUSED:
            return ("PAUSED", self.last_message or "Press RESUME to continue")
        if self.state == STATE_DONE:
            return ("SESSION COMPLETE", f"{self.progress[1]} recordings collected")
        if self.state == STATE_ERROR:
            return ("ERROR", self.last_message)
        if self.state == STATE_REST:
            return ("RELAX", f"Next: {label}")
        if self.state == STATE_PREPARE:
            return (f"GET READY: {label}",
                    cfg.gesture_cue(label) or "Hold the start position")
        if self.state == STATE_COUNTDOWN:
            ticks = max(1, int(math.ceil(self.remaining)))
            return (str(ticks), f"{label} in {ticks}")
        if self.state == STATE_RECORDING:
            cue = cfg.gesture_cue(label)
            return ("RECORD NOW", f"{label}  --  {cue}" if cue else label)
        if self.state == STATE_STOPPING:
            return ("STOP", label)
        if self.state == STATE_REVIEW:
            done, total = self.progress
            return (f"CHECK: {label}",
                    f"{done + 1} / {total}   --   ACCEPT & NEXT (Space) "
                    f"or REDO (Ctrl+R)")
        if self.state == STATE_SAVED:
            done, total = self.progress
            return ("SAVED", f"Repetition {done} / {total}")
        return (self.state, self.last_message)

    # -- control --------------------------------------------------------------

    def start(self, plan=None, start_index: int = 0) -> None:
        self.plan = list(plan) if plan is not None else build_plan()
        self.index = int(np.clip(start_index, 0, max(0, len(self.plan))))
        self.manual = False
        self.paused = False
        if not self.plan:
            self._enter(STATE_DONE, 0.0)
            return
        self._enter(STATE_REST, cfg.REST_TIME)

    def start_manual(self, label: str, repetition: int = 0) -> None:
        """
        Free-running single recording: starts immediately and runs until
        stop_manual(). Used for the manual RECORD button.
        """
        self.plan = [(label, repetition)]
        self.index = 0
        self.manual = True
        self.paused = False
        self._begin_capture()
        if self.active_capture is not None:
            self.active_capture.mark_label_start()
        self._enter(STATE_RECORDING, 0.0)       # 0 = no automatic end

    def stop_manual(self) -> None:
        if self.manual and self.state == STATE_RECORDING:
            self._finish_capture()
            if cfg.REVIEW_EACH_RECORDING:
                self._enter(STATE_REVIEW, 0.0)
            else:
                self.index += 1
                self._enter(STATE_SAVED, 0.6)

    # -- review step ----------------------------------------------------------

    @property
    def reviewing(self) -> bool:
        return self.state == STATE_REVIEW

    def accept_review(self) -> None:
        """Keep the take and move to the next repetition."""
        if self.state != STATE_REVIEW:
            return
        self.index += 1
        if self.manual:
            self._enter(STATE_IDLE, 0.0)
        elif self.index >= len(self.plan):
            self._enter(STATE_DONE, 0.0)
            self._on_finished()
        else:
            self._enter(STATE_REST, cfg.REST_TIME)

    def redo_review(self) -> None:
        """
        Reject the take and record the SAME repetition again.

        The index is not advanced, so the plan still owes you this gesture.
        Quarantining the rejected file is the caller's job -- this class never
        touches the filesystem.
        """
        if self.state != STATE_REVIEW:
            return
        if self.manual:
            self._enter(STATE_IDLE, 0.0)
            return
        self._enter(STATE_REST, cfg.REST_TIME)

    def pause(self, discard_in_flight: bool = False) -> None:
        """
        Pause between repetitions.

        A recording in flight is normally finished and handed over -- freezing
        mid-gesture would produce a truncated sample carrying a full label, and
        the operator pressing PAUSE usually did so between repetitions anyway.

        `discard_in_flight=True` is for an involuntary pause (the Myo link
        dropped). The partial recording is thrown away and the index is NOT
        advanced, so the same repetition is retried once the link is back.
        Otherwise a dropout would silently cost one sample of that gesture.
        """
        if not self.running or self.paused:
            return
        if self.state == STATE_REVIEW:
            return          # already waiting for the operator; nothing to pause

        if self.state in (STATE_RECORDING, STATE_STOPPING):
            if discard_in_flight:
                self._discard_capture()
            else:
                self._finish_capture()
                self.index += 1

        self.paused = True
        self.pause_started = time.perf_counter()
        self._enter(STATE_PAUSED, 0.0)

    def resume(self) -> None:
        if not self.paused:
            return
        self.paused = False
        self.last_message = ""
        if self.index >= len(self.plan):
            self._enter(STATE_DONE, 0.0)
        else:
            self._enter(STATE_REST, cfg.REST_TIME)

    def stop(self) -> None:
        """Abort the session. Any in-flight recording is still handed over."""
        if self.state in (STATE_RECORDING, STATE_STOPPING):
            self._finish_capture()
        self.paused = False
        self.active_capture = None
        self._enter(STATE_IDLE, 0.0)

    def skip(self) -> None:
        """Drop the current item and move to the next one."""
        if self.state in (STATE_RECORDING, STATE_STOPPING):
            self._finish_capture()
        self.index += 1
        if self.index >= len(self.plan):
            self._enter(STATE_DONE, 0.0)
        else:
            self._enter(STATE_REST, cfg.REST_TIME)

    # -- the clock ------------------------------------------------------------

    def tick(self) -> None:
        """
        Advance the machine. Safe (and cheap) to call at any rate.

        The capture window is deliberately WIDER than the labeled window:

            |<- PRE_GESTURE_MS ->|<---- RECORD_TIME ---->|<- POST_GESTURE_MS ->|
            capture opens         label window opens      label closes   capture closes

        So a subject who reacts a fraction early, or an explosive contraction
        that overshoots the cue, is still fully inside the raw file.
        """
        if self.paused or not self.running:
            return

        # Pre-roll: open the buffer before the countdown finishes.
        if self.state == STATE_COUNTDOWN and self.active_capture is None:
            if self.remaining <= cfg.PRE_GESTURE_MS / 1000.0:
                self._begin_capture()

        if self.state_duration <= 0.0:
            return                              # manual recording: no deadline
        if self.elapsed < self.state_duration:
            return

        if self.state == STATE_REST:
            self._enter(STATE_PREPARE, max(0.2, cfg.PREPARE_TIME))
        elif self.state == STATE_PREPARE:
            self._enter(STATE_COUNTDOWN, cfg.COUNTDOWN_TIME)
        elif self.state == STATE_COUNTDOWN:
            if self.active_capture is None:
                self._begin_capture()
            if self.active_capture is not None:
                self.active_capture.mark_label_start()
            self._enter(STATE_RECORDING, cfg.RECORD_TIME)
        elif self.state == STATE_RECORDING:
            # Label window closes here, but the buffer keeps running so the
            # relaxation phase of the gesture is preserved in the raw file.
            if self.active_capture is not None:
                self.active_capture.mark_label_end()
            self._enter(STATE_STOPPING, max(0.05, cfg.POST_GESTURE_MS / 1000.0))
        elif self.state == STATE_STOPPING:
            self._finish_capture()
            if cfg.REVIEW_EACH_RECORDING:
                # Hand control over. Duration 0 = no deadline: the machine
                # sits here until accept_review() or redo_review() is called.
                self._enter(STATE_REVIEW, 0.0)
            else:
                self.index += 1
                self._enter(STATE_SAVED, max(0.3, cfg.PAUSE_TIME))
        elif self.state == STATE_SAVED:
            if self.index >= len(self.plan):
                self._enter(STATE_DONE, 0.0)
                self._on_finished()
            elif cfg.AUTO_ADVANCE:
                self._enter(STATE_REST, cfg.REST_TIME)
            else:
                self.pause()

    # -- internals ------------------------------------------------------------

    def _enter(self, state: str, duration: float) -> None:
        previous = self.state
        self.state = state
        self.state_duration = float(duration)
        self.state_entered = time.perf_counter()
        if previous != state:
            self._on_state(state, previous)

    def _begin_capture(self) -> None:
        item = self.current
        if item is None:
            return
        label, repetition = item
        self.active_capture = self._on_capture_start(label, repetition)

    def _finish_capture(self) -> None:
        """
        Close the capture and hand it over.

        If the label window was never closed (abort, pause, manual stop) it is
        closed here, so a recording can never carry an open-ended label.
        """
        item = self.current
        capture = self.active_capture
        self.active_capture = None
        if capture is None or item is None:
            return
        if capture.label_end_index is None:
            capture.mark_label_end()
        self._on_capture_stop(capture, item[0], item[1])

    def _discard_capture(self) -> None:
        """Abandon an incomplete recording without saving it."""
        item = self.current
        capture = self.active_capture
        self.active_capture = None
        if capture is None or item is None:
            return
        self._on_capture_discard(capture, item[0], item[1])
