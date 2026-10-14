"""
FREE TAKE: record one long stretch now, cut labelled clips out of it later.

No armband and no display needed (Qt runs offscreen).

Two properties matter more than the rest and are each checked twice:

  * a take is NEVER training data. It lives in `_takes/`, carries stage
    "take", and `load_dataset()` must not see it -- otherwise five minutes of
    mostly-rest signal would be loaded as if it were a gesture.
  * cutting a clip out of a take does not modify the take. The take file stays
    byte-identical, and the clip's samples are bit-exact copies of the take's
    -- safety rules 1 and 3 in CLAUDE.md.
"""
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import emg_dataset_tool as T

OK = []


def check(name, condition, detail=""):
    OK.append(bool(condition))
    print(f"{'PASS' if condition else 'FAIL'}  {name}"
          + (f"  [{detail}]" if detail else ""))


def synth_take(seconds=20.0, fs=50.0, seed=7):
    """
    A take that looks like a real one: long stretches of rest with four
    bursts in it, which is exactly the shape that must never be loaded as a
    single labelled gesture.
    """
    rng = np.random.default_rng(seed)
    n = int(seconds * fs)
    t = np.arange(n) / fs
    emg = (np.abs(rng.normal(0, 8, (n, 8))) + 20).astype(np.int32)
    for k in range(4):
        lo = int((2.0 + k * 4.0) * fs)
        hi = min(n, lo + int(1.5 * fs))
        if hi - lo < 2:                      # a short take has fewer bursts
            break
        emg[lo:hi] += rng.integers(300, 900, (hi - lo, 8)).astype(np.int32)
    return emg, t


# ==============================================================================
# PART 1 -- data layer, no Qt
# ==============================================================================

def test_data_layer():
    print("\n--- data layer ---")
    root = tempfile.mkdtemp(prefix="emg_take_")
    try:
        w = T.DatasetWriter(root, "p1", "s1")

        check("the take folder is underscore-prefixed",
              T.FREE_TAKE_FOLDER.startswith("_"), T.FREE_TAKE_FOLDER)
        check("takes_dir sits inside the session",
              os.path.dirname(w.takes_dir) == w.session_dir)
        check("'take' is a stage of its own", T.STAGE_TAKE == "take")
        check("next_take_number starts at 1", w.next_take_number() == 1)

        emg, t = synth_take(20.0)
        t0 = time.time() - 20.0
        rec = T.Recording(emg=emg, timestamps=t0 + t, label="FIST",
                          person_id="p1", session_id="s1", repetition=0,
                          stage=T.STAGE_RAW,
                          meta={"capture_style": "free_take",
                                "intended_gesture": "FIST"})
        written = w.save_take(rec, 1)

        check("stage flipped to take on save", rec.stage == T.STAGE_TAKE)
        check("csv written", "csv" in written and os.path.exists(written["csv"]))
        check("npz written", "npz" in written and os.path.exists(written["npz"]))
        check("meta written", "meta" in written and os.path.exists(written["meta"]))
        check("written into the takes folder",
              os.path.dirname(written["npz"]) == w.takes_dir,
              os.path.basename(os.path.dirname(written["npz"])))
        name = os.path.basename(written["npz"])
        check("filename carries the take stage", name.endswith("_take.npz"), name)
        check("filename carries the take number", "_take001_" in name, name)
        check("filename carries the intended gesture",
              name.startswith("FIST_"), name)
        check("stage reads back out of the filename",
              T._stage_from_filename(written["npz"]) == T.STAGE_TAKE)

        # -- THE invariant: a take is not training data ----------------------
        found_raw = T.discover_recordings(root, stage=T.STAGE_RAW)
        found_any = T.discover_recordings(root, stage="prefer_trimmed")
        check("discovery (raw) does not see the take", found_raw == [],
              f"{len(found_raw)} found")
        check("discovery (prefer_trimmed) does not see the take",
              found_any == [], f"{len(found_any)} found")
        check("load_dataset does not see the take",
              T.load_dataset(root, stage="prefer_trimmed") == [])

        # -- round trip -------------------------------------------------------
        back = T.load_recording(written["npz"])
        check("take reloads", back.n_samples == rec.n_samples,
              f"{back.n_samples} vs {rec.n_samples}")
        check("take reloads bit-exact", np.array_equal(back.emg, emg))
        check("reloaded stage is take", back.stage == T.STAGE_TAKE, back.stage)
        check("reloaded label is the intended gesture", back.label == "FIST")
        check("reloaded person/session",
              back.person_id == "p1" and back.session_id == "s1")
        back_csv = T.load_recording(written["csv"])
        check("csv copy is bit-exact too", np.array_equal(back_csv.emg, emg))

        # -- manifest ---------------------------------------------------------
        entries = w.read_manifest()
        takes = [e for e in entries if e.get("stage") == T.STAGE_TAKE]
        check("manifest got one take line", len(takes) == 1)
        check("manifest records the duration",
              abs(takes[0]["duration_s"] - rec.duration) < 0.01)
        check("a take does not count as a recording",
              w.count_existing() == 0, str(w.count_existing()))
        check("a take does not bump repetition numbers",
              w.next_repetition("FIST") == 1)
        check("next_take_number advanced", w.next_take_number() == 2)

        # -- listing ----------------------------------------------------------
        listed = w.list_takes()
        check("list_takes finds it", len(listed) == 1)
        check("listing reports the label", listed[0]["label"] == "FIST")
        check("listing reports the sample count",
              listed[0]["n_samples"] == rec.n_samples)
        check("listing reports the duration",
              abs(listed[0]["duration_s"] - rec.duration) < 0.01)
        check("listing prefers npz for loading",
              listed[0]["path"].endswith(".npz"), listed[0]["path"])

        # -- a second take does not overwrite the first ----------------------
        emg2, t2 = synth_take(8.0, seed=11)
        rec2 = T.Recording(emg=emg2, timestamps=time.time() + t2, label="OPEN",
                           person_id="p1", session_id="s1", repetition=0,
                           stage=T.STAGE_RAW)
        written2 = w.save_take(rec2)
        check("second take written", os.path.exists(written2["npz"]))
        check("first take still there", os.path.exists(written["npz"]))
        check("the two takes are different files",
              written2["npz"] != written["npz"])
        check("second take numbered 002",
              "_take002_" in os.path.basename(written2["npz"]),
              os.path.basename(written2["npz"]))
        listed = w.list_takes()
        check("both takes listed", len(listed) == 2)
        check("newest first", listed[0]["created_unix"] >= listed[1]["created_unix"])
        check("still nothing in the dataset",
              T.load_dataset(root, stage="prefer_trimmed") == [])

        # -- a damaged sidecar must not break the listing ---------------------
        sidecar = os.path.join(w.takes_dir,
                               os.path.splitext(os.path.basename(
                                   written2["npz"]))[0] + "_meta.json")
        check("sidecar exists for take 2", os.path.exists(sidecar))
        with open(sidecar, "w", encoding="utf-8") as handle:
            handle.write("{ not json")
        listed = w.list_takes()
        check("damaged sidecar tolerated", len(listed) == 2, str(len(listed)))
        check("label falls back to the filename",
              any(i["label"] == "OPEN" for i in listed))
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ==============================================================================
# PART 2 -- the GUI, offscreen, with the armband pointed at a dead port
# ==============================================================================

def test_gui():
    print("\n--- gui ---")
    from PySide6 import QtWidgets

    root = tempfile.mkdtemp(prefix="emg_gui_take_")
    T.DATA_FOLDER = root
    T.PROJECT_DIR = root
    T.PERSON_ID = "tester"
    T.SESSION_ID = "sess_ft"
    T.MYO_PORT = "COM_DOES_NOT_EXIST"      # keep the real armband out of this
    T.LABEL_DEFAULT_SELECTION_S = 2.0

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    T.apply_theme(app)
    window = None
    try:
        window = T.MainWindow()
        w = window

        # -- the panel --------------------------------------------------------
        check("free take button built", hasattr(w, "free_take_button"))
        check("intended gesture combo built", hasattr(w, "take_gesture_combo"))
        check("take list built", hasattr(w, "take_list"))
        check("open take button built", hasattr(w, "open_take_button"))
        check("status label built", hasattr(w, "free_take_status"))
        check("no take running at startup", w.free_capture is None)
        check("status says idle", "idle" in w.free_take_status.text())
        check("history says there are none",
              "no takes" in w.take_list.item(0).text())

        # -- the default cut length is settable -------------------------------
        check("cut length spin built", hasattr(w, "cut_length_spin"))
        check("spin starts at the configured default",
              abs(w.cut_length_spin.value() - 2.0) < 1e-9)
        check("button text shows the length",
              "2.00 s" in w.select_button.text(), w.select_button.text())
        w.cut_length_spin.setValue(3.5)
        check("spin mirrors into the config",
              abs(T.LABEL_DEFAULT_SELECTION_S - 3.5) < 1e-9,
              str(T.LABEL_DEFAULT_SELECTION_S))
        check("button text followed",
              "3.50 s" in w.select_button.text(), w.select_button.text())
        check("the new length reaches the snapshot",
              abs(T.recording_config_snapshot()["default_cut_seconds"] - 3.5) < 1e-9)

        rng = np.random.default_rng(3)
        now = time.time()
        for i in range(400):
            w.acq.ring.push(now - 8.0 + i * 0.02,
                            (np.abs(rng.normal(0, 20, 8)) + 25).astype(int))
        w.arm_selection()
        bounds = w.plot.selection_bounds()
        check("a live selection uses the configured length",
              bounds is not None and abs((bounds[1] - bounds[0]) - 3.5) < 1e-6,
              f"{bounds[1] - bounds[0]:.3f}s" if bounds else "none")
        w.label_mode_button.setChecked(False)
        w.cut_length_spin.setValue(2.0)

        # -- starting a take --------------------------------------------------
        w.take_gesture_combo.setCurrentText("FIST")
        w.free_take_button.setChecked(True)          # drives toggle_free_take
        check("take started", w.free_capture is not None)
        check("button became a stop button",
              "STOP" in w.free_take_button.text(), w.free_take_button.text())
        check("status says recording",
              "RECORDING" in w.free_take_status.text())
        check("intended gesture recorded", w.free_take_label == "FIST")
        check("take numbered", w.free_take_number == 1)
        check("the capture carries the intended gesture",
              w.free_capture.meta.get("intended_gesture") == "FIST")
        check("capture_style marks it a free take",
              w.free_capture.meta.get("capture_style") == "free_take")

        cap_rate = T.nominal_rate(w.acq.mode)
        check("the capture cap is the FREE TAKE cap, not the 120 s one",
              w.free_capture.max_samples
              > T.MAX_RECORDING_SECONDS * cap_rate * 1.5,
              f"{w.free_capture.max_samples} samples")
        check("the cap matches FREE_TAKE_MAX_MINUTES",
              abs(w.free_capture.max_samples
                  - T.FREE_TAKE_MAX_MINUTES * 60.0 * cap_rate * 1.5) < 2,
              f"{w.free_capture.max_samples}")

        # nothing else may own the capture slot
        w._refresh_controls()
        check("START SESSION blocked during a take",
              not w.start_button.isEnabled())
        check("manual RECORD blocked during a take",
              not w.record_button.isEnabled())
        check("the gesture cannot be changed mid-take",
              not w.take_gesture_combo.isEnabled())

        # A dead link outranks the take in the banner -- but it must name the
        # take, so it is clear the take itself did not fail.
        w._refresh_banner()
        check("a dead link still outranks the take in the banner",
              "DISCONNECTED" in w.banner.big.text(), w.banner.big.text())
        check("but the open take is named",
              "free take" in w.banner.small.text(), w.banner.small.text())

        # With a healthy link the take owns the banner.
        # `state` is read-only on purpose (only the acquisition thread sets
        # it), so the test writes the backing field the thread writes.
        saved_state = w.acq._state
        w.acq._state = T.CONN_CONNECTED
        w._refresh_banner()
        check("banner announces the free take",
              "FREE TAKE" in w.banner.big.text(), w.banner.big.text())
        check("banner shows the gesture and the clock",
              "FIST" in w.banner.small.text(), w.banner.small.text())
        check("graph shows the REC overlay",
              w.plot.overlay.text() == "REC", w.plot.overlay.text())
        w.acq._state = saved_state

        # -- fill it, the way the acquisition thread does ---------------------
        emg, t_rel = synth_take(20.0)
        start_unix = time.time()
        for i in range(emg.shape[0]):
            w.free_capture.append(start_unix + float(t_rel[i]), emg[i])
        check("capture collected every sample",
              len(w.free_capture) == emg.shape[0],
              f"{len(w.free_capture)}")

        w._tick_free_take()
        check("status reports the elapsed time",
              "RECORDING" in w.free_take_status.text())
        check("detail reports the sample count",
              "samples" in w.free_take_detail.text())

        # -- stopping saves it ------------------------------------------------
        w.free_take_button.setChecked(False)         # drives stop_free_take
        check("capture released", w.free_capture is None)
        check("button reset", w.free_take_button.text() == "START FREE TAKE")
        check("status says saved", "saved" in w.free_take_status.text(),
              w.free_take_status.text())
        check("capture slot given back to the acquisition thread",
              not w.acq.capturing)

        takes = w.writer.list_takes()
        check("the take is on disk", len(takes) == 1)
        take_path = takes[0]["path"] if takes else ""
        check("history list shows it", w.take_list.count() == 1
              and "FIST" in w.take_list.item(0).text())
        check("history row carries the path",
              w.take_list.item(0).data(T.QtCore.Qt.UserRole) == take_path)

        loaded = T.load_recording(take_path)
        check("the saved take is bit-exact", np.array_equal(loaded.emg, emg))
        check("the saved take is the whole stretch",
              loaded.n_samples == emg.shape[0], f"{loaded.n_samples}")
        check("THE TAKE IS NOT TRAINING DATA",
              T.load_dataset(root, stage="prefer_trimmed") == [],
              f"{len(T.load_dataset(root, stage='prefer_trimmed'))} found")

        # -- a take too short to be worth a file ------------------------------
        before = len(w.writer.list_takes())
        w.free_take_button.setChecked(True)
        w.free_capture.append(time.time(), emg[0])
        w.free_take_button.setChecked(False)
        check("a too-short take is discarded",
              len(w.writer.list_takes()) == before,
              f"{len(w.writer.list_takes())}")
        check("and says so", "discarded" in w.free_take_status.text(),
              w.free_take_status.text())

        # -- cutting clips out of it ------------------------------------------
        rec = T.load_recording(take_path)
        rec.source = take_path
        writer = T.DatasetWriter(
            T.MainWindow._dataset_root_for_take(take_path),
            rec.person_id, rec.session_id)
        check("dataset root derived from a take path",
              os.path.normcase(writer.root) == os.path.normcase(w.writer.root),
              writer.root)

        T.LABEL_DEFAULT_SELECTION_S = 1.5
        dialog = T.TrimDialog(rec, writer, w, extract_mode=True)
        check("dialog is in extract mode", dialog.extract_mode is True)
        check("extract panel built", hasattr(dialog, "extract_button"))
        check("clip length spin built", hasattr(dialog, "clip_length_spin"))
        check("clip length starts at the configured default",
              abs(dialog.clip_length - 1.5) < 1e-9, str(dialog.clip_length))
        check("gesture pre-set from the take",
              dialog.extract_gesture.currentText() == "FIST")
        check("no clips yet", dialog.extracted == [])
        check("clip list says so",
              "no clips" in dialog.clip_list.item(0).text())
        check("opened at the start of the take, not on auto-detect",
              abs(dialog.start_spin.value()) < 1e-6,
              f"{dialog.start_spin.value():.3f}")
        check("opening selection is one clip long",
              abs((dialog.end_spin.value() - dialog.start_spin.value()) - 1.5) < 0.05,
              f"{dialog.end_spin.value() - dialog.start_spin.value():.3f}")

        take_bytes = open(take_path, "rb").read()

        # a selection below the minimum must be refused
        dialog._set_selection(0.0, 0.05)
        check("a too-short clip is refused", dialog.extract_clip() is None)
        check("and says why", "too short" in dialog.extract_status.text()
              or "minimum" in dialog.extract_status.text(),
              dialog.extract_status.text())
        check("nothing was written", dialog.extracted == [])

        # cut the first burst out: it sits at 2.0-3.5 s in synth_take()
        dialog._set_selection(2.0, 3.5)
        start_i, end_i = dialog.selection_indices()
        clip = dialog.extract_clip()
        check("clip extracted", clip is not None)
        check("clip is a training stage", clip.stage == T.STAGE_RAW, clip.stage)
        check("clip took the chosen label", clip.label == "FIST")
        check("clip got repetition 1", clip.repetition == 1, str(clip.repetition))
        check("CLIP SAMPLES ARE BIT-EXACT COPIES OF THE TAKE",
              np.array_equal(clip.emg, emg[start_i:end_i]))
        check("clip timestamps come from the take",
              np.array_equal(clip.timestamps, rec.timestamps[start_i:end_i]))
        check("clip has its own uid", clip.uid != rec.uid)
        check("clip records which take it came from",
              clip.meta.get("parent_take_uid") == rec.uid)
        check("clip records where in the take",
              clip.meta.get("take_start_index") == start_i
              and clip.meta.get("take_end_index") == end_i)
        check("clip records the take's label",
              clip.meta.get("parent_take_label") == "FIST")
        check("capture_style marks it cut from a take",
              clip.meta.get("capture_style") == "cut_from_take")
        check("the whole clip is the gesture (no pre/post roll)",
              clip.label_start_index == 0
              and clip.label_end_index == clip.n_samples)
        check("THE TAKE FILE IS UNTOUCHED",
              open(take_path, "rb").read() == take_bytes)

        clip_path = clip.primary_path(writer.root)
        check("clip is on disk", bool(clip_path) and os.path.exists(clip_path))
        check("clip went into its gesture folder, not _takes",
              os.path.basename(os.path.dirname(clip_path)) == "FIST",
              os.path.dirname(clip_path))
        check("clip list shows it", dialog.clip_list.count() == 1)
        check("clip band drawn on the graph", len(dialog._clip_bands) > 0)
        check("status names the file", "wrote" in dialog.extract_status.text())

        # extracting steps the window forward so the next cut is a new span
        check("selection stepped past the clip",
              dialog.start_spin.value() >= 3.4,
              f"{dialog.start_spin.value():.3f}")

        # -- a second clip, a different gesture -------------------------------
        dialog.extract_gesture.setCurrentText("OPEN")
        dialog._set_selection(6.0, 7.5)
        clip2 = dialog.extract_clip()
        check("second clip extracted", clip2 is not None)
        check("second clip took the new label", clip2.label == "OPEN")
        check("second clip is repetition 1 of ITS label",
              clip2.repetition == 1, str(clip2.repetition))
        check("second clip has its own uid",
              clip2.uid != clip.uid and clip2.uid != rec.uid)
        clip2_path = clip2.primary_path(writer.root)
        check("second clip went to the OPEN folder",
              os.path.basename(os.path.dirname(clip2_path)) == "OPEN")
        check("both clips tracked", len(dialog.extracted) == 2)
        check("clip list shows both", dialog.clip_list.count() == 2)

        # -- a third, of the same label as the first --------------------------
        dialog.extract_gesture.setCurrentText("FIST")
        dialog._set_selection(10.0, 11.5)
        clip3 = dialog.extract_clip()
        check("third clip extracted", clip3 is not None)
        check("repetition numbers increment per label",
              clip3.repetition == 2, str(clip3.repetition))
        check("THE TAKE FILE IS STILL UNTOUCHED",
              open(take_path, "rb").read() == take_bytes)

        # -- stepping ---------------------------------------------------------
        dialog.clip_length_spin.setValue(2.0)
        check("clip length mirrors into the config",
              abs(T.LABEL_DEFAULT_SELECTION_S - 2.0) < 1e-9)
        dialog.select_first_clip()
        check("FIRST jumps to the start",
              abs(dialog.start_spin.value()) < 1e-6)
        width = dialog.end_spin.value() - dialog.start_spin.value()
        check("FIRST uses the clip length", abs(width - 2.0) < 0.05,
              f"{width:.3f}")
        dialog.step_selection(+1)
        check("NEXT advances by one clip length",
              abs(dialog.start_spin.value() - 2.0) < 0.05,
              f"{dialog.start_spin.value():.3f}")
        check("NEXT keeps the width",
              abs((dialog.end_spin.value() - dialog.start_spin.value()) - 2.0) < 0.05)
        dialog.step_selection(-1)
        check("PREV goes back", abs(dialog.start_spin.value()) < 0.05,
              f"{dialog.start_spin.value():.3f}")
        dialog.step_selection(-1)
        check("PREV clamps at the start of the take",
              dialog.start_spin.value() >= -1e-6,
              f"{dialog.start_spin.value():.3f}")
        for _ in range(60):
            dialog.step_selection(+1)
        check("NEXT clamps at the end of the take",
              dialog.end_spin.value() <= rec.duration + 1e-6,
              f"{dialog.end_spin.value():.3f} of {rec.duration:.3f}")

        dialog.fit_selection_to_length()
        check("FIT LENGTH keeps the left edge",
              abs((dialog.end_spin.value() - dialog.start_spin.value()) - 2.0) < 0.05
              or dialog.start_spin.value() + 2.0 > rec.duration)
        dialog.close()

        # -- the clips ARE training data --------------------------------------
        recs = T.load_dataset(root, stage="prefer_trimmed")
        check("THE CLIPS ARE TRAINING DATA", len(recs) == 3, f"{len(recs)}")
        labels = sorted(r.label for r in recs)
        check("with the labels they were given",
              labels == ["FIST", "FIST", "OPEN"], str(labels))
        check("the take is still excluded",
              all(r.stage != T.STAGE_TAKE for r in recs))
        check("loaded clip is bit-exact against the take",
              any(np.array_equal(r.emg, emg[start_i:end_i]) for r in recs))

        X, y, groups, meta = T.make_windows(recs)
        check("the clips window for training", X.shape[0] > 0, str(X.shape))
        check("windows carry the person for leave-one-person-out",
              set(groups) == {"tester"}, str(set(groups)))

        # -- opening a take through the normal route --------------------------
        # This is what a real click does: open_take_for_cutting() loads the
        # take, runs the dialog, and absorbs whatever came out of it. Only
        # exec() is stood in for, because it blocks on a modal window.
        check("a take path is recognised as a take",
              T._stage_from_filename(take_path) == T.STAGE_TAKE)
        check("open_take_for_cutting exists",
              callable(getattr(w, "open_take_for_cutting", None)))

        recs_before = len(w.recordings)
        rows_before = w.table.rowCount()
        real_exec = T.TrimDialog.exec

        def fake_exec(dialog):
            check("the route built an extract-mode dialog",
                  dialog.extract_mode is True)
            dialog.extract_gesture.setCurrentText("WAVE_LEFT")
            dialog._set_selection(14.0, 15.5)
            dialog.extract_clip()
            dialog.extract_gesture.setCurrentText("FIST")
            dialog._set_selection(17.0, 18.5)
            dialog.extract_clip()
            return 1

        T.TrimDialog.exec = fake_exec
        try:
            w.open_take_for_cutting(take_path)
        finally:
            T.TrimDialog.exec = real_exec

        check("clips land in the recordings list",
              len(w.recordings) == recs_before + 2,
              f"{len(w.recordings)} vs {recs_before}")
        check("CLIPS ARE VISIBLE IN THE RECORDINGS TABLE",
              w.table.rowCount() == rows_before + 2,
              f"{w.table.rowCount()} vs {rows_before}")
        last_row = w.table.rowCount() - 1
        check("the table row names the gesture",
              w.table.item(last_row, 1).text() == "FIST",
              w.table.item(last_row, 1).text())
        check("the table row shows the repetition",
              w.table.item(last_row, 2).text()
              == str(w.recordings[-1].repetition),
              w.table.item(last_row, 2).text())
        check("the table row shows the duration",
              "s" in w.table.item(last_row, 3).text(),
              w.table.item(last_row, 3).text())
        check("the earlier of the two is in the table too",
              w.table.item(last_row - 1, 1).text() == "WAVE_LEFT",
              w.table.item(last_row - 1, 1).text())

        check("the on-disk counts updated",
              "WAVE_LEFT 1" in w.label_counts.text(), w.label_counts.text())
        check("the counts show every label cut from the take",
              "FIST 3" in w.label_counts.text(), w.label_counts.text())
        check("the status line reports how many were extracted",
              "2 clip(s)" in w.free_take_status.text(),
              w.free_take_status.text())
        check("the log names each clip",
              "Extracted 2 clip(s)" in w.log_box.toPlainText())

        check("and they are on disk as training data",
              len(T.load_dataset(root, stage="prefer_trimmed")) == 5,
              f"{len(T.load_dataset(root, stage='prefer_trimmed'))}")
        check("the take is STILL not training data",
              all(r.stage != T.STAGE_TAKE
                  for r in T.load_dataset(root, stage="prefer_trimmed")))

        # -- markers travel with a take ---------------------------------------
        # A marker takes its timestamp from the newest RING sample, and the
        # acquisition thread pushes every sample to the ring AND to the open
        # capture. The test has to do both, or the two timebases drift apart
        # and no marker would ever land inside a take.
        def feed(ts):
            sample = (np.abs(rng.normal(0, 20, 8)) + 25).astype(int)
            w.acq.ring.push(ts, sample)
            if w.free_capture is not None:
                w.free_capture.append(ts, sample)

        base = time.time()
        for i in range(50):
            feed(base + i * 0.02)                 # before the take starts
        w.marker_text.setText("before the take")
        w.add_note_marker()
        outside_uid = w.session_markers[-1].uid
        check("a marker can be dropped while idle",
              len(w.session_markers) == 1)

        w.free_take_button.setChecked(True)
        marked_start = base + 2.0
        for i in range(150):
            feed(marked_start + i * 0.02)
        w.add_break_marker()
        inside_uid = w.session_markers[-1].uid
        check("the break landed inside the take's span",
              marked_start <= w.session_markers[-1].t_unix,
              f"{w.session_markers[-1].t_unix - marked_start:+.3f}s")
        for i in range(150):
            feed(marked_start + 3.0 + i * 0.02)
        w.free_take_button.setChecked(False)

        takes = w.writer.list_takes()
        check("second real take saved", len(takes) == 2, str(len(takes)))
        newest = T.load_recording(takes[0]["path"])
        travelled = newest.meta.get("markers")
        check("markers inside the take travel in its metadata",
              isinstance(travelled, list) and len(travelled) >= 1,
              str(travelled))
        uids = [m.get("uid") for m in (travelled or [])]
        check("the marker that travelled is the one inside the take",
              inside_uid in uids, str(uids))
        check("a marker from before the take does NOT travel with it",
              outside_uid not in uids, str(uids))
        check("still nothing extra in the dataset",
              len(T.load_dataset(root, stage="prefer_trimmed")) == 5,
              f"{len(T.load_dataset(root, stage='prefer_trimmed'))}")

        # -- windows fit the screen they open on ------------------------------
        avail_w, avail_h = T.available_screen_size(w)
        check("a screen size is always reported",
              avail_w >= 640 and avail_h >= 480, f"{avail_w}x{avail_h}")
        check("the main window is no wider than the screen",
              w.width() <= avail_w, f"{w.width()} vs {avail_w}")
        check("the main window is no taller than the screen",
              w.height() <= avail_h, f"{w.height()} vs {avail_h}")
        check("THE MINIMUM SIZE FITS THE SCREEN -- so it can always be resized",
              w.minimumWidth() <= avail_w and w.minimumHeight() <= avail_h,
              f"min {w.minimumWidth()}x{w.minimumHeight()} vs {avail_w}x{avail_h}")

        # A window asking for more than the screen has must be clamped.
        probe = QtWidgets.QWidget()
        got_w, got_h = T.fit_to_screen(probe, 99999, 99999,
                                       min_width=99999, min_height=99999)
        check("an oversized request is clamped to the screen",
              got_w <= avail_w and got_h <= avail_h, f"{got_w}x{got_h}")
        check("an oversized MINIMUM is clamped too",
              probe.minimumWidth() <= avail_w
              and probe.minimumHeight() <= avail_h,
              f"{probe.minimumWidth()}x{probe.minimumHeight()}")
        got_w, got_h = T.fit_to_screen(probe, 500, 400)
        check("a request that fits is honoured", (got_w, got_h) == (500, 400),
              f"{got_w}x{got_h}")
        probe.deleteLater()

        # The cut dialog has eight channel rows and wants to be tall.
        tall = T.TrimDialog(rec, writer, w, extract_mode=True)
        check("the cut dialog fits the screen too",
              tall.width() <= avail_w and tall.height() <= avail_h,
              f"{tall.width()}x{tall.height()}")
        check("and its minimum fits as well",
              tall.minimumWidth() <= avail_w
              and tall.minimumHeight() <= avail_h)
        tall.close()

        check("the side panel cannot push the graph off the window",
              w.findChild(QtWidgets.QScrollArea) is None
              or w.findChild(QtWidgets.QScrollArea).minimumWidth() <= avail_w)

        # -- multi-select: N selected -> N separate recordings ----------------
        check("the take list allows several selections",
              w.take_list.selectionMode()
              == QtWidgets.QAbstractItemView.ExtendedSelection)

        # Three clips of the SAME length out of one take. The property under
        # test is that they stay three recordings and are never joined into
        # one long one.
        clip_s = 1.5
        before = len(T.load_dataset(root, stage="prefer_trimmed"))

        def exec_three(dialog):
            dialog.extract_gesture.setCurrentText("THUMB_UP")
            for start in (2.0, 6.0, 10.0):
                dialog._set_selection(start, start + clip_s)
                dialog.extract_clip()
            return 1

        T.TrimDialog.exec = exec_three
        try:
            produced = w.open_take_for_cutting(take_path)
        finally:
            T.TrimDialog.exec = real_exec

        check("three clips written", len(produced) == 3, str(len(produced)))
        thumbs = [r for r in T.load_dataset(root, stage="prefer_trimmed")
                  if r.label == "THUMB_UP"]
        check("THREE SEPARATE RECORDINGS, NOT ONE JOINED ONE",
              len(thumbs) == 3, str(len(thumbs)))
        check("the dataset grew by exactly three",
              len(T.load_dataset(root, stage="prefer_trimmed")) == before + 3)
        durations = sorted(r.duration for r in thumbs)
        check("each is one clip long, none is the sum of three",
              all(abs(d - clip_s) < 0.1 for d in durations),
              str([round(d, 2) for d in durations]))
        check("no recording is as long as 3 clips joined",
              max(durations) < clip_s * 2,
              f"{max(durations):.2f}s vs {clip_s * 3:.2f}s joined")
        check("they got separate repetition numbers",
              len({r.repetition for r in thumbs}) == 3,
              str(sorted(r.repetition for r in thumbs)))
        check("each is its own uid",
              len({r.uid for r in thumbs}) == 3)
        starts = sorted(r.meta.get("take_start_s") for r in thumbs)
        check("each records its own offset in the take",
              starts == [2.0, 6.0, 10.0], str(starts))
        check("each is contiguous in time (no join discontinuity)",
              all(float(np.max(np.diff(r.timestamps))) < 0.5 for r in thumbs))

        # -- several takes in a row -------------------------------------------
        all_takes = [t["path"] for t in w.writer.list_takes()]
        check("two takes on disk to work through", len(all_takes) == 2)
        recs_before = len(w.recordings)

        def exec_one(dialog):
            dialog.extract_gesture.setCurrentText("WAVE_RIGHT")
            dialog._set_selection(1.0, 1.0 + clip_s)
            dialog.extract_clip()
            return 1

        T.TrimDialog.exec = exec_one
        try:
            w.open_takes_for_cutting(all_takes)
        finally:
            T.TrimDialog.exec = real_exec

        waves = [r for r in T.load_dataset(root, stage="prefer_trimmed")
                 if r.label == "WAVE_RIGHT"]
        check("one clip per take, kept separate", len(waves) == 2,
              str(len(waves)))
        check("both reached the recordings table",
              len(w.recordings) == recs_before + 2)
        check("the status totals them up",
              "2 takes" in w.free_take_status.text(),
              w.free_take_status.text())
        check("clips from different takes name different parents",
              len({r.meta.get("parent_take_uid") for r in waves}) == 2)

        # -- OPEN FILE(S) splits takes from plain recordings ------------------
        plain = thumbs[0].primary_path(w.writer.root)
        mixed = [take_path, plain]
        seen = {"take": 0, "plain": 0}

        def exec_counting(dialog):
            if dialog.extract_mode:
                seen["take"] += 1
            else:
                seen["plain"] += 1
            return 0

        T.TrimDialog.exec = exec_counting
        try:
            w.open_paths_for_review(mixed)
        finally:
            T.TrimDialog.exec = real_exec
        check("a take in the selection got the cut dialog", seen["take"] == 1,
              str(seen))
        check("a plain recording got the trim dialog", seen["plain"] == 1,
              str(seen))
    finally:
        if window is not None:
            try:
                window.acq.stop()
            except Exception:
                pass
            window.close()
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    test_data_layer()
    test_gui()
    print("\n" + "=" * 16 + f"  {sum(OK)}/{len(OK)} checks passed  " + "=" * 16)
    sys.exit(0 if all(OK) else 1)
