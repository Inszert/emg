"""
MARKERS & NOTES: timestamped annotations on the live stream, cutting at a
marker, and annotating a recording that is already on disk.

No armband and no display needed (Qt runs offscreen).

The property that matters most here is the one checked twice, in both halves:
annotating never modifies the signal file. A note is written to a sidecar, so
`*_raw.*` stays byte-identical -- safety rule 1 in CLAUDE.md.
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


# ==============================================================================
# PART 1 -- data layer, no Qt
# ==============================================================================

def test_data_layer():
    root = tempfile.mkdtemp(prefix="emg_markers_")
    try:
        w = T.DatasetWriter(root, "p1", "s1")

        check("markers.jsonl absent before any write",
              not os.path.exists(w.markers_path))
        check("empty log loads as []", T.load_markers(w.markers_path) == [])

        t0 = time.time()
        m1 = T.Marker(t0, "electrode slipped", "note",
                      person_id="p1", session_id="s1")
        m2 = T.Marker(t0 + 5, "block 2 starts", "break",
                      person_id="p1", session_id="s1")
        w.append_marker(m1)
        w.append_marker(m2)
        check("log file created", os.path.exists(w.markers_path))

        got = w.load_markers()
        check("both markers read back", len(got) == 2, f"{len(got)}")
        check("oldest first", got[0].uid == m1.uid and got[1].uid == m2.uid)
        check("text survives", got[0].text == "electrode slipped")
        check("kind survives", got[1].kind == "break" and got[1].is_break)
        check("timestamp survives", abs(got[0].t_unix - t0) < 1e-3)
        check("uid preserved", got[0].uid == m1.uid)
        check("person/session recorded",
              got[0].person_id == "p1" and got[0].session_id == "s1")

        # Append-only: a new marker must not disturb the bytes already written.
        before = open(w.markers_path, encoding="utf-8").read()
        w.append_marker(T.Marker(t0 + 9, "third", "note"))
        after = open(w.markers_path, encoding="utf-8").read()
        check("append-only (old bytes untouched)", after.startswith(before))
        check("three markers now", len(w.load_markers()) == 3)

        w.append_marker(T.Marker(t0 + 10, "poznámka – číslo 5", "note"))
        check("unicode round-trips", "poznámka" in w.load_markers()[-1].text)

        with open(w.markers_path, "a", encoding="utf-8") as handle:
            handle.write("{not json at all\n\n")
        check("damaged line skipped, not fatal", len(w.load_markers()) == 4)

        span = T.markers_between(w.load_markers(), t0 + 1, t0 + 9.5)
        check("markers_between filters by time", len(span) == 2,
              f"{[m.text for m in span]}")

        check("short_text elides", len(T.Marker(t0, "x" * 80).short_text(10)) == 10)
        check("short_text of empty note", T.Marker(t0, "").short_text() == "note")
        check("short_text of empty break",
              T.Marker(t0, "", "break").short_text() == "BREAK")
        check("short_text collapses whitespace",
              T.Marker(t0, "a\n\n  b").short_text() == "a b")
        check("unknown kind falls back to note",
              T.Marker(t0, "x", "nonsense").kind == "note")

        # -- notes sidecar on a real recording -------------------------------
        rng = np.random.default_rng(0)
        emg = np.abs(rng.normal(0, 30, (100, 8))) + 20
        ts = np.linspace(0, 2, 100) + time.time()
        rec = T.Recording(emg=emg, timestamps=ts, label="FIST", person_id="p1",
                          session_id="s1", repetition=1)
        written = w.save(rec, T.STAGE_RAW)
        check("recording saved", bool(written.get("csv") or written.get("npz")))

        data_path = rec.primary_path(w.root)
        check("primary_path resolves to a real file",
              bool(data_path) and os.path.exists(data_path), str(data_path))
        check("no notes initially", T.load_recording_notes(data_path) == [])

        sidecar = T.notes_sidecar_path(data_path)
        check("sidecar name derives from the data file",
              sidecar.endswith("_raw" + T.NOTES_SIDECAR_SUFFIX),
              os.path.basename(sidecar))

        csv_before = open(data_path, "rb").read()
        path, _marker = w.add_note_to_recording(rec, "this one felt weak")
        check("sidecar written", os.path.exists(path))
        check("DATA FILE UNTOUCHED by annotating",
              open(data_path, "rb").read() == csv_before)

        notes = T.load_recording_notes(data_path)
        check("note reads back", len(notes) == 1 and "felt weak" in notes[0].text)

        w.add_note_to_recording(rec, "second thought: keep it")
        notes = T.load_recording_notes(data_path)
        check("two notes accumulate", len(notes) == 2, f"{len(notes)}")
        check("notes keep their order",
              "felt weak" in notes[0].text and "keep it" in notes[1].text)
        check("no .tmp file left behind", not os.path.exists(path + ".tmp"))

        payload = json.load(open(path, encoding="utf-8"))
        check("sidecar records the recording uid",
              payload.get("recording_uid") == rec.uid)
        check("sidecar names its recording",
              payload.get("recording_file") == os.path.basename(data_path))

        trimmed = rec.slice(10, 60, T.STAGE_TRIMMED)
        w.save(trimmed, T.STAGE_TRIMMED)
        tpath = trimmed.primary_path(w.root, T.STAGE_TRIMMED)
        check("trimmed copy is a different file", tpath != data_path)
        check("trimmed copy does not inherit the raw take's notes",
              T.load_recording_notes(tpath) == [])
    finally:
        shutil.rmtree(root, ignore_errors=True)


# ==============================================================================
# PART 2 -- the GUI, offscreen
# ==============================================================================

def test_gui():
    from PySide6 import QtWidgets

    root = tempfile.mkdtemp(prefix="emg_gui_markers_")
    T.DATA_FOLDER = root
    T.PROJECT_DIR = root
    T.PERSON_ID = "tester"
    T.SESSION_ID = "sess_mk"
    T.MYO_PORT = "COM_DOES_NOT_EXIST"      # keep the real armband out of this

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    T.apply_theme(app)
    window = None
    try:
        window = T.MainWindow()
        w = window

        check("marker text field built", hasattr(w, "marker_text"))
        check("ADD NOTE button built", hasattr(w, "add_note_button"))
        check("BREAK button built", hasattr(w, "add_break_button"))
        check("cut checkbox built", hasattr(w, "marker_cut_check"))
        check("cut button built", hasattr(w, "cut_marker_button"))
        check("marker list built", hasattr(w, "marker_list"))
        check("OPEN FILE button built", hasattr(w, "open_file_button"))
        check("session_markers starts empty", w.session_markers == [])
        check("cut button disabled with no markers",
              not w.cut_marker_button.isEnabled())

        w.marker_text.setText("   ")
        w.add_note_marker()
        check("blank note refused", len(w.session_markers) == 0)
        check("status explains why", "type the note" in w.marker_status.text())

        rng = np.random.default_rng(1)
        now = time.time()
        for i in range(300):
            sample = (np.abs(rng.normal(0, 25, 8)) + 20).astype(int)
            w.acq.ring.push(now - 6.0 + i * 0.02, sample)
        _, latest = w.acq.ring.latest()
        check("ring has samples", bool(latest))

        w.marker_text.setText("electrode slipped")
        w.add_note_marker()
        check("note logged", len(w.session_markers) == 1)
        check("note text stored", w.session_markers[0].text == "electrode slipped")
        check("note kind is note", w.session_markers[0].kind == "note")
        check("input cleared after adding", w.marker_text.text() == "")
        check("marker handed to the plot", len(w.plot.markers) == 1)
        check("list shows it", w.marker_list.count() == 1)
        check("cut button enabled once a marker exists",
              w.cut_marker_button.isEnabled())
        check("marker time tracks the signal, not the GUI clock",
              abs(w.session_markers[0].t_unix - latest) < 1e-6,
              f"{w.session_markers[0].t_unix - latest:+.4f}")
        check("markers.jsonl written", os.path.exists(w.writer.markers_path))

        w.add_break_marker()
        check("a break needs no text", len(w.session_markers) == 2)
        check("break kind recorded", w.session_markers[1].kind == "break")
        check("break caption falls back to BREAK",
              w.session_markers[1].short_text() == "BREAK")

        # -- cut since the last marker ---------------------------------------
        t_mark = w.session_markers[-1].t_unix
        for i in range(120):
            sample = (np.abs(rng.normal(0, 200, 8)) + 300).astype(int)
            w.acq.ring.push(t_mark + 0.02 * (i + 1), sample)

        before = len(w.recordings)
        w.label_gesture_combo.setCurrentText("FIST")
        w.cut_since_last_marker()
        check("cut produced a recording", len(w.recordings) == before + 1)
        rec = w.recordings[-1]
        check("cut takes its label from the combo", rec.label == "FIST")
        check("cut records the marker it came from",
              rec.meta.get("cut_from_marker_uid") == w.session_markers[-1].uid)
        check("capture_style marks it as a marker cut",
              rec.meta.get("capture_style") == "cut_at_marker")
        data_path = rec.primary_path(w.writer.root)
        check("cut file is on disk", bool(data_path) and os.path.exists(data_path))
        check("cut spans marker -> now", abs(rec.duration - 2.4) < 0.25,
              f"{rec.duration:.2f}s")
        check("markers inside the span travel in the metadata",
              isinstance(rec.meta.get("markers"), list)
              and len(rec.meta["markers"]) >= 1)

        # -- the switch -------------------------------------------------------
        w.marker_cut_check.setChecked(False)
        check("checkbox mirrors into the config",
              T.MARKER_CUT_ENABLED is False)
        check("cut button disabled when switched off",
              not w.cut_marker_button.isEnabled())
        before = len(w.recordings)
        w.cut_since_last_marker()
        check("cut refused when switched off", len(w.recordings) == before)
        check("status says it is off", "switched off" in w.marker_status.text())
        w.marker_cut_check.setChecked(True)

        # -- reload / subject switching ---------------------------------------
        w.session_markers = []
        w.plot.clear_markers()
        w._load_session_markers()
        check("markers reload from disk", len(w.session_markers) == 2)
        check("reloaded markers are redrawn", len(w.plot.markers) == 2)
        check("reloaded text intact",
              w.session_markers[0].text == "electrode slipped")

        w.person_edit.setText("other_person")
        w._on_identity_changed()
        check("a different subject starts with no markers",
              w.session_markers == [])
        w.person_edit.setText("tester")
        w._on_identity_changed()
        check("switching back restores them", len(w.session_markers) == 2)

        # -- annotating an existing take in the dialog ------------------------
        dialog = T.TrimDialog(rec, w.writer, w)
        check("dialog built the notes panel", hasattr(dialog, "notes_list"))
        check("dialog found the marker inside the take",
              len(dialog.inside_markers) >= 1)
        check("dialog drew separator lines", len(dialog._note_lines) > 0)

        csv_before = open(data_path, "rb").read()
        dialog.note_input.setText("")
        dialog.add_note()
        check("blank dialog note refused", dialog.notes_added == 0)

        dialog.note_input.setText("subject said this felt weak")
        dialog.add_note()
        check("dialog note saved", dialog.notes_added == 1)
        check("sidecar exists", os.path.exists(T.notes_sidecar_path(data_path)))
        check("DATA FILE UNTOUCHED by a dialog note",
              open(data_path, "rb").read() == csv_before)
        notes = T.load_recording_notes(data_path)
        check("dialog note readable from disk",
              len(notes) == 1 and "felt weak" in notes[0].text)
        dialog.note_input.setText("second note")
        dialog.add_note()
        check("dialog notes accumulate",
              len(T.load_recording_notes(data_path)) == 2)
        check("dialog list shows notes and markers together",
              dialog.notes_list.count() >= 3, str(dialog.notes_list.count()))
        dialog.close()

        derived = T.MainWindow._dataset_root_for(data_path)
        check("dataset root derived from a recording path",
              os.path.normcase(derived) == os.path.normcase(w.writer.root))

        # -- drawing ----------------------------------------------------------
        w.plot.update_data(*w.acq.ring.snapshot())
        w.plot._draw_markers(time.time())
        visible = sum(1 for cap in w.plot.marker_labels if cap.isVisible())
        check("marker captions drawn", visible >= 1, str(visible))
        w.plot.clear_markers()
        check("clear_markers hides every caption",
              not any(cap.isVisible() for cap in w.plot.marker_labels))
        check("clear_markers empties the list", w.plot.markers == [])
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
