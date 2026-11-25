"""
Markers and notes sidecars.
"""

import json
import os
import time
import uuid

from emg_app import config as cfg
from emg_app.utils import ensure_dir, iso_time


# ------------------------------------------------------------------------------
# Markers / notes
# ------------------------------------------------------------------------------
# Timestamped free text, kept apart from the signal it describes. Written
# append-only so an annotation can never damage a recording, and so two notes
# about the same moment both survive.

MARKER_KINDS = ("note", "break")


class Marker:
    """One timestamped annotation."""

    def __init__(self, t_unix: float, text: str, kind: str = "note",
                 uid: str = None, created_unix: float = None,
                 person_id: str = "", session_id: str = ""):
        self.t_unix = float(t_unix)
        self.text = str(text)
        self.kind = kind if kind in MARKER_KINDS else "note"
        self.uid = uid or uuid.uuid4().hex[:8]
        self.created_unix = float(
            created_unix if created_unix is not None else time.time())
        self.person_id = str(person_id)
        self.session_id = str(session_id)

    # -- display --------------------------------------------------------------

    @property
    def is_break(self) -> bool:
        return self.kind == "break"

    def short_text(self, limit: int = None) -> str:
        """The text, elided for drawing on the graph."""
        limit = cfg.MARKER_LABEL_MAX_CHARS if limit is None else limit
        text = " ".join(self.text.split())
        if not text:
            return "BREAK" if self.is_break else "note"
        return text if len(text) <= limit else text[: max(1, limit - 1)] + "\u2026"

    # -- serialisation --------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "uid": self.uid,
            "kind": self.kind,
            "text": self.text,
            "t_unix": round(self.t_unix, 6),
            "t_iso": iso_time(self.t_unix),
            "created_unix": round(self.created_unix, 6),
            "created_iso": iso_time(self.created_unix),
            "person_id": self.person_id,
            "session_id": self.session_id,
        }

    @staticmethod
    def from_dict(entry: dict):
        return Marker(
            t_unix=entry.get("t_unix", 0.0),
            text=entry.get("text", ""),
            kind=entry.get("kind", "note"),
            uid=entry.get("uid"),
            created_unix=entry.get("created_unix"),
            person_id=entry.get("person_id", ""),
            session_id=entry.get("session_id", ""),
        )

    def __repr__(self) -> str:                               # pragma: no cover
        return (f"Marker({self.kind}, {iso_time(self.t_unix)}, "
                f"{self.short_text(18)!r})")


def append_marker_jsonl(path: str, marker: Marker) -> str:
    """Append one marker as a JSON line. The file is never rewritten."""
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(marker.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_markers(path: str):
    """Read a marker log, oldest first. A damaged line is skipped, not fatal."""
    markers = []
    if not path or not os.path.exists(path):
        return markers
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                markers.append(Marker.from_dict(json.loads(line)))
            except (json.JSONDecodeError, TypeError, AttributeError):
                continue                                     # pragma: no cover
    markers.sort(key=lambda m: m.t_unix)
    return markers


def markers_between(markers, start_unix: float, end_unix: float):
    """The markers falling inside a time span, oldest first."""
    return [m for m in markers if start_unix <= m.t_unix <= end_unix]


def notes_sidecar_path(data_path: str) -> str:
    """
    Where the notes for a recording live:

        FIST_rep001_..._raw.csv  ->  FIST_rep001_..._raw_notes.json

    The stage suffix is kept, so notes follow the file they describe rather
    than being shared between a raw take and its trimmed copy.
    """
    stem, _ = os.path.splitext(data_path)
    return stem + cfg.NOTES_SIDECAR_SUFFIX


def load_recording_notes(data_path: str):
    """Notes attached to one recording, oldest first. Missing file = []."""
    path = notes_sidecar_path(data_path)
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except (json.JSONDecodeError, OSError):                  # pragma: no cover
        return []
    entries = payload.get("notes", []) if isinstance(payload, dict) else payload
    markers = [Marker.from_dict(e) for e in entries if isinstance(e, dict)]
    markers.sort(key=lambda m: m.created_unix)
    return markers


def append_recording_note(data_path: str, marker: Marker,
                          rec_uid: str = "") -> str:
    """
    Add a note to a recording's sidecar.

    Read-modify-write of the *sidecar* only. The recording's own data file is
    never opened. Writing the whole sidecar each time is fine: it holds a
    handful of short strings, and it keeps the file valid JSON that any tool
    can read, unlike an append-only JSONL sidecar per recording. The write
    goes through a temp file and os.replace so an interrupted save cannot
    leave a half-written sidecar behind.
    """
    path = notes_sidecar_path(data_path)
    existing = load_recording_notes(data_path)
    existing.append(marker)
    payload = {
        "recording_file": os.path.basename(data_path),
        "recording_uid": rec_uid,
        "updated_iso": iso_time(time.time()),
        "notes": [m.to_dict() for m in existing],
    }
    ensure_dir(os.path.dirname(path))
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
    os.replace(tmp, path)
    return path
