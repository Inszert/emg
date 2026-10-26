# -*- coding: utf-8 -*-
"""Generate the application icon with QPainter (no other dependencies).

The mark: a dark rounded tile, a ring (the armband) in the accent colour, and
an EMG envelope trace crossing it. Only generic shapes -- no third-party logo.

Writes assets/icons/icon_<size>.png for 16, 32, 48, 64, 128, 256 and a
multi-size assets/icons/app.ico (PNG-compressed entries, which Windows Vista+
and Qt both read). Strokes are specified as a fraction of the size, with a
floor in pixels, so the 16 px version stays legible instead of being a blur.

Usage:  python tools/make_icons.py [output_dir]
"""
import os
import struct
import sys

from PySide6 import QtCore, QtGui

SIZES = (16, 32, 48, 64, 128, 256)
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.dirname(HERE), "assets", "icons")

TILE_TOP = "#1b2230"
TILE_BOT = "#0b0e13"
RING = "#38bdf8"          # COL_ACCENT
RING_DIM = "#0b5f86"
TRACE = "#f8fafc"


def render(size: int) -> QtGui.QImage:
    s = float(size)
    img = QtGui.QImage(size, size, QtGui.QImage.Format_ARGB32)
    img.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)

    # Tile
    inset = max(0.5, s * 0.03)
    tile = QtCore.QRectF(inset, inset, s - 2 * inset, s - 2 * inset)
    grad = QtGui.QLinearGradient(0, 0, 0, s)
    grad.setColorAt(0, QtGui.QColor(TILE_TOP))
    grad.setColorAt(1, QtGui.QColor(TILE_BOT))
    p.setPen(QtCore.Qt.NoPen)
    p.setBrush(grad)
    p.drawRoundedRect(tile, s * 0.22, s * 0.22)

    # Band: a ring, thick enough to survive 16 px
    ring_w = max(2.0, s * 0.12)
    r = s * 0.33
    c = QtCore.QPointF(s / 2, s / 2)
    p.setBrush(QtCore.Qt.NoBrush)
    pen = QtGui.QPen(QtGui.QColor(RING), ring_w)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    p.setPen(pen)
    p.drawEllipse(c, r, r)
    # a darker 'pod' segment at the top gives the ring an armband look at size
    if size >= 32:
        pen2 = QtGui.QPen(QtGui.QColor(RING_DIM), ring_w)
        pen2.setCapStyle(QtCore.Qt.FlatCap)
        p.setPen(pen2)
        rect = QtCore.QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)
        p.drawArc(rect, int(200 * 16), int(140 * 16))

    # EMG trace across the middle
    tw = max(1.6, s * 0.075)
    pen3 = QtGui.QPen(QtGui.QColor(TRACE), tw)
    pen3.setCapStyle(QtCore.Qt.RoundCap)
    pen3.setJoinStyle(QtCore.Qt.RoundJoin)
    p.setPen(pen3)
    pts = [(0.14, 0.50), (0.34, 0.50), (0.41, 0.30), (0.49, 0.70),
           (0.57, 0.22), (0.65, 0.58), (0.70, 0.50), (0.86, 0.50)]
    path = QtGui.QPainterPath()
    for i, (x, y) in enumerate(pts):
        pt = QtCore.QPointF(x * s, y * s)
        path.moveTo(pt) if i == 0 else path.lineTo(pt)
    p.drawPath(path)
    p.end()
    return img


def write_ico(path: str, images) -> None:
    """ICO container with PNG payloads. 256 is stored as width/height 0."""
    blobs = []
    for img in images:
        ba = QtCore.QByteArray()
        buf = QtCore.QBuffer(ba)
        buf.open(QtCore.QIODevice.WriteOnly)
        img.save(buf, "PNG")
        blobs.append((img.width(), bytes(ba)))
    head = struct.pack("<HHH", 0, 1, len(blobs))
    offset = 6 + 16 * len(blobs)
    entries, data = b"", b""
    for w, blob in blobs:
        b = 0 if w >= 256 else w
        entries += struct.pack("<BBBBHHII", b, b, 0, 0, 1, 32, len(blob), offset)
        data += blob
        offset += len(blob)
    with open(path, "wb") as fh:
        fh.write(head + entries + data)


def render_glyph(kind: str, color: str, px: int = 24) -> QtGui.QImage:
    """Small control glyphs the stylesheet points at (tick, chevrons)."""
    img = QtGui.QImage(px, px, QtGui.QImage.Format_ARGB32)
    img.fill(QtCore.Qt.transparent)
    p = QtGui.QPainter(img)
    p.setRenderHint(QtGui.QPainter.Antialiasing, True)
    pen = QtGui.QPen(QtGui.QColor(color), px * 0.14)
    pen.setCapStyle(QtCore.Qt.RoundCap)
    pen.setJoinStyle(QtCore.Qt.RoundJoin)
    p.setPen(pen)
    pts = {"check": [(0.22, 0.52), (0.42, 0.72), (0.78, 0.30)],
           "down": [(0.22, 0.38), (0.50, 0.66), (0.78, 0.38)],
           "up": [(0.22, 0.62), (0.50, 0.34), (0.78, 0.62)]}[kind]
    path = QtGui.QPainterPath()
    for i, (x, y) in enumerate(pts):
        pt = QtCore.QPointF(x * px, y * px)
        path.moveTo(pt) if i == 0 else path.lineTo(pt)
    p.drawPath(path)
    p.end()
    return img


GLYPHS = (("check", "#06121a"), ("down", "#9aa3b2"), ("up", "#9aa3b2"))


def main(out_dir: str = DEFAULT_OUT) -> None:
    os.makedirs(out_dir, exist_ok=True)
    for kind, color in GLYPHS:
        render_glyph(kind, color).save(os.path.join(out_dir, f"ui_{kind}.png"), "PNG")
    images = []
    for size in SIZES:
        img = render(size)
        if not img.save(os.path.join(out_dir, f"icon_{size}.png"), "PNG"):
            raise SystemExit(f"could not write icon_{size}.png")
        images.append(img)
    write_ico(os.path.join(out_dir, "app.ico"), images)
    print(f"wrote {len(SIZES)} PNGs and app.ico to {out_dir}".encode("ascii", "replace").decode())


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
