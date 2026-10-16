"""Drawing detections, person status and the status bar onto frames with OpenCV.

Each box's colour is computed from that box's own class (or its person's
status); there is no shared colour variable carried between detections.
"""

from __future__ import annotations

import zlib
from collections.abc import Sequence

import cv2
import numpy as np

from .compliance import FrameAssessment, PersonAssessment, PpeSchema, Status
from .detector import Detection

Color = tuple[int, int, int]  # BGR

RED: Color = (40, 40, 230)
GREEN: Color = (60, 180, 60)
AMBER: Color = (0, 170, 255)
BLUE: Color = (230, 150, 40)
WHITE: Color = (255, 255, 255)
BLACK: Color = (0, 0, 0)
_OTHER_PALETTE: tuple[Color, ...] = (
    (180, 130, 70), (160, 90, 160), (90, 160, 160), (130, 130, 200), (200, 170, 120), (120, 120, 120),
)
STATUS_COLORS: dict[Status, Color] = {Status.VIOLATION: RED, Status.COMPLIANT: GREEN, Status.UNKNOWN: AMBER}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def class_color(label: str, schema: PpeSchema) -> Color:
    """Colour for one detection, from its class alone."""
    if schema.is_violation(label):
        return RED
    if schema.is_ppe(label):
        return GREEN
    if label in schema.person_labels:
        return BLUE
    return _OTHER_PALETTE[zlib.crc32(label.encode("utf-8")) % len(_OTHER_PALETTE)]


def _scale(frame: np.ndarray) -> tuple[float, int]:
    """Font scale and line thickness that stay readable from 640 px to 4K."""
    scale = max(0.45, min(1.2, frame.shape[1] / 1600))
    return scale, max(1, int(round(scale * 2)))


Rect = tuple[int, int, int, int]  # x1, y1, x2, y2


def _overlaps(a: Rect, b: Rect) -> bool:
    return a[0] < b[2] and b[0] < a[2] and a[1] < b[3] and b[1] < a[3]


def put_label(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: Color,
    min_top: int = 0,
    taken: list[Rect] | None = None,
) -> None:
    """Text on a filled box whose bottom-left corner is ``origin``, kept inside the frame.

    If the label would reach above ``min_top`` (e.g. under the status bar), it is
    drawn just below ``origin`` instead, i.e. inside the detection box. Labels
    already drawn this frame are listed in ``taken``; a label that would cover one
    of them steps down one label height at a time until it is clear.
    """
    scale, thickness = _scale(frame)
    (tw, th), baseline = cv2.getTextSize(text, FONT, scale, thickness)
    label_height = th + baseline + 4
    x = min(max(0, origin[0]), max(0, frame.shape[1] - tw - 6))
    y = origin[1]
    if y - label_height < min_top:
        y = max(y, min_top) + label_height
    if taken is not None:
        # Try the preferred spot, then step down, then step up (near the bottom edge), staying
        # inside the frame and below the status bar. A very crowded spot keeps the preferred place.
        downs = [y + k * label_height for k in range(8)]
        ups = [y - k * label_height for k in range(1, 8)]
        for candidate in downs + ups:
            if candidate > frame.shape[0] or candidate - label_height < min_top:
                continue
            rect = (x, candidate - label_height, x + tw + 6, candidate)
            if not any(_overlaps(rect, other) for other in taken):
                y = candidate
                break
        taken.append((x, y - label_height, x + tw + 6, y))
    cv2.rectangle(frame, (x, y - th - baseline - 4), (x + tw + 6, y), color, cv2.FILLED)
    cv2.putText(frame, text, (x + 3, y - baseline - 2), FONT, scale, WHITE, thickness, cv2.LINE_AA)


def person_text(person: PersonAssessment) -> str:
    text = f"{person.person.label} {person.person.score:.2f}"
    if person.status is Status.VIOLATION:
        return f"{text} | NO {', '.join(person.missing)}"
    if person.status is Status.COMPLIANT:
        return f"{text} | PPE OK"
    return f"{text} | not seen: {', '.join(person.unseen)}" if person.unseen else text


def draw_frame(
    frame: np.ndarray,
    detections: Sequence[Detection],
    schema: PpeSchema,
    assessment: FrameAssessment,
    min_top: int = 0,
) -> None:
    """Draw every detection; persons get their assessed status when PPE is supported.

    ``min_top`` is the height of the status bar, so labels are never hidden under it.
    """
    assessed = {id(p.person): p for p in assessment.persons}
    _, thickness = _scale(frame)
    # (detection, colour, label text, label anchor). Persons first so PPE item labels, placed
    # later, step around the person labels; person status sits at the bottom of the box so it
    # never covers head-worn PPE labels.
    items = []
    for detection in sorted(detections, key=lambda d: d.label not in schema.person_labels):
        x1, y1, _, y2 = detection.box
        person = assessed.get(id(detection))
        if person is not None:
            items.append((detection, STATUS_COLORS[person.status], person_text(person), (x1, y2)))
        else:
            text = f"{detection.label} {detection.score:.2f}"
            items.append((detection, class_color(detection.label, schema), text, (x1, y1)))
    # All boxes first, then all labels, so no box line is drawn across a label.
    for detection, color, _, _ in items:
        x1, y1, x2, y2 = detection.box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness + 1)
    taken: list[Rect] = []  # label areas already used this frame, so labels never cover each other
    for _, color, text, anchor in items:
        put_label(frame, text, anchor, color, min_top, taken)


def status_bar_height(frame: np.ndarray, rows: int) -> int:
    scale, _ = _scale(frame)
    return (int(32 * scale) + 6) * rows + 8


def draw_status_bar(frame: np.ndarray, lines: Sequence[str], warning: str | None = None) -> None:
    """Semi-transparent bar at the top with run statistics, plus an optional warning line."""
    scale, thickness = _scale(frame)
    line_height = int(32 * scale) + 6
    rows = list(lines) + ([warning] if warning else [])
    bar_height = status_bar_height(frame, len(rows))
    overlay = frame[:bar_height].copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_height), BLACK, cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, frame[:bar_height], 0.45, 0, dst=frame[:bar_height])
    for row, text in enumerate(rows):
        color = AMBER if warning is not None and row == len(rows) - 1 else WHITE
        y = 4 + line_height * (row + 1) - 6
        (text_width, _), _ = cv2.getTextSize(text, FONT, scale, thickness)
        row_scale = scale * min(1.0, (frame.shape[1] - 16) / max(text_width, 1))  # shrink long lines to fit
        cv2.putText(frame, text, (8, y), FONT, row_scale, color, thickness, cv2.LINE_AA)
