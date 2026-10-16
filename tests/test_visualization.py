"""Drawing uses each detection's own colour; nothing leaks between boxes."""

from __future__ import annotations

from conftest import PPE_LABELS, blank

from ppe_detection.compliance import PpeSchema, assess
from ppe_detection.detector import Detection
from ppe_detection.visualization import (
    BLUE,
    GREEN,
    RED,
    STATUS_COLORS,
    class_color,
    draw_frame,
    draw_status_bar,
    status_bar_height,
)

SCHEMA = PpeSchema.from_labels(PPE_LABELS)


def test_colour_depends_only_on_the_class():
    assert class_color("NO-Hardhat", SCHEMA) == RED
    assert class_color("Safety Vest", SCHEMA) == GREEN
    assert class_color("Person", SCHEMA) == BLUE
    assert class_color("vehicle", SCHEMA) == class_color("vehicle", SCHEMA)  # deterministic
    # Order of drawing must not change any colour (the original bug).
    assert [class_color(x, SCHEMA) for x in ("NO-Mask", "Mask", "NO-Mask")] == [RED, GREEN, RED]


def test_draw_frame_marks_pixels_with_status_colours():
    frame = blank(400, 300)
    person = Detection(5, "Person", 0.9, (20, 60, 200, 290))
    violation = Detection(2, "NO-Hardhat", 0.8, (60, 62, 140, 120))
    assessment = assess([person, violation], SCHEMA, required=("hardhat",))
    draw_frame(frame, [person, violation], SCHEMA, assessment)
    assert tuple(frame[200, 20]) == STATUS_COLORS[assessment.persons[0].status]  # left edge of person box
    assert tuple(frame[100, 60]) == RED  # left edge of violation box


def test_labels_near_the_top_move_below_the_status_bar():
    frame = blank(640, 360)
    top_person = Detection(5, "Person", 0.9, (100, 5, 300, 300))
    bar = status_bar_height(frame, 3)
    draw_frame(frame, [top_person], SCHEMA, assess([], SCHEMA), min_top=bar)
    label_rows = [y for y in range(frame.shape[0]) if (frame[y, 110:200] == BLUE).all(axis=-1).any()]
    # Only the box's top edge (rows ~4-6) is above the bar's bottom; the filled label sits just below the bar.
    assert not any(8 < y < bar for y in label_rows)
    assert any(bar <= y < bar + 40 for y in label_rows)


def test_status_bar_handles_warning_line():
    frame = blank(640, 360)
    draw_status_bar(frame, ["FPS 10.0"], warning="PPE model not installed")
    assert frame[:20].any() and not frame[200:].any()


def test_long_status_lines_are_shrunk_to_fit_narrow_frames():
    frame = blank(320, 240)
    lines = ["FPS 10.0 | inference 100.0 ms | CPU"]
    warning = "UNVERIFIED MODEL | PPE model not installed - showing person only, no PPE compliance assessed"
    draw_status_bar(frame, lines, warning)
    bar = status_bar_height(frame, 2)
    assert not frame[:bar, -6:].any()  # nothing drawn in the last 6 px: the text fits


def test_labels_of_neighbouring_boxes_do_not_overlap():
    """Two people standing close: their labels would land on the same spot (seen on a real webcam)."""
    import ppe_detection.visualization as vis

    frame = blank(640, 360)
    placed = []
    original = vis.cv2.rectangle

    def record(img, p1, p2, color, thickness, *args):
        if thickness == vis.cv2.FILLED:
            placed.append((p1[0], p1[1], p2[0], p2[1]))
        return original(img, p1, p2, color, thickness, *args)

    vis.cv2.rectangle = record
    try:
        a = Detection(5, "Person", 0.9, (300, 150, 380, 300))
        b = Detection(5, "Person", 0.8, (320, 152, 400, 300))
        draw_frame(frame, [a, b], PpeSchema.from_labels(["person"]), assess([], SCHEMA))
    finally:
        vis.cv2.rectangle = original
    assert len(placed) == 2
    first, second = placed
    assert not (first[0] < second[2] and second[0] < first[2] and first[1] < second[3] and second[1] < first[3])
