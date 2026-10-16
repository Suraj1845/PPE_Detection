"""Person-level PPE assessment by spatial association of detections.

A PPE item (``Hardhat``, ``Safety Vest``...) or a violation (``NO-Hardhat``...)
is attributed to a person only if most of the item's box lies inside the
region of that person's box where the item is worn:

* hardhat and mask -> the head region, the top 40 % of the person box
* vest             -> the torso region, 15 %-80 % of the person box height

Each item goes to the single person whose region contains the largest share
of it, provided that share is at least ``min_overlap``. A person is reported:

* ``violation``  if a required item is detected as missing (``NO-...``) on them,
* ``compliant``  if every required item is detected as present on them,
* ``unknown``    otherwise (e.g. the item is not visible or nothing was detected).

Nothing is inferred for people whose items are not detected, and items that
cannot be attributed to anybody stay object-level detections.

Class names are matched exactly as the model's metadata lists them; models
without person and PPE classes (such as the COCO model) get ``ppe_supported``
= False and no person-level claims at all.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from .detector import Detection

Box = tuple[int, int, int, int]

#: Exact class names of the 10-class Construction Site Safety scheme the original
#: project used, mapped to (category, worn?). "Person" is handled separately.
PPE_CLASS_ROLES: Mapping[str, tuple[str, bool]] = {
    "Hardhat": ("hardhat", True),
    "NO-Hardhat": ("hardhat", False),
    "Mask": ("mask", True),
    "NO-Mask": ("mask", False),
    "Safety Vest": ("vest", True),
    "NO-Safety Vest": ("vest", False),
}
PERSON_CLASSES = frozenset({"Person", "person"})

#: Vertical span of the person box where each category is worn (fractions of its height).
REGIONS: Mapping[str, tuple[float, float]] = {
    "hardhat": (0.0, 0.40),
    "mask": (0.0, 0.40),
    "vest": (0.15, 0.80),
}


class Status(StrEnum):
    COMPLIANT = "compliant"
    VIOLATION = "violation"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class PpeSchema:
    """Which of a model's classes are persons, PPE items and violations."""

    person_labels: frozenset[str]
    item_roles: Mapping[str, tuple[str, bool]]

    @classmethod
    def from_labels(cls, labels: Iterable[str]) -> PpeSchema:
        labels = list(labels)
        return cls(
            person_labels=frozenset(x for x in labels if x in PERSON_CLASSES),
            item_roles={x: PPE_CLASS_ROLES[x] for x in labels if x in PPE_CLASS_ROLES},
        )

    @property
    def ppe_supported(self) -> bool:
        return bool(self.person_labels) and bool(self.item_roles)

    @property
    def categories(self) -> frozenset[str]:
        return frozenset(category for category, _ in self.item_roles.values())

    def is_violation(self, label: str) -> bool:
        role = self.item_roles.get(label)
        return role is not None and not role[1]

    def is_ppe(self, label: str) -> bool:
        role = self.item_roles.get(label)
        return role is not None and role[1]


@dataclass
class PersonAssessment:
    person: Detection
    items: dict[str, Detection] = field(default_factory=dict)  # category -> strongest attributed detection
    status: Status = Status.UNKNOWN
    missing: tuple[str, ...] = ()  # required categories detected as NOT worn
    unseen: tuple[str, ...] = ()  # required categories with no attributed detection


@dataclass
class FrameAssessment:
    ppe_supported: bool
    persons: list[PersonAssessment] = field(default_factory=list)
    unattributed: list[Detection] = field(default_factory=list)  # PPE/violation boxes on nobody

    def count(self, status: Status) -> int:
        return sum(1 for p in self.persons if p.status is status)


def undetectable(schema: PpeSchema, required: Iterable[str]) -> tuple[str, ...]:
    """Required categories this model has no classes for (e.g. ``mask`` on a hardhat-only model)."""
    return tuple(c for c in required if c not in schema.categories)


def region_of(person: Box, category: str) -> Box:
    """The part of a person box where ``category`` is worn."""
    x1, y1, x2, y2 = person
    top, bottom = REGIONS[category]
    height = y2 - y1
    return x1, int(round(y1 + top * height)), x2, int(round(y1 + bottom * height))


def overlap_fraction(item: Box, region: Box) -> float:
    """Share of the item box's area that lies inside ``region`` (0-1)."""
    ix1, iy1, ix2, iy2 = item
    rx1, ry1, rx2, ry2 = region
    item_area = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if item_area == 0:
        return 0.0
    inter = max(0, min(ix2, rx2) - max(ix1, rx1)) * max(0, min(iy2, ry2) - max(iy1, ry1))
    return inter / item_area


def assess(
    detections: Sequence[Detection],
    schema: PpeSchema,
    required: Iterable[str] = ("hardhat", "vest"),
    min_overlap: float = 0.5,
) -> FrameAssessment:
    """Attribute PPE detections to persons and classify each person."""
    if not schema.ppe_supported:
        return FrameAssessment(ppe_supported=False)

    # Required items the model cannot detect stay required: they can never be "seen", so
    # nobody is reported compliant on the strength of an item that was never checked.
    required = tuple(required)
    persons = [PersonAssessment(d) for d in detections if d.label in schema.person_labels]
    result = FrameAssessment(ppe_supported=True, persons=persons)

    for item in detections:
        role = schema.item_roles.get(item.label)
        if role is None:
            continue
        category = role[0]
        # Largest share wins; on a tie the smaller (tighter-fitting) person box wins.
        scored = [
            (overlap_fraction(item.box, region_of(p.person.box, category)), -p.person.area, i)
            for i, p in enumerate(persons)
        ]
        best_share, _, best_index = max(scored, default=(0.0, 0, -1))
        best = persons[best_index] if best_index >= 0 else None
        if best is None or best_share < min_overlap:
            result.unattributed.append(item)
            continue
        current = best.items.get(category)
        if current is None or item.score > current.score:
            best.items[category] = item

    for person in persons:
        worn = {c: schema.item_roles[d.label][1] for c, d in person.items.items()}
        person.missing = tuple(c for c in required if worn.get(c) is False)
        person.unseen = tuple(c for c in required if c not in worn)
        if person.missing:
            person.status = Status.VIOLATION
        elif required and not person.unseen:
            person.status = Status.COMPLIANT
        else:
            person.status = Status.UNKNOWN
    return result
