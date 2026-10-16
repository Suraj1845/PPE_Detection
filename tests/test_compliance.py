"""Person-to-PPE association tests on hand-made boxes."""

from __future__ import annotations

from conftest import PPE_LABELS

from ppe_detection.compliance import PpeSchema, Status, assess, overlap_fraction, region_of
from ppe_detection.detector import Detection

SCHEMA = PpeSchema.from_labels(PPE_LABELS)


def det(label: str, box: tuple[int, int, int, int], score: float = 0.9) -> Detection:
    return Detection(PPE_LABELS.index(label), label, score, box)


PERSON_A = det("Person", (100, 100, 200, 400))  # head region y 100-220, torso y 145-340
PERSON_B = det("Person", (400, 100, 500, 400))


def test_schema_roles_from_exact_class_names():
    assert SCHEMA.ppe_supported
    assert SCHEMA.categories == {"hardhat", "mask", "vest"}
    assert SCHEMA.is_violation("NO-Hardhat") and SCHEMA.is_ppe("Safety Vest")
    assert not SCHEMA.is_violation("Safety Cone") and not SCHEMA.is_ppe("machinery")


def test_coco_model_makes_no_person_level_claims():
    coco = PpeSchema.from_labels(["person", "bicycle", "car"])
    result = assess([Detection(0, "person", 0.9, (0, 0, 10, 10))], coco)
    assert not coco.ppe_supported and not result.ppe_supported and result.persons == []


def test_geometry_helpers():
    assert region_of((100, 100, 200, 400), "hardhat") == (100, 100, 200, 220)
    assert region_of((100, 100, 200, 400), "vest") == (100, 145, 200, 340)
    assert overlap_fraction((0, 0, 10, 10), (5, 0, 20, 10)) == 0.5
    assert overlap_fraction((0, 0, 0, 10), (0, 0, 10, 10)) == 0.0


def test_compliant_only_when_every_required_item_is_seen_on_the_person():
    hardhat = det("Hardhat", (120, 95, 180, 140))  # mostly in A's head region
    vest = det("Safety Vest", (110, 180, 190, 320))
    result = assess([PERSON_A, hardhat, vest], SCHEMA)
    (a,) = result.persons
    assert a.status is Status.COMPLIANT and a.missing == () and a.unseen == ()


def test_violation_is_attributed_to_the_right_person():
    no_hardhat = det("NO-Hardhat", (420, 100, 480, 160))  # on B's head
    vest_a = det("Safety Vest", (110, 180, 190, 320))
    result = assess([PERSON_A, PERSON_B, no_hardhat, vest_a], SCHEMA)
    a, b = result.persons
    assert b.status is Status.VIOLATION and b.missing == ("hardhat",)
    assert a.status is Status.UNKNOWN and a.unseen == ("hardhat",)  # vest seen, hardhat not seen


def test_item_in_the_frame_but_not_on_anyone_is_not_credited():
    floor_hardhat = det("Hardhat", (300, 380, 340, 400))  # lying between the two people
    vest_a = det("Safety Vest", (110, 180, 190, 320))
    result = assess([PERSON_A, PERSON_B, floor_hardhat, vest_a], SCHEMA)
    assert floor_hardhat in result.unattributed
    assert all(p.status is not Status.COMPLIANT for p in result.persons)


def test_vest_at_head_height_does_not_count_as_worn():
    high_vest = det("Safety Vest", (110, 0, 190, 110))  # above the person, outside the torso region
    result = assess([PERSON_A, high_vest], SCHEMA, required=("vest",))
    assert result.persons[0].status is Status.UNKNOWN and high_vest in result.unattributed


def test_strongest_detection_wins_when_both_present_and_missing_are_seen():
    worn = det("Hardhat", (120, 100, 180, 140), score=0.6)
    missing = det("NO-Hardhat", (120, 100, 180, 140), score=0.8)
    result = assess([PERSON_A, worn, missing], SCHEMA, required=("hardhat",))
    assert result.persons[0].status is Status.VIOLATION


def test_overlapping_people_item_goes_to_the_tighter_box():
    big = det("Person", (80, 80, 260, 500))
    small = det("Person", (100, 100, 200, 400))
    hardhat = det("Hardhat", (130, 100, 170, 130))  # fully inside both head regions
    result = assess([big, small, hardhat], SCHEMA, required=("hardhat",))
    big_a, small_a = result.persons
    assert "hardhat" in small_a.items and "hardhat" not in big_a.items


def test_not_required_items_do_not_cause_violations():
    no_mask = det("NO-Mask", (130, 120, 170, 160))
    hardhat = det("Hardhat", (120, 95, 180, 140))
    vest = det("Safety Vest", (110, 180, 190, 320))
    result = assess([PERSON_A, no_mask, hardhat, vest], SCHEMA, required=("hardhat", "vest"))
    assert result.persons[0].status is Status.COMPLIANT
    result = assess([PERSON_A, no_mask, hardhat, vest], SCHEMA, required=("hardhat", "vest", "mask"))
    assert result.persons[0].status is Status.VIOLATION and result.persons[0].missing == ("mask",)


def test_min_overlap_threshold_is_respected():
    edge_hardhat = det("Hardhat", (180, 100, 240, 140))  # one third inside A's head region
    assert assess([PERSON_A, edge_hardhat], SCHEMA, min_overlap=0.5).unattributed == [edge_hardhat]
    assert assess([PERSON_A, edge_hardhat], SCHEMA, min_overlap=0.3).unattributed == []


def test_required_item_the_model_cannot_detect_is_never_confirmed():
    from ppe_detection.compliance import undetectable

    hardhat_only = PpeSchema.from_labels(["Person", "Hardhat", "NO-Hardhat"])
    person = Detection(0, "Person", 0.9, (100, 100, 200, 400))
    hardhat = Detection(1, "Hardhat", 0.9, (120, 95, 180, 140))
    assert undetectable(hardhat_only, ("hardhat", "vest")) == ("vest",)
    (p,) = assess([person, hardhat], hardhat_only, required=("hardhat", "vest")).persons
    assert p.status is Status.UNKNOWN and p.unseen == ("vest",)
