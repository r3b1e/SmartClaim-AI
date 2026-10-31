import pytest

from estimator import Detection, Policy, Vehicle, estimate
from estimator.policy import compulsory_deductible, part_depreciation


@pytest.mark.parametrize("age,rate", [(0.3, 0.0), (0.8, 0.05), (1.5, 0.10), (2.5, 0.15),
                                      (3.5, 0.25), (4.5, 0.35), (8, 0.40), (12, 0.50)])
def test_metal_depreciation_by_age(age, rate):
    assert part_depreciation("metal", age) == rate


def test_material_depreciation_ignores_age():
    assert part_depreciation("plastic", 0.1) == 0.5
    assert part_depreciation("glass", 15) == 0.0


def test_compulsory_deductible():
    assert compulsory_deductible(1200) == 1000
    assert compulsory_deductible(1500) == 1000
    assert compulsory_deductible(2000) == 2000


def test_normal_car_has_no_claim():
    est = estimate("F_Normal", 0.95, Vehicle(), Policy(vehicle_age_years=2))
    assert est.low is None and est.high is None
    assert any("No damage" in n for n in est.notes)


@pytest.mark.parametrize("label", ["F_Breakage", "F_Crushed", "R_Breakage", "R_Crushed"])
def test_low_never_exceeds_high(label):
    est = estimate(label, 0.9, Vehicle(), Policy(vehicle_age_years=3))
    assert 0 < est.low.repair_bill <= est.high.repair_bill


def test_crushed_costs_more_than_breakage():
    v, p = Vehicle(), Policy(vehicle_age_years=3)
    assert estimate("F_Crushed", 0.9, v, p).low.repair_bill > estimate("F_Breakage", 0.9, v, p).high.repair_bill


def test_segment_scales_cost():
    p = Policy(vehicle_age_years=3)
    hatch = estimate("R_Crushed", 0.9, Vehicle("hatchback"), p).high.repair_bill
    suv = estimate("R_Crushed", 0.9, Vehicle("suv"), p).high.repair_bill
    assert suv > hatch


def test_settlement_arithmetic():
    sc = estimate("F_Crushed", 0.9, Vehicle(), Policy(vehicle_age_years=3)).high
    assert sc.insurer_pays == pytest.approx(sc.repair_bill - sc.depreciation - sc.deductible, abs=1)
    assert sc.insurer_pays + sc.customer_pays == pytest.approx(sc.repair_bill, abs=1)


def test_zero_dep_removes_depreciation():
    sc = estimate("F_Crushed", 0.9, Vehicle(), Policy(vehicle_age_years=6, zero_dep=True)).high
    assert sc.depreciation == 0
    assert sc.customer_pays == sc.deductible


def test_total_loss_settles_at_idv():
    est = estimate("F_Crushed", 0.9, Vehicle("luxury"), Policy(vehicle_age_years=12, idv=100_000))
    assert est.high.total_loss
    assert est.high.insurer_pays == 100_000 - 1000


def test_low_confidence_is_flagged():
    est = estimate("R_Breakage", 0.3, Vehicle(), Policy(vehicle_age_years=1))
    assert any("manual inspection" in n for n in est.notes)


def test_detections_drive_line_items():
    dets = [Detection("lamp_broken", 0.02), Detection("scratch", 0.05)]
    est = estimate("F_Breakage", 0.9, Vehicle(), Policy(vehicle_age_years=2), dets)
    assert {(i.part, i.action) for i in est.low.items} == {("headlamp", "replace"), ("front_bumper", "paint")}
    # high adds the classifier's typical plan for a front breakage
    assert {(i.part, i.action) for i in est.high.items} == {
        ("headlamp", "replace"), ("front_bumper", "replace"), ("grille", "replace")}


def test_large_dent_is_replaced_in_high_plan():
    dets = [Detection("dent", 0.30)]
    est = estimate("R_Crushed", 0.9, Vehicle(), Policy(vehicle_age_years=2), dets)
    assert ("boot_lid", "repair") in {(i.part, i.action) for i in est.low.items}
    high = {(i.part, i.action) for i in est.high.items}
    assert ("boot_lid", "replace") in high
    assert ("rear_panel", "replace") in high  # typical hidden damage on a crushed rear


def test_duplicate_parts_keep_most_extensive_action():
    dets = [Detection("scratch", 0.05), Detection("crack", 0.05)]
    est = estimate("F_Breakage", 0.9, Vehicle(), Policy(vehicle_age_years=2), dets)
    assert [(i.part, i.action) for i in est.low.items] == [("front_bumper", "repair")]


def test_unknown_segment_rejected():
    with pytest.raises(ValueError):
        estimate("F_Breakage", 0.9, Vehicle("truck"), Policy(vehicle_age_years=2))


def test_part_from_part_model_overrides_zone():
    dets = [Detection("dent", 0.05, part="front_door", part_fraction=0.1)]
    est = estimate("R_Breakage", 0.9, Vehicle(), Policy(vehicle_age_years=2), dets)
    assert [(i.part, i.action) for i in est.low.items] == [("front_door", "repair")]


def test_dent_covering_most_of_part_is_replaced():
    dets = [Detection("dent", 0.05, part="front_door", part_fraction=0.6)]
    est = estimate("F_Breakage", 0.9, Vehicle(), Policy(vehicle_age_years=2), dets)
    assert ("front_door", "replace") in {(i.part, i.action) for i in est.high.items}


def test_scratch_on_glass_is_not_billed():
    dets = [Detection("scratch", 0.02, part="windshield"), Detection("broken_part", 0.02, part="mirror")]
    est = estimate("F_Breakage", 0.9, Vehicle(), Policy(vehicle_age_years=2), dets)
    assert [i.part for i in est.low.items] == ["mirror"]


def test_damage_found_despite_normal_classification():
    dets = [Detection("scratch", 0.02, part="front_bumper")]
    est = estimate("F_Normal", 0.7, Vehicle(), Policy(vehicle_age_years=2), dets)
    assert est.high is not None
    assert any("verify" in n for n in est.notes)


def test_classifier_zone_corrects_front_rear_part_confusion():
    dets = [Detection("broken_part", 0.05, part="front_bumper"), Detection("dent", 0.05, part="front_door")]
    est = estimate("R_Breakage", 0.9, Vehicle(), Policy(vehicle_age_years=2), dets)
    assert {i.part for i in est.low.items} == {"rear_bumper", "front_door"}


def test_weak_detection_still_reflects_classified_severity():
    dets = [Detection("broken_part", 0.02, part="grille")]
    est = estimate("F_Crushed", 0.9, Vehicle(), Policy(vehicle_age_years=2), dets)
    assert [i.part for i in est.low.items] == ["grille"]
    assert est.high.repair_bill == estimate("F_Crushed", 0.9, Vehicle(), Policy(vehicle_age_years=2)).high.repair_bill


def test_normal_classification_adds_nothing_to_detections():
    dets = [Detection("broken_part", 0.02, part="mirror")]
    est = estimate("F_Normal", 0.9, Vehicle(), Policy(vehicle_age_years=2), dets)
    assert [i.part for i in est.high.items] == ["mirror"]
