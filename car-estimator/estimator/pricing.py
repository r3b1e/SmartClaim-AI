"""Turns damage evidence into repair line items and an insurance claim estimate."""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path

from .policy import Policy, TOTAL_LOSS_THRESHOLD

HERE = Path(__file__).parent
ACTION_RANK = {"paint": 0, "repair": 1, "replace": 2}


def load_json(name):
    with open(HERE / name) as f:
        return json.load(f)


@dataclass
class Detection:
    damage_type: str      # e.g. "dent", "scratch", "lamp_broken"
    area_fraction: float  # damaged area / image area
    confidence: float = 1.0
    part: str | None = None             # priced part the damage sits on, if a part model found one
    part_fraction: float | None = None  # damaged area / area of that part


@dataclass
class Vehicle:
    segment: str = "hatchback"
    city_tier: str = "tier1"


@dataclass
class LineItem:
    part: str
    action: str
    parts: float = 0.0
    labour: float = 0.0
    paint: float = 0.0
    gst: float = 0.0
    depreciation: float = 0.0

    @property
    def bill(self):
        return self.parts + self.labour + self.paint + self.gst


@dataclass
class Scenario:
    items: list[LineItem]
    repair_bill: float
    depreciation: float
    deductible: float
    insurer_pays: float
    customer_pays: float
    total_loss: bool


@dataclass
class Estimate:
    damage_label: str
    confidence: float
    low: Scenario | None
    high: Scenario | None
    notes: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


def merge_plan(steps):
    """Keep one action per part, preferring the most extensive one."""
    plan = {}
    for part, action in steps:
        if part not in plan or ACTION_RANK[action] > ACTION_RANK[plan[part]]:
            plan[part] = action
    return list(plan.items())


def is_large_dent(det, rules):
    limits = rules["large_dent"]
    if det.part_fraction is not None:
        return det.part_fraction > limits["part_fraction"]
    return det.area_fraction > limits["image_fraction"]


def build_plans(label, detections, rules):
    """Return (low_plan, high_plan) as lists of (part, action)."""
    zone, level = label.split("_")
    if not detections:
        if level == "Normal":
            return [], []
        scenario = rules["classifier_scenarios"][label]
        return merge_plan(scenario["low"]), merge_plan(scenario["high"])

    low, high = [], []
    for det in detections:
        part = det.part or rules["zone_parts"][det.damage_type][zone]
        part = rules["zone_counterparts"][zone].get(part, part)
        if det.part:
            det.part = part  # keep the reported part consistent with the priced one
        low_action, high_action = rules["damage_actions"][det.damage_type]
        if det.damage_type == "dent" and is_large_dent(det, rules):
            high_action = "replace"
        low.append((part, low_action))
        high.append((part, high_action))
    # low = only what the damage model sees; high adds what is typical for the classified
    # severity, which covers damage hidden behind the panels
    if level != "Normal":
        high += [tuple(step) for step in rules["classifier_scenarios"][label]["high"]]
    return merge_plan(low), merge_plan(high)


def price_item(part, action, vehicle, policy, pricing):
    spec = pricing["parts"][part]
    seg = pricing["segments"][vehicle.segment]
    rate = pricing["labour_rate_per_hour"][vehicle.city_tier]
    gst = pricing["gst"]
    item = LineItem(part, action)

    if action in ("repair", "replace"):
        # parts without a repair price can only be replaced
        if action == "repair" and spec["repair"] is None:
            action = item.action = "replace"
        item.parts = spec[action] * seg
        item.labour = spec["labour_hours"][action] * rate
    if spec["painted"] or action == "paint":
        item.paint = pricing["paint_per_panel"] * seg

    item.gst = item.parts * gst["parts"] + item.labour * gst["labour"] + item.paint * gst["paint"]

    # Depreciation applies to replaced parts and to the material share of paint
    if action == "replace":
        item.depreciation += item.parts * (1 + gst["parts"]) * policy.depreciation(spec["material"])
    paint_material = item.paint * pricing["paint_material_share"] * (1 + gst["paint"])
    item.depreciation += paint_material * policy.paint_depreciation
    return item


def price_plan(plan, vehicle, policy, pricing):
    # refinishing only applies to painted panels; scratches on glass or lamps are cosmetic
    plan = [(part, action) for part, action in plan
            if action != "paint" or pricing["parts"][part]["painted"]]
    if not plan:
        return None
    items = [price_item(part, action, vehicle, policy, pricing) for part, action in plan]
    bill = sum(i.bill for i in items)
    depreciation = sum(i.depreciation for i in items)
    total_loss = policy.idv is not None and bill > TOTAL_LOSS_THRESHOLD * policy.idv
    if total_loss:
        insurer = max(policy.idv - policy.deductible, 0)
    else:
        insurer = max(bill - depreciation - policy.deductible, 0)
    return Scenario(
        items=items,
        repair_bill=round(bill),
        depreciation=round(depreciation),
        deductible=policy.deductible,
        insurer_pays=round(insurer),
        customer_pays=round(max(bill - insurer, 0)),
        total_loss=total_loss,
    )


def estimate(label, confidence, vehicle, policy, detections=None, min_confidence=0.5):
    """Estimate repair cost and claim payout from the damage classifier output."""
    pricing = load_json("pricing.json")
    rules = load_json("damage_rules.json")
    if vehicle.segment not in pricing["segments"]:
        raise ValueError(f"unknown segment {vehicle.segment!r}, expected one of {list(pricing['segments'])}")
    if vehicle.city_tier not in pricing["labour_rate_per_hour"]:
        raise ValueError(f"unknown city tier {vehicle.city_tier!r}")

    low_plan, high_plan = build_plans(label, detections or [], rules)
    result = Estimate(
        damage_label=label,
        confidence=confidence,
        low=price_plan(low_plan, vehicle, policy, pricing),
        high=price_plan(high_plan, vehicle, policy, pricing),
    )
    if label.endswith("Normal") and not detections:
        result.notes.append("No damage detected; nothing to claim.")
    elif label.endswith("Normal"):
        result.notes.append("Classifier saw no damage but the damage detector did; please verify the photo.")
    if confidence < min_confidence:
        result.notes.append(f"Low model confidence ({confidence:.0%}); manual inspection recommended.")
    if result.high and result.high.total_loss:
        result.notes.append("High estimate exceeds 75% of IDV: possible constructive total loss.")
    if policy.idv is None:
        result.notes.append("IDV not provided; total-loss check skipped.")
    return result
