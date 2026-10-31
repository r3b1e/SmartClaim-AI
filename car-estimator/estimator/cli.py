"""Photo + vehicle/policy details -> insurance claim estimate.

Example:
    python -m estimator.cli photo.jpg --segment sedan --vehicle-age 3 --engine-cc 1200 --idv 550000
"""

import argparse
import json
from pathlib import Path

from .policy import Policy
from .pricing import Vehicle, estimate
from . import vision


def format_inr(x):
    return f"₹{x:,.0f}"


def print_estimate(est):
    print(f"Damage: {est.damage_label}  (confidence {est.confidence:.0%})")
    for name, sc in (("LOW", est.low), ("HIGH", est.high)):
        if sc is None:
            continue
        print(f"\n{name} estimate")
        for i in sc.items:
            print(f"  {i.part:<17} {i.action:<8} {format_inr(i.bill):>10}")
        print(f"  {'Repair bill (incl. GST)':<27} {format_inr(sc.repair_bill):>10}")
        print(f"  {'- depreciation':<27} {format_inr(sc.depreciation):>10}")
        print(f"  {'- deductible':<27} {format_inr(sc.deductible):>10}")
        print(f"  {'Insurer pays':<27} {format_inr(sc.insurer_pays):>10}")
        print(f"  {'You pay':<27} {format_inr(sc.customer_pays):>10}")
        if sc.total_loss:
            print("  ** Constructive total loss: settled at IDV **")
    for note in est.notes:
        print(f"\nNote: {note}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("image")
    p.add_argument("--segment", default="hatchback", choices=["hatchback", "sedan", "suv", "luxury"])
    p.add_argument("--city-tier", default="tier1", choices=["tier1", "tier2", "tier3"])
    p.add_argument("--vehicle-age", type=float, required=True, help="years since first registration")
    p.add_argument("--engine-cc", type=int, default=1200)
    p.add_argument("--idv", type=float, help="Insured Declared Value in INR")
    p.add_argument("--zero-dep", action="store_true", help="policy has the zero-depreciation add-on")
    p.add_argument("--voluntary-deductible", type=int, default=0)
    p.add_argument("--cls-weights", type=Path, default=vision.DEFAULT_CLS_WEIGHTS)
    p.add_argument("--damage-weights", type=Path, default=vision.default_damage_weights(),
                   help="damage segmentation model (CarDD or car_damage); skipped if missing")
    p.add_argument("--parts-weights", type=Path, default=vision.DEFAULT_PARTS_WEIGHTS,
                   help="car-part segmentation model; locates which part is damaged")
    p.add_argument("--json", action="store_true", help="print the estimate as JSON")
    args = p.parse_args(argv)

    label, conf = vision.classify(args.image, args.cls_weights)
    detections = None
    if args.damage_weights.exists():
        detections, _ = vision.detect_damage(args.image, args.damage_weights, args.parts_weights)
    est = estimate(
        label, conf,
        Vehicle(args.segment, args.city_tier),
        Policy(args.vehicle_age, args.engine_cc, args.idv, args.zero_dep, args.voluntary_deductible),
        detections,
    )
    if detections is None:
        est.notes.append("Damage model not found; estimate uses the location/severity classifier only.")
    if args.json:
        print(json.dumps(est.to_dict(), indent=2))
    else:
        print_estimate(est)


if __name__ == "__main__":
    main()
