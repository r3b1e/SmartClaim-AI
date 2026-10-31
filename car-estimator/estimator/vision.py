"""Runs the trained YOLO models on a photo and returns the evidence the pricing step needs."""

from functools import lru_cache
from pathlib import Path

from .pricing import Detection, load_json

REPO = Path(__file__).resolve().parents[2]
DEFAULT_CLS_WEIGHTS = REPO / "runs/classify/yolo_runs/car_damage_cls/weights/best.pt"
DEFAULT_PARTS_WEIGHTS = REPO / "runs/segment/car_parts/weights/best.pt"
# CarDD is preferred once trained; the DrBimmer damage model is the open-licence fallback
DAMAGE_WEIGHTS_CANDIDATES = [
    REPO / "runs/segment/cardd_seg/weights/best.pt",
    REPO / "runs/segment/car_damage/weights/best.pt",
]
MIN_PART_OVERLAP = 0.2  # share of a damage mask that must fall on a part to attribute it


def default_damage_weights():
    return next((w for w in DAMAGE_WEIGHTS_CANDIDATES if w.exists()), DAMAGE_WEIGHTS_CANDIDATES[-1])


def device():
    import torch
    if torch.cuda.is_available():
        return 0
    return "mps" if torch.backends.mps.is_available() else "cpu"


@lru_cache(maxsize=None)
def load_model(weights):
    from ultralytics import YOLO
    return YOLO(str(weights))


def normalise(name):
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def classify(image, weights=DEFAULT_CLS_WEIGHTS):
    """Return (label, confidence), e.g. ("F_Crushed", 0.91)."""
    result = load_model(weights).predict(image, device=device(), verbose=False)[0]
    return result.names[result.probs.top1], float(result.probs.top1conf)


def segment(image, weights, min_confidence):
    """Run a segmentation model; returns the ultralytics result (or None if nothing found)."""
    result = load_model(weights).predict(image, device=device(), conf=min_confidence, retina_masks=True, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0 or result.masks is None:
        return None
    return result


def detect_damage(image, damage_weights=None, parts_weights=DEFAULT_PARTS_WEIGHTS, min_confidence=0.3):
    """Return damage detections, each attributed to the car part it overlaps most.

    Also returns the raw (damage, parts) results so callers can draw them.
    """
    damage_weights = damage_weights or default_damage_weights()
    parts = segment(image, parts_weights, min_confidence) if Path(parts_weights).exists() else None
    damage = segment(image, damage_weights, min_confidence)
    if damage is None:
        return [], (None, parts)
    part_classes = load_json("damage_rules.json")["part_classes"]
    known_damage = load_json("damage_rules.json")["damage_actions"]

    dmg_masks = damage.masks.data.bool()
    image_area = dmg_masks[0].numel()
    part_masks = parts.masks.data.bool() if parts is not None else None

    detections = []
    for mask, cls, conf in zip(dmg_masks, damage.boxes.cls, damage.boxes.conf):
        damage_type = normalise(damage.names[int(cls)])
        if damage_type not in known_damage:
            continue
        area = int(mask.sum())
        part, part_fraction = None, None
        if part_masks is not None and area:
            overlaps = (part_masks & mask).sum(dim=(1, 2)).float() / area
            best = int(overlaps.argmax())
            part_name = part_classes.get(normalise(parts.names[int(parts.boxes.cls[best])]))
            if overlaps[best] >= MIN_PART_OVERLAP and part_name:
                part = part_name
                part_fraction = area / max(int(part_masks[best].sum()), 1)
        detections.append(Detection(damage_type, area / image_area, float(conf), part, part_fraction))
    return detections, (damage, parts)


def annotate(results):
    """RGB image with damage masks drawn (parts drawn underneath when available)."""
    damage, parts = results
    if damage is None:
        # no damage regions found: still show which car parts were recognised
        return parts.plot(boxes=False, conf=False)[..., ::-1] if parts is not None else None
    canvas = parts.plot(boxes=False, labels=False, conf=False) if parts is not None else None
    drawn = damage.plot(img=canvas) if canvas is not None else damage.plot()
    return drawn[..., ::-1]  # BGR -> RGB
