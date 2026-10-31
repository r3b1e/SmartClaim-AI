"""Convert a Supervisely polygon dataset (ann/*.json + img/*) into the YOLO segmentation layout.

Used for the MIT-licensed DrBimmer "Car Parts and Damages" dataset:

    prepare_supervisely.py --src "data/drbimmer/Car parts dataset" --out parts_yolo
    prepare_supervisely.py --src "data/drbimmer/Car damages dataset" --out damage_yolo

Class names are normalised to snake_case ("Back-bumper" -> "back_bumper").
"""

import argparse
import json
import os
import random
import shutil
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent


def normalise(name):
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def link(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert(src, out, val_fraction, seed):
    ann_files = sorted(src.rglob("ann/*.json"))
    if not ann_files:
        raise SystemExit(f"no Supervisely ann/*.json found under {src}")

    samples, counts = [], Counter()
    for ann_file in ann_files:
        img = ann_file.parent.parent / "img" / ann_file.stem  # "x.jpg.json" -> img/"x.jpg"
        if not img.exists():
            continue
        ann = json.loads(ann_file.read_text())
        objects = [o for o in ann["objects"] if o["geometryType"] == "polygon" and len(o["points"]["exterior"]) >= 3]
        if not objects:
            continue
        samples.append((img, ann["size"], objects))
        counts.update(normalise(o["classTitle"]) for o in objects)

    names = sorted(counts)
    index = {n: i for i, n in enumerate(names)}
    random.Random(seed).shuffle(samples)
    n_val = max(1, int(len(samples) * val_fraction))
    splits = {"val": samples[:n_val], "train": samples[n_val:]}

    for split, items in splits.items():
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, size, objects in items:
            w, h = size["width"], size["height"]
            # unique, filesystem-safe name without spaces
            stem = img.stem.replace(" ", "_")
            link(img, out / "images" / split / f"{stem}{img.suffix.lower()}")
            lines = []
            for o in objects:
                coords = [f"{min(max(v / (w if i % 2 == 0 else h), 0), 1):.6f}"
                          for pt in o["points"]["exterior"] for i, v in enumerate(pt)]
                lines.append(" ".join([str(index[normalise(o["classTitle"])])] + coords))
            (out / "labels" / split / f"{stem}.txt").write_text("\n".join(lines))
        print(f"{split}: {len(items)} images")

    yaml = [f"path: {out.resolve()}", "train: images/train", "val: images/val", "names:"]
    yaml += [f"  {i}: {n}" for i, n in enumerate(names)]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n")
    print("instances per class:", dict(counts.most_common()))
    print(f"wrote {out / 'data.yaml'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--val-fraction", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    out = args.out if args.out.is_absolute() else HERE / args.out
    convert(args.src, out, args.val_fraction, args.seed)
