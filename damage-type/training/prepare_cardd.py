"""Convert the CarDD COCO release into the YOLO segmentation layout.

Point --src at the unzipped CarDD release. The script finds every COCO
`instances_*.json` under it, works out the split (train/val/test) from the file
name, locates the matching image folder, and writes:

    cardd_yolo/
      images/{train,val,test}/   hard links to the original images (no extra disk)
      labels/{train,val,test}/   one .txt per image: class x1 y1 x2 y2 ... (normalised polygon)
      data.yaml
"""

import argparse
import json
import os
import shutil
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent


def split_name(ann_file):
    stem = ann_file.stem.lower()
    for split in ("train", "val", "test"):
        if split in stem:
            return split
    raise ValueError(f"cannot tell the split of {ann_file}")


def find_image_dir(src, file_name):
    for path in src.rglob(Path(file_name).name):
        return path.parent
    raise FileNotFoundError(f"image {file_name} not found under {src}")


def polygons(ann, w, h):
    """Yield normalised polygons; falls back to the bbox for RLE-encoded masks."""
    seg = ann.get("segmentation")
    if isinstance(seg, list) and seg:
        for poly in seg:
            if len(poly) >= 6:
                yield [v / (w if i % 2 == 0 else h) for i, v in enumerate(poly)]
    else:
        x, y, bw, bh = ann["bbox"]
        yield [x / w, y / h, (x + bw) / w, y / h, (x + bw) / w, (y + bh) / h, x / w, (y + bh) / h]


def link(src, dst):
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def convert(src, out):
    ann_files = sorted(src.rglob("instances_*.json"))
    if not ann_files:
        raise SystemExit(f"no COCO instances_*.json found under {src}")

    names = None
    for ann_file in ann_files:
        split = split_name(ann_file)
        coco = json.loads(ann_file.read_text())
        cats = sorted(coco["categories"], key=lambda c: c["id"])
        split_names = [c["name"].strip().lower().replace(" ", "_") for c in cats]
        if names is None:
            names = split_names
        elif names != split_names:
            raise SystemExit(f"category mismatch in {ann_file}: {split_names} vs {names}")
        cat_index = {c["id"]: i for i, c in enumerate(cats)}

        by_image = {}
        for ann in coco["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann)

        img_dir = find_image_dir(src, coco["images"][0]["file_name"])
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

        counts = Counter()
        for img in coco["images"]:
            file_name = Path(img["file_name"]).name
            link(img_dir / file_name, out / "images" / split / file_name)
            lines = []
            for ann in by_image.get(img["id"], []):
                cls = cat_index[ann["category_id"]]
                for poly in polygons(ann, img["width"], img["height"]):
                    lines.append(" ".join([str(cls)] + [f"{min(max(v, 0), 1):.6f}" for v in poly]))
                counts[names[cls]] += 1
            (out / "labels" / split / Path(file_name).with_suffix(".txt")).write_text("\n".join(lines))
        print(f"{split}: {len(coco['images'])} images, {sum(counts.values())} instances {dict(counts)}")

    splits = {s: f"images/{s}" for s in ("train", "val", "test") if (out / "images" / s).exists()}
    yaml = [f"path: {out.resolve()}"] + [f"{s}: {p}" for s, p in splits.items()]
    yaml += ["names:"] + [f"  {i}: {n}" for i, n in enumerate(names)]
    (out / "data.yaml").write_text("\n".join(yaml) + "\n")
    print(f"wrote {out / 'data.yaml'} with classes {names}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", type=Path, required=True, help="unzipped CarDD release folder")
    p.add_argument("--out", type=Path, default=HERE / "cardd_yolo")
    args = p.parse_args()
    convert(args.src, args.out)
