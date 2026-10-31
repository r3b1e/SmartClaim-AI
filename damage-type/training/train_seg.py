"""Train a YOLO segmentation model on a converted dataset.

    train_seg.py --data parts_yolo/data.yaml  --name car_parts    # which part is where
    train_seg.py --data damage_yolo/data.yaml --name car_damage   # what kind of damage
    train_seg.py --data cardd_yolo/data.yaml  --name cardd_seg    # CarDD, once the licence is granted

Weights land in runs/segment/<name>/weights/best.pt, where car-estimator picks them up.
"""

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO

HERE = Path(__file__).parent
REPO = HERE.parents[1]
PROJECT = REPO / "runs/segment"


def default_device():
    if torch.cuda.is_available():
        return "0"
    return "mps" if torch.backends.mps.is_available() else "cpu"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--model", default="yolo11n-seg.pt", help="e.g. yolo11s-seg.pt for higher accuracy")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--batch", type=int, default=16, help="16 fits comfortably in 16 GB unified memory at 640px")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=20, help="stop early after this many epochs without improvement")
    p.add_argument("--device", default=default_device())
    p.add_argument("--resume", action="store_true", help="continue an interrupted run")
    args = p.parse_args()

    if args.resume:
        model = YOLO(str(PROJECT / args.name / "weights/last.pt"))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
            workers=args.workers, device=args.device, project=str(PROJECT), name=args.name,
            exist_ok=True, patience=args.patience,
        )
    split = "test" if "test:" in args.data.read_text() else "val"
    metrics = model.val(data=str(args.data), split=split, device=args.device)
    print(f"box mAP50-95: {metrics.box.map:.3f}  mask mAP50-95: {metrics.seg.map:.3f}")
    print("best weights:", PROJECT / args.name / "weights/best.pt")
