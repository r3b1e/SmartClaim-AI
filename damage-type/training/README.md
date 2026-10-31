# Damage and part segmentation models

The claim estimator uses two segmentation models on top of the front/rear classifier:

| Model | Finds | Weights |
|---|---|---|
| `car_parts` | 21 parts (bumpers, hood, doors, lamps, glass, wheels…) | `runs/segment/car_parts/weights/best.pt` |
| `car_damage` | 8 damage types (dent, scratch, broken/missing part, paint chip, flaking, corrosion, crack) | `runs/segment/car_damage/weights/best.pt` |

Each damage mask is attributed to the part it overlaps most, so the estimate prices the actual damaged part.

## Train on the DrBimmer dataset (MIT licence)

From the repository root, using the project venv:

```
.venv/bin/python -c "from huggingface_hub import snapshot_download; snapshot_download('DrBimmer/car-parts-and-damage-dataset', repo_type='dataset', local_dir='damage-type/training/data/drbimmer')"
cd damage-type/training
# note: the dataset's folder names are swapped -- "Car damages dataset" holds the parts labels and vice versa
../../.venv/bin/python prepare_supervisely.py --src "data/drbimmer/Car damages dataset" --out parts_yolo
../../.venv/bin/python prepare_supervisely.py --src "data/drbimmer/Car parts dataset"   --out damage_yolo
../../.venv/bin/python train_seg.py --data parts_yolo/data.yaml  --name car_parts  --epochs 30
../../.venv/bin/python train_seg.py --data damage_yolo/data.yaml --name car_damage --epochs 40
```

## Optional: CarDD

CarDD (https://cardd-ustc.github.io) is licensed for non-commercial research; fill in its licensing form
to get the download link. Then:

```
../../.venv/bin/python prepare_cardd.py --src /path/to/CarDD_release
../../.venv/bin/python train_seg.py --data cardd_yolo/data.yaml --name cardd_seg
```

When `runs/segment/cardd_seg/weights/best.pt` exists, the estimator prefers it over `car_damage`.
