"""
Builds a 4:1:1 stratified train/val/test split (matching LAD-Net paper's
AppleSet6 split ratio, Section 4.1.1) from the pooled Tomato9 data built by
extract_tomato9.py.

Tomato___Bacterial_spot is excluded from the split: the source AI Challenger
2018 data has essentially no usable images for this class (1 in train, 0 in
validation -- classes 44/45 were reportedly dropped by the competition
organizers themselves due to data quality issues, per community writeups).
The folder is left in data/tomato9_pooled/ for the record, but the actual
train/val/test split covers the other 9 classes only (8 diseases + healthy).
"""
import argparse
import random
import shutil
from pathlib import Path

EXCLUDED_CLASSES = {"Tomato___Bacterial_spot"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=r"E:\plant_disease\LADNet_repro\data\tomato9_pooled")
    ap.add_argument("--output", default=r"E:\plant_disease\LADNet_repro\data\tomato9_split")
    ap.add_argument("--train_frac", type=float, default=4 / 6)
    ap.add_argument("--val_frac", type=float, default=1 / 6)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    random.seed(args.seed)
    source = Path(args.source)
    out = Path(args.output)

    classes = sorted(d.name for d in source.iterdir() if d.is_dir() and d.name not in EXCLUDED_CLASSES)
    print(f"classes used ({len(classes)}): {classes}")
    print(f"excluded (insufficient data): {sorted(EXCLUDED_CLASSES)}")

    totals = {"train": 0, "val": 0, "test": 0}
    for c in classes:
        files = [f for f in (source / c).iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
        random.shuffle(files)
        n = len(files)
        n_train = round(n * args.train_frac)
        n_val = round(n * args.val_frac)
        splits = {"train": files[:n_train], "val": files[n_train:n_train + n_val], "test": files[n_train + n_val:]}
        for split, split_files in splits.items():
            (out / split / c).mkdir(parents=True, exist_ok=True)
            for f in split_files:
                shutil.copy2(f, out / split / c / f.name)
            totals[split] += len(split_files)
        print(f"  {c:45s} total={n:5d} train={len(splits['train']):5d} val={len(splits['val']):5d} test={len(splits['test']):5d}")

    print(f"\nTOTAL train={totals['train']} val={totals['val']} test={totals['test']} (grand total={sum(totals.values())})")
    print(f"saved to: {out}")


if __name__ == "__main__":
    main()
