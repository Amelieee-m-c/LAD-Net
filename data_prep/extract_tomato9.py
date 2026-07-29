"""
Extracts the "Tomato9" subset (9 diseases + healthy, matching PlantVillage's
tomato taxonomy) from the raw AI Challenger 2018 Plant Disease Recognition
dataset (train + validation annotations; testA/testB have no released
labels, so they're unused here).

Class ID -> name mapping sourced from
https://github.com/foamliu/Crop-Disease-Detection/blob/master/labels.csv
(cross-checked: matches the known 61-class "species-disease-severity"
scheme). Severity pairs (general/serious) are merged into one class each,
matching PlantVillage/Tomato9's flat per-disease taxonomy. Class 42/43
(tomato powdery mildew) is deliberately excluded -- PlantVillage's tomato
set has no powdery mildew class, and excluding it is exactly what turns
AI Challenger's 11-category tomato section into "Tomato9" (9 diseases +
healthy = 10 classes).
"""
import argparse
import json
import shutil
from pathlib import Path

CLASS_MAP = {
    41: "Tomato___healthy",
    44: "Tomato___Bacterial_spot", 45: "Tomato___Bacterial_spot",
    46: "Tomato___Early_blight", 47: "Tomato___Early_blight",
    48: "Tomato___Late_blight", 49: "Tomato___Late_blight",
    50: "Tomato___Leaf_Mold", 51: "Tomato___Leaf_Mold",
    52: "Tomato___Target_Spot", 53: "Tomato___Target_Spot",
    54: "Tomato___Septoria_leaf_spot", 55: "Tomato___Septoria_leaf_spot",
    56: "Tomato___Spider_mites", 57: "Tomato___Spider_mites",
    58: "Tomato___Tomato_Yellow_Leaf_Curl_Virus", 59: "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    60: "Tomato___Tomato_mosaic_virus",
    # 42, 43 (powdery mildew) deliberately excluded -- not part of Tomato9
}


def process_split(json_path: Path, images_dir: Path, out_dir: Path):
    with open(json_path, encoding="utf-8") as f:
        annotations = json.load(f)

    counts = {}
    skipped_missing = 0
    for entry in annotations:
        cls_id = entry["disease_class"]
        if cls_id not in CLASS_MAP:
            continue
        class_name = CLASS_MAP[cls_id]
        src = images_dir / entry["image_id"]
        if not src.exists():
            skipped_missing += 1
            continue
        dst_dir = out_dir / class_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / entry["image_id"])
        counts[class_name] = counts.get(class_name, 0) + 1

    return counts, skipped_missing


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", default=r"E:\plant_disease\LADNet_repro\data_raw\ai_challenger_extracted\unzipped")
    ap.add_argument("--output", default=r"E:\plant_disease\LADNet_repro\data\tomato9_pooled")
    args = ap.parse_args()

    raw_root = Path(args.raw_root)
    out_dir = Path(args.output)

    total_counts = {}
    for split, json_name in [
        ("train", "AgriculturalDisease_train_annotations.json"),
        ("val", "AgriculturalDisease_validation_annotations.json"),
    ]:
        split_dir = raw_root / split / f"AgriculturalDisease_{'trainingset' if split == 'train' else 'validationset'}"
        json_path = split_dir / json_name
        images_dir = split_dir / "images"
        print(f"processing {split}: {json_path}")
        counts, skipped = process_split(json_path, images_dir, out_dir)
        print(f"  skipped (missing file): {skipped}")
        for k, v in counts.items():
            total_counts[k] = total_counts.get(k, 0) + v

    print("\n=== Tomato9 pooled totals ===")
    grand_total = 0
    for name, count in sorted(total_counts.items()):
        print(f"  {name:45s} {count}")
        grand_total += count
    print(f"\nTOTAL: {grand_total} images across {len(total_counts)} classes")
    print(f"saved to: {out_dir}")


if __name__ == "__main__":
    main()
