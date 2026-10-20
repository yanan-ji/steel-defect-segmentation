#!/usr/bin/env python3
"""Create fixed original-image splits before any patch extraction."""
import argparse, sys
from pathlib import Path
import yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steel_defect_segmentation.labels import labels_by_image, read_label_table
from steel_defect_segmentation.splits import make_stratified_image_splits, save_splits

def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/data.yaml"); args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text()); root = Path(cfg["data_root"])
    table = read_label_table(root / cfg["train_csv"]); f = cfg["splits"]
    labels = dict(labels_by_image(table))
    # Competition CSV commonly omits entirely defect-free images; recover them from train_images.
    for path in (root / cfg["image_dir"]).glob("*.jpg"): labels.setdefault(path.name, {})
    splits = make_stratified_image_splits(labels, cfg["seed"], f["train"], f["val"], f["test"])
    save_splits(splits, cfg["splits_dir"])
    (Path(cfg["splits_dir"]) / "split_metadata.yaml").write_text(yaml.safe_dump({"seed":cfg["seed"], "fractions":f, "strategy":"original-image 4-bit multilabel stratification; immutable after creation"}))
    print({name: len(ids) for name, ids in splits.items()})
if __name__ == "__main__": main()
