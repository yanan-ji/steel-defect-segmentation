#!/usr/bin/env python3
"""Report real label statistics, sample overlays, and saved-split leakage."""
import argparse, sys, json
from pathlib import Path
import cv2, numpy as np, yaml
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from steel_defect_segmentation.labels import NUM_CLASSES, decode_multilabel, labels_by_image, read_label_table

def main():
    p = argparse.ArgumentParser(); p.add_argument("--config", default="configs/data.yaml"); p.add_argument("--visualize", type=int, default=0); p.add_argument("--output", default="reports/figures"); args = p.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text()); root = Path(cfg["data_root"]); table = read_label_table(root / cfg["train_csv"]); grouped = dict(labels_by_image(table))
    for path in (root / cfg["image_dir"]).glob("*.jpg"): grouped.setdefault(path.name, {})
    positive = table[table.EncodedPixels.notna()].groupby("ClassId").ImageId.nunique()
    print(f"images: {len(grouped)}")
    empty = sum(not any(isinstance(v,str) and v.strip() for v in rows.values()) for rows in grouped.values())
    print(f"no-defect image ratio: {empty / len(grouped):.6f}")
    for c in range(1, NUM_CLASSES+1): print(f"class {c} defect images: {positive.get(c,0)}")
    counts = np.zeros(NUM_CLASSES, dtype=np.uint64); total_pixels = 0
    for image_id, rows in grouped.items():
        image = cv2.imread(str(root / cfg["image_dir"] / image_id))
        if image is None: raise FileNotFoundError(image_id)
        masks = decode_multilabel(rows, *image.shape[:2]); counts += masks.sum(axis=(1,2)); total_pixels += image.shape[0]*image.shape[1]
    pixel_props={str(c):float(count/total_pixels) for c,count in enumerate(counts,1)}
    for c, value in pixel_props.items(): print(f"class {c} defect pixel proportion: {value:.8f}")
    split_dir = Path(cfg["splits_dir"]); splits = {n:set((split_dir/f"{n}_image_ids.txt").read_text().split()) for n in ("train","val","test")}
    overlap = set.union(*(splits[a]&splits[b] for a,b in (("train","val"),("train","test"),("val","test"))))
    if overlap: raise RuntimeError(f"cross-split duplicate check FAILED ({len(overlap)} IDs)")
    print("cross-split duplicate check: PASS")
    split_stats={}
    for name, ids in splits.items():
        subset=table[table.ImageId.isin(ids)]; split_stats[name]={"images":len(ids),"no_defect_images":sum(not any(isinstance(v,str) and v.strip() for v in grouped[i].values()) for i in ids),"class_defect_images":{str(c):int(subset[(subset.ClassId==c)&subset.EncodedPixels.notna()].ImageId.nunique()) for c in range(1,5)}}
    report={"images":len(grouped),"no_defect_ratio":empty/len(grouped),"class_defect_images":{str(c):int(positive.get(c,0)) for c in range(1,5)},"class_pixel_proportions":pixel_props,"split_distribution":split_stats,"cross_split_duplicates":0}
    Path("reports").mkdir(exist_ok=True); Path("reports/data_audit.json").write_text(json.dumps(report,indent=2)); print("saved reports/data_audit.json")
    if args.visualize:
        out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
        for image_id, rows in list(grouped.items())[:args.visualize]:
            image=cv2.imread(str(root/cfg["image_dir"]/image_id)); mask=decode_multilabel(rows,*image.shape[:2]).any(axis=0)
            overlay=image.copy(); overlay[mask.astype(bool)] = (0,0,255)
            cv2.imwrite(str(out/image_id),cv2.addWeighted(image,.65,overlay,.35,0))
if __name__ == "__main__": main()
