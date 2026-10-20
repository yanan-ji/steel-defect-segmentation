#!/usr/bin/env python3
"""One-epoch, bounded real-data smoke test; never a formal training command."""
import argparse, json, random, sys
from pathlib import Path
import numpy as np, torch, yaml
from torch import nn
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from steel_defect_segmentation.dataset import RandomCropOrResize, SteelDefectDataset
from steel_defect_segmentation.labels import labels_by_image, read_label_table
from steel_defect_segmentation.metrics import segmentation_metrics
from steel_defect_segmentation.model import ResNet18UNet
def device(): return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
def main():
 p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/data.yaml");p.add_argument("--max-train",type=int,default=8);p.add_argument("--max-val",type=int,default=4);p.add_argument("--out",default="reports/smoke");a=p.parse_args(); cfg=yaml.safe_load(Path(a.config).read_text()); random.seed(cfg["seed"]);np.random.seed(cfg["seed"]);torch.manual_seed(cfg["seed"])
 root=Path(cfg["data_root"]); labels=labels_by_image(read_label_table(root/cfg["train_csv"])); split=Path(cfg["splits_dir"]); train=(split/"train_image_ids.txt").read_text().split()[:a.max_train]; val=(split/"val_image_ids.txt").read_text().split()[:a.max_val]
 transform=RandomCropOrResize(cfg["patch"]["height"],cfg["patch"]["width"],cfg["seed"]); mk=lambda ids:DataLoader(SteelDefectDataset(root/cfg["image_dir"],ids,labels,transform),batch_size=min(cfg["loader"]["batch_size"],len(ids)),shuffle=True,num_workers=0)
 dev=device(); model=ResNet18UNet().to(dev); opt=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"]); loss_fn=nn.BCEWithLogitsLoss(); model.train(); losses=[]
 for batch in mk(train):
  x,y=batch["image"].to(dev),batch["mask"].to(dev,dtype=torch.float32); opt.zero_grad(); logits=model(x); loss=loss_fn(logits,y); assert torch.isfinite(loss); loss.backward(); assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()); opt.step(); losses.append(loss.item())
 model.eval(); vals=[]; metrics=[]
 with torch.no_grad():
  for batch in mk(val):
   x,y=batch["image"].to(dev),batch["mask"].to(dev,dtype=torch.float32); logits=model(x); vals.append(loss_fn(logits,y).item());metrics.append(segmentation_metrics(logits,y,cfg["training"]["threshold"]))
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True); torch.save({"model":model.state_dict(),"seed":cfg["seed"],"smoke":True},out/"smoke_checkpoint.pt"); result={"device":str(dev),"train_batches":len(losses),"val_batches":len(vals),"train_loss_mean":float(np.mean(losses)),"val_loss_mean":float(np.mean(vals)),"metrics":metrics};(out/"result.json").write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=="__main__": main()
