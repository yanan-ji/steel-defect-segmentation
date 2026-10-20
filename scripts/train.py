#!/usr/bin/env python3
"""Formal training entry point; consumes existing immutable manifests only."""
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
def choose_device(): return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
def main():
 p=argparse.ArgumentParser();p.add_argument("--config",default="configs/data.yaml");p.add_argument("--epochs",type=int);p.add_argument("--out",default="reports/training");a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());seed=cfg["seed"];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
 root=Path(cfg["data_root"]); split=Path(cfg["splits_dir"])
 if not all((split/f"{n}_image_ids.txt").exists() for n in ("train","val","test")): raise FileNotFoundError("Create and audit fixed split manifests first; training never creates splits.")
 labels=labels_by_image(read_label_table(root/cfg["train_csv"])); patch=cfg["patch"]; make=lambda n,shuffle:DataLoader(SteelDefectDataset(root/cfg["image_dir"],(split/f"{n}_image_ids.txt").read_text().split(),labels,RandomCropOrResize(patch["height"],patch["width"],seed)),batch_size=cfg["loader"]["batch_size"],shuffle=shuffle,num_workers=cfg["loader"]["num_workers"])
 train,val=make("train",True),make("val",False);dev=choose_device();model=ResNet18UNet().to(dev);criterion=nn.BCEWithLogitsLoss();opt=torch.optim.AdamW(model.parameters(),lr=cfg["training"]["learning_rate"],weight_decay=cfg["training"]["weight_decay"]);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",patience=2,factor=.5);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);best=float("inf");stale=0;history=[]
 for epoch in range(a.epochs or cfg["training"]["epochs"]):
  model.train(); losses=[]
  for b in train:
   x,y=b["image"].to(dev),b["mask"].to(dev,dtype=torch.float32);opt.zero_grad();loss=criterion(model(x),y);loss.backward();opt.step();losses.append(loss.item())
  model.eval(); vals=[]; all_logits=[];all_y=[]
  with torch.no_grad():
   for b in val:
    x,y=b["image"].to(dev),b["mask"].to(dev,dtype=torch.float32);z=model(x);vals.append(criterion(z,y).item());all_logits.append(z);all_y.append(y)
  v=float(np.mean(vals)); scheduler.step(v); record={"epoch":epoch+1,"train_loss":float(np.mean(losses)),"val_loss":v,"lr":opt.param_groups[0]["lr"],**segmentation_metrics(torch.cat(all_logits),torch.cat(all_y),cfg["training"]["threshold"])};history.append(record);print(json.dumps(record))
  if v<best: best=v;stale=0;torch.save({"model":model.state_dict(),"epoch":epoch+1,"val_loss":v,"seed":seed},out/"best.pt")
  else: stale+=1
  if stale>=cfg["training"]["patience"]: print("early stopping");break
 (out/"history.json").write_text(json.dumps(history,indent=2))
if __name__=="__main__":main()
