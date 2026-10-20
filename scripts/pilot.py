#!/usr/bin/env python3
"""Two-epoch pilot only: representative frozen train/val subsets, full-image validation."""
import argparse,json,random,sys,time
from pathlib import Path
import cv2,numpy as np,torch,yaml
from torch import nn
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from steel_defect_segmentation.dataset import RandomCropOrResize,SteelDefectDataset
from steel_defect_segmentation.labels import labels_by_image,read_label_table,decode_multilabel
from steel_defect_segmentation.metrics import ConfusionAccumulator
from steel_defect_segmentation.model import ResNet18UNet
from steel_defect_segmentation.tiling import tile_image,stitch_logits
def pick(ids, labels, n=12):
 """Normal plus one representative of each class, then deterministic remaining IDs."""
 out=[]
 for c in range(0,5):
  choices=[i for i in ids if (not labels.get(i,{}) if c==0 else isinstance(labels.get(i,{}).get(c),str) and labels[i][c].strip())]
  if choices: out.append(sorted(choices)[0])
 for i in ids:
  if i not in out: out.append(i)
  if len(out)>=n: break
 return out
def evaluate(model,ids,root,cfg,labels,dev,out):
 model.eval(); acc=ConfusionAccumulator(); lossfn=nn.BCEWithLogitsLoss();losses=[]
 with torch.no_grad():
  for k,image_id in enumerate(ids):
   image=cv2.cvtColor(cv2.imread(str(root/cfg['image_dir']/image_id)),cv2.COLOR_BGR2RGB); target=decode_multilabel(labels.get(image_id,{}),*image.shape[:2]).astype(np.float32); chunks=tile_image(image,256,512,256,384); logits=[];positions=[]
   for top,left,tile in chunks:
    x=torch.from_numpy(np.ascontiguousarray(tile.transpose(2,0,1))).float().div(255).unsqueeze(0).to(dev);logits.append(model(x)[0].cpu().numpy());positions.append((top,left))
   full=stitch_logits(logits,positions,*image.shape[:2]); z=torch.from_numpy(full).unsqueeze(0);y=torch.from_numpy(target).unsqueeze(0);losses.append(lossfn(z,y).item());acc.update(z,y,cfg['training']['threshold'])
   if k<3:
    pred=(torch.sigmoid(z)[0].numpy()>=.5).any(0); gt=target.any(0); vis=cv2.cvtColor(image,cv2.COLOR_RGB2BGR); vis[gt]=(0,255,0);vis[pred]=(0,0,255);cv2.imwrite(str(out/f'val_overlay_{image_id}'),vis)
 return float(np.mean(losses)),acc.compute()
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/data.yaml');p.add_argument('--train-size',type=int,default=16);p.add_argument('--val-size',type=int,default=12);p.add_argument('--out',default='reports/pilot');a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);root=Path(cfg['data_root']);labels=labels_by_image(read_label_table(root/cfg['train_csv']));sd=Path(cfg['splits_dir']);train=pick((sd/'train_image_ids.txt').read_text().split(),labels,a.train_size);val=pick((sd/'val_image_ids.txt').read_text().split(),labels,a.val_size);out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'pilot_train_ids.txt').write_text('\n'.join(train)+'\n');(out/'pilot_val_ids.txt').write_text('\n'.join(val)+'\n')
 dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');ds=SteelDefectDataset(root/cfg['image_dir'],train,labels,RandomCropOrResize(256,512,cfg['seed']));dl=DataLoader(ds,batch_size=4,shuffle=True,num_workers=0);m=ResNet18UNet().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=cfg['training']['learning_rate'],weight_decay=cfg['training']['weight_decay']);lossfn=nn.BCEWithLogitsLoss();hist=[];start=time.perf_counter()
 for ep in range(2):
  m.train();losses=[];grad_ok=True
  for b in dl:
   x,y=b['image'].to(dev),b['mask'].to(dev,dtype=torch.float32);opt.zero_grad();z=m(x);assert x.shape[1:]==(3,256,512) and y.shape[1:]==(4,256,512);loss=lossfn(z,y);assert torch.isfinite(loss);loss.backward();grad_ok &= any(q.grad is not None and torch.isfinite(q.grad).all() for q in m.parameters());opt.step();losses.append(loss.item())
  vl,metric=evaluate(m,val,root,cfg,labels,dev,out);record={'epoch':ep+1,'train_loss':float(np.mean(losses)),'val_loss':vl,'gradient_finite':bool(grad_ok),**metric};hist.append(record);torch.save({'model':m.state_dict(),'epoch':ep+1,'optimizer':opt.state_dict()},out/'last.pt');
  if ep==0 or vl<=min(x['val_loss'] for x in hist[:-1]):torch.save({'model':m.state_dict(),'epoch':ep+1},out/'best.pt')
 result={'device':str(dev),'epochs':2,'train_ids':len(train),'val_ids':len(val),'elapsed_seconds':time.perf_counter()-start,'history':hist,'memory_allocated_bytes':torch.mps.current_allocated_memory() if dev.type=='mps' else None};(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
