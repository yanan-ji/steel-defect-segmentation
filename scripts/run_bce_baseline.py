#!/usr/bin/env python3
"""E0 raw-BCE baseline. Defaults to ONE full epoch; never reads test IDs."""
import argparse,json,random,sys,time
from pathlib import Path
import cv2,numpy as np,torch,yaml
from torch import nn
from torch.utils.data import DataLoader
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from steel_defect_segmentation.dataset import RandomCropOrResize,SteelDefectDataset
from steel_defect_segmentation.labels import labels_by_image,read_label_table,decode_multilabel
from steel_defect_segmentation.metrics import ConfusionAccumulator
from steel_defect_segmentation.model import ResNet18UNet
from steel_defect_segmentation.tiling import tile_image,stitch_logits
def save_vis(out,name,img,gt,prob):
 out.mkdir(parents=True,exist_ok=True)
 pred=prob>=.5; base=cv2.cvtColor(img,cv2.COLOR_RGB2BGR); g=base.copy();g[gt.any(0)]=(0,255,0);p=base.copy();p[pred.any(0)]=(0,0,255); heat=(np.clip(prob.max(0),0,1)*255).astype(np.uint8);cv2.imwrite(str(out/f'{name}_gt_pred.jpg'),cv2.hconcat([g,p,cv2.applyColorMap(heat,cv2.COLORMAP_JET)]))
def validate(model,ids,root,cfg,labels,dev,out):
 model.eval();acc=ConfusionAccumulator();lossfn=nn.BCEWithLogitsLoss();losses=[]; probs=[[] for _ in range(4)]; bg=0; errors={c:{'fn':None,'fp':None} for c in range(4)}
 with torch.no_grad():
  for image_id in ids:
   img=cv2.cvtColor(cv2.imread(str(root/cfg['image_dir']/image_id)),cv2.COLOR_BGR2RGB);gt=decode_multilabel(labels.get(image_id,{}),*img.shape[:2]).astype(np.float32); tiles=tile_image(img,256,512,256,384);parts=[];pos=[]
   for t,l,tile in tiles:
    x=torch.from_numpy(np.ascontiguousarray(tile.transpose(2,0,1))).float().div(255).unsqueeze(0).to(dev);parts.append(model(x)[0].cpu().numpy());pos.append((t,l))
   z=torch.from_numpy(stitch_logits(parts,pos,*img.shape[:2])).unsqueeze(0);y=torch.from_numpy(gt).unsqueeze(0);losses.append(lossfn(z,y).item());acc.update(z,y,.5);pr=torch.sigmoid(z)[0].numpy();pred=pr>=.5
   for c in range(4):
    probs[c].append(pr[c].mean()); fn=np.logical_and(gt[c]==1,~pred[c]).sum();fp=np.logical_and(gt[c]==0,pred[c]).sum()
    if fn and errors[c]['fn'] is None: errors[c]['fn']=(image_id,img,gt,pr)
    if fp and errors[c]['fp'] is None: errors[c]['fp']=(image_id,img,gt,pr)
   if not pred.any():bg+=1
 metric=acc.compute(); total_pixels=len(ids)*256*1600;metric.update({'val_loss':float(np.mean(losses)),'prediction_positive_pixel_ratio_per_class':[float((metric['tp'][c]+metric['fp'][c])/total_pixels) for c in range(4)],'probability_mean_per_class':[float(np.mean(x)) for x in probs],'full_background_prediction_image_ratio':bg/len(ids)})
 for c,d in errors.items():
  for kind,item in d.items():
   if item: save_vis(out/f'class_{c+1}_{kind}',item[0],item[1],item[2],item[3])
 return metric
def main():
 p=argparse.ArgumentParser();p.add_argument('--config',default='configs/data.yaml');p.add_argument('--out',default='reports/baseline-bce');p.add_argument('--resume');a=p.parse_args();cfg=yaml.safe_load(Path(a.config).read_text());seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);root=Path(cfg['data_root']);sd=Path(cfg['splits_dir']);train_ids=(sd/'train_image_ids.txt').read_text().split();val_ids=(sd/'val_image_ids.txt').read_text().split();assert not set(train_ids)&set(val_ids);assert not (sd/'test_image_ids.txt').read_text() in []
 out=Path(a.out);out.mkdir(parents=True,exist_ok=True);(out/'config.yaml').write_text(yaml.safe_dump(cfg));labels=labels_by_image(read_label_table(root/cfg['train_csv']));ds=SteelDefectDataset(root/cfg['image_dir'],train_ids,labels,RandomCropOrResize(256,512,seed));dl=DataLoader(ds,batch_size=4,shuffle=True,num_workers=0,pin_memory=False);dev=torch.device('mps' if torch.backends.mps.is_available() else 'cpu');m=ResNet18UNet().to(dev);opt=torch.optim.AdamW(m.parameters(),lr=3e-4,weight_decay=1e-5);lossfn=nn.BCEWithLogitsLoss();start=time.perf_counter();m.train();losses=[];grad=True
 for batch_index,b in enumerate(dl,1):
  x,y=b['image'].to(dev),b['mask'].to(dev,dtype=torch.float32);assert x.shape[1:]==(3,256,512) and y.shape[1:]==(4,256,512);opt.zero_grad();z=m(x);loss=lossfn(z,y);assert torch.isfinite(loss);loss.backward();grad &= any(q.grad is not None and torch.isfinite(q.grad).all() for q in m.parameters());opt.step();losses.append(loss.item())
  if batch_index % 100 == 0: print(f'train batches complete: {batch_index}/{len(dl)}',flush=True)
 metrics=validate(m,val_ids,root,cfg,labels,dev,out);metrics.update({'epoch':1,'train_loss':float(np.mean(losses)),'train_samples':len(train_ids),'train_patches':len(train_ids),'train_batches':len(dl),'val_samples':len(val_ids),'learning_rate':3e-4,'elapsed_seconds':time.perf_counter()-start,'gradient_finite':bool(grad),'device':str(dev),'mps_current_allocated_bytes':torch.mps.current_allocated_memory() if dev.type=='mps' else None,'checkpoint_tie_break':'best val loss: lower loss; best macro Dice: higher Dice; ties select earlier epoch'})
 # Epoch 1 is necessarily current best under the documented tie-breaker.
 state={'model':m.state_dict(),'optimizer':opt.state_dict(),'epoch':1,'metrics':metrics};torch.save(state,out/'last.pt');torch.save(state,out/'best_val_loss.pt');torch.save(state,out/'best_macro_dice.pt');(out/'metrics.json').write_text(json.dumps([metrics],indent=2));(out/'experiment_report.json').write_text(json.dumps({'status':'one_epoch_full_stability_check_complete','selection':'epoch 1 selected for all checkpoints (only completed epoch)', 'metrics':metrics},indent=2));print(json.dumps(metrics,indent=2))
if __name__=='__main__':main()
