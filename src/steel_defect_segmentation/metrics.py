"""Metrics with explicit empty-mask rules."""
from __future__ import annotations
import torch
class ConfusionAccumulator:
    """Accumulates whole-validation TP/FP/FN; never averages per-image Dice."""
    def __init__(self, classes=4): self.tp=torch.zeros(classes); self.fp=torch.zeros(classes); self.fn=torch.zeros(classes); self.empty=0; self.empty_fp=0
    def update(self, logits, target, threshold=.5):
        p=(torch.sigmoid(logits).cpu()>=threshold); t=target.cpu().bool(); self.tp+=(p&t).sum((0,2,3)); self.fp+=(p&~t).sum((0,2,3)); self.fn+=(~p&t).sum((0,2,3)); e=(t.sum((1,2,3))==0); self.empty+=int(e.sum()); self.empty_fp+=int((p.sum((1,2,3))>0)[e].sum())
    def compute(self):
        precision=self.tp/(self.tp+self.fp).clamp_min(1); recall=self.tp/(self.tp+self.fn).clamp_min(1); dice=2*self.tp/(2*self.tp+self.fp+self.fn).clamp_min(1); iou=self.tp/(self.tp+self.fp+self.fn).clamp_min(1)
        return {"dice_per_class":dice.tolist(),"iou_per_class":iou.tolist(),"precision_per_class":precision.tolist(),"recall_per_class":recall.tolist(),"macro_dice":float(dice.mean()),"empty_image_false_positive_rate":self.empty_fp/self.empty if self.empty else float("nan"),"tp":self.tp.tolist(),"fp":self.fp.tolist(),"fn":self.fn.tolist()}
def segmentation_metrics(logits: torch.Tensor, target: torch.Tensor, threshold=.5):
    """Empty target+prediction scores 1; empty target with positives scores 0. Positive-only reports exclude empty targets."""
    pred=(torch.sigmoid(logits)>=threshold).float(); target=target.float(); dims=(0,2,3); inter=(pred*target).sum(dims); ps=pred.sum(dims); ts=target.sum(dims); denom=ps+ts
    dice=torch.where(denom==0,torch.ones_like(denom),(2*inter)/denom); union=ps+ts-inter; iou=torch.where(union==0,torch.ones_like(union),inter/union)
    positive=ts>0; pos_dice=torch.where(positive,dice,torch.nan); empty_images=(target.sum((1,2,3))==0); false_positive=(pred.sum((1,2,3))>0); return {"dice_per_class":dice.cpu().tolist(),"iou_per_class":iou.cpu().tolist(),"positive_dice_per_class":pos_dice.cpu().tolist(),"empty_image_false_positive_rate":float(false_positive[empty_images].float().mean().item()) if empty_images.any() else float("nan")}
