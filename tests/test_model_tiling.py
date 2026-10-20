import numpy as np, torch
from steel_defect_segmentation.model import ResNet18UNet
from steel_defect_segmentation.tiling import starts, stitch_logits

def test_unet_logits_and_backward():
    model=ResNet18UNet(); x=torch.rand(1,3,64,128); y=torch.zeros(1,4,64,128); logits=model(x)
    assert logits.shape == y.shape
    torch.nn.BCEWithLogitsLoss()(logits,y).backward()

def test_1600_tail_and_overlap_stitch():
    positions=[(0,left) for left in starts(1600,512,400)]
    assert positions[-1] == (0,1088)
    tiles=[np.ones((4,256,512),np.float32) for _ in positions]
    assert np.allclose(stitch_logits(tiles,positions,256,1600),1)
