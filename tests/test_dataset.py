import cv2, numpy as np
from torch.utils.data import DataLoader
from steel_defect_segmentation.dataset import RandomCropOrResize, SteelDefectDataset

def test_synthetic_dataset_and_loader(tmp_path):
    bgr=np.zeros((12,20,3),np.uint8); bgr[2:5,3:8]=(0,0,255); cv2.imwrite(str(tmp_path/"x.jpg"),bgr)
    ds=SteelDefectDataset(tmp_path,["x.jpg"],{"x.jpg":{1:"39 3",2:None,3:"100 2",4:None}},RandomCropOrResize(8,10,seed=1))
    sample=ds[0]; assert sample["image"].shape == (3,8,10); assert sample["mask"].shape == (4,8,10)
    batch=next(iter(DataLoader(ds,batch_size=1))); assert batch["mask"].shape == (1,4,8,10)
