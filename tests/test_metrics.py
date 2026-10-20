import torch, pytest
from steel_defect_segmentation.metrics import ConfusionAccumulator

def test_accumulator_uses_global_tp_fp_fn_not_image_mean():
    # One true-positive pixel and one false-positive pixel across two images.
    logits=torch.tensor([[[[10.,-10.]]],[[[10.,-10.]]]])
    target=torch.tensor([[[[1.,0.]]],[[[0.,0.]]]])
    a=ConfusionAccumulator(1);a.update(logits,target);r=a.compute()
    assert r['tp']==[1.0] and r['fp']==[1.0] and r['fn']==[0.0]
    assert r['dice_per_class'][0] == pytest.approx(2/3) and r['empty_image_false_positive_rate']==1.0
