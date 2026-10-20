import numpy as np
import pytest
from steel_defect_segmentation.rle import decode_rle, encode_rle
from steel_defect_segmentation.labels import decode_multilabel

def test_rle_round_trip_fortran_order():
    mask=np.zeros((4,5),np.uint8); mask[0,1]=mask[3,1]=mask[2,4]=1
    assert np.array_equal(decode_rle(encode_rle(mask),4,5),mask)

def test_empty_rle(): assert not decode_rle("",3,7).any()

def test_four_class_multilabel_shape_and_overlap():
    rows={1:"1 2",2:"2 2",3:None,4:"12 1"}; masks=decode_multilabel(rows,3,4)
    assert masks.shape == (4,3,4) and masks[0].sum()==2 and masks[1].sum()==2

def test_invalid_rle_rejected():
    with pytest.raises(ValueError): decode_rle("0 1",2,2)
