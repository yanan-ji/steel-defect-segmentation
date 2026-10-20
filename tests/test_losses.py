import pytest
import torch

from steel_defect_segmentation.losses import BCEDiceLoss, SoftDiceLoss


def test_multilabel_dice_gradient_and_combined_loss():
    logits = torch.zeros((2, 4, 5, 7), requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 1:3, 2:4] = 1
    target[1, 2, 3, 5] = 1
    bce, dice, total = BCEDiceLoss(1.0).components(logits, target)
    assert total.item() == pytest.approx((bce + dice).item())
    assert 0 < dice.item() < 1
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert logits.grad.abs().sum() > 0


def test_absent_class_excluded_and_empty_batch_zero_dice():
    target = torch.zeros((1, 4, 2, 2))
    target[:, 0, 0, 0] = 1
    logits = torch.zeros_like(target, requires_grad=True)
    loss = SoftDiceLoss()(logits, target)
    # Only class 1 participates: intersection=.5, denominator=2+1.
    assert loss.item() == pytest.approx(1 - (1 + 1e-6) / (3 + 1e-6))
    loss.backward()
    assert logits.grad[:, 1:].abs().sum().item() == 0
    empty_logits = torch.zeros_like(target, requires_grad=True)
    empty = SoftDiceLoss()(empty_logits, torch.zeros_like(target))
    assert empty.item() == 0
    empty.backward()
    assert empty_logits.grad is not None and empty_logits.grad.abs().sum().item() == 0


def test_extreme_logits_finite_and_shape_rejected():
    logits = torch.tensor([1000.0, -1000.0] * 8).reshape(1, 4, 2, 2).requires_grad_()
    target = torch.zeros_like(logits)
    target[:, 0, 0, 0] = 1
    loss = BCEDiceLoss()(logits, target)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    with pytest.raises(ValueError):
        SoftDiceLoss()(torch.zeros(1, 3, 2, 2), torch.zeros(1, 3, 2, 2))


def test_bce_and_dice_have_separate_valid_gradients():
    logits = torch.zeros((2, 4, 8, 8), requires_grad=True)
    target = torch.zeros_like(logits)
    target[0, 0, 2:5, 1:4] = 1
    target[1, 2, 4:7, 4:7] = 1
    bce, dice, total = BCEDiceLoss().components(logits, target)
    bce_grad = torch.autograd.grad(bce, logits, retain_graph=True)[0]
    dice_grad = torch.autograd.grad(dice, logits, retain_graph=True)[0]
    total_grad = torch.autograd.grad(total, logits)[0]
    assert torch.isfinite(bce_grad).all() and torch.isfinite(dice_grad).all()
    assert bce_grad.abs().sum() > 0 and dice_grad.abs().sum() > 0
    assert torch.allclose(total_grad, bce_grad + dice_grad)
    assert dice_grad[:, 1].abs().sum() == 0  # absent class 2: BCE remains active
    assert bce_grad[:, 1].abs().sum() > 0


def test_empty_batch_keeps_only_bce_gradient():
    logits = torch.zeros((1, 4, 3, 3), requires_grad=True)
    target = torch.zeros_like(logits)
    bce, dice, total = BCEDiceLoss().components(logits, target)
    bce_grad = torch.autograd.grad(bce, logits, retain_graph=True)[0]
    total_grad = torch.autograd.grad(total, logits)[0]
    assert dice.item() == 0
    assert torch.allclose(total_grad, bce_grad)
    assert total_grad.abs().sum() > 0
