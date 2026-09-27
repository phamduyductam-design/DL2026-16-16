import math
import torch
from src.training.losses import inpainting_loss,restore
from src.evaluation.metrics import tensor_metrics,metrics

def test_hand_calculated():
    image=torch.zeros(2,3,2,2)
    prediction=torch.tensor([.5,.25]).view(2,1,1,1).expand_as(image).clone().requires_grad_()
    mask=torch.tensor([[[[1.,0.],[0.,0.]]],[[[1.,1.],[1.,0.]]]])
    loss=inpainting_loss(prediction,image,mask)
    assert torch.isclose(loss,torch.tensor((.55+.275)/2))
    m=tensor_metrics(prediction,image,mask)
    torch.testing.assert_close(m['MAE_hole'],torch.tensor([.5,.25]))
    torch.testing.assert_close(m['MSE_hole'],torch.tensor([.25,.0625]))
    torch.testing.assert_close(m['PSNR_hole'],torch.tensor([10*math.log10(4),10*math.log10(16)]))
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    result=restore(prediction,image,mask)
    assert torch.equal(result*(1-mask),image*(1-mask))

def test_ssim_identity_and_zero_mse():
    image=torch.rand(1,3,16,16)
    m=metrics(image,image,torch.ones(1,1,16,16))
    assert m['SSIM_full'][0]==1
    assert math.isinf(m['PSNR_hole'][0])

def test_empty_regions_finite_loss():
    p=torch.rand(1,3,8,8,requires_grad=True)
    for m in [torch.zeros(1,1,8,8),torch.ones(1,1,8,8)]:
        loss=inpainting_loss(p,torch.zeros_like(p),m)
        assert torch.isfinite(loss)
