import pytest
import torch
from src.models import make_model
from src.utils.seed import seed_everything

@pytest.mark.parametrize('name',['autoencoder','unet'])
def test_forward_backward_and_skip_shapes(name):
    torch.set_num_threads(2)
    seed_everything(42)
    model=make_model(name)
    x=torch.rand(1,4,128,128)
    features=model.encoder(x)
    assert [tuple(t.shape) for t in features]==[(1,32,128,128),(1,64,64,64),(1,128,32,32),(1,256,16,16)]
    y=model(x)
    assert y.shape==(1,3,128,128)
    assert 0<=y.min()<=y.max()<=1
    y.mean().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    seed_everything(42)
    other=make_model(name)
    assert all(torch.equal(a,b) for a,b in zip(model.parameters(),other.parameters()))
