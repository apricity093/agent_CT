import numpy as np
import pytest
import torch
from inv_framework.mri.models.varnet import NormUnet, VarNet, VarNetBlock
from inv_framework.mri.models import complex_math as cm
from inv_framework.mri.varnet_inference import load_checkpoint
from inv_framework.mri.api import run_mri_task


def test_complex_normalization_padding_and_encoding():
    torch.manual_seed(3)
    z = torch.randn(1, 1, 19, 27, 2)
    net = NormUnet(2, 2).eval()
    channels = net.complex_to_chan_dim(z)
    torch.testing.assert_close(net.chan_complex_to_last_dim(channels), z)
    norm, mean, std = net.norm(channels)
    torch.testing.assert_close(net.unnorm(norm,mean,std),channels)
    padded, spec = net.pad(norm)
    torch.testing.assert_close(net.unpad(padded,*spec), norm)
    assert net(z).shape == z.shape
    torch.testing.assert_close(cm.ifft2c(cm.fft2c(z)), z)
    maps = torch.randn(1,3,19,27,2)
    maps /= cm.rss_complex(maps,1)[:,None,:,:,None]
    block = VarNetBlock(net)
    torch.testing.assert_close(block.sens_reduce(block.sens_expand(z,maps),maps),z,atol=2e-6,rtol=2e-6)


def test_checkpoint_strict_and_learned_maps(tmp_path):
    net = VarNet(num_cascades=1,chans=2,pools=2,sens_chans=2,sens_pools=2).eval()
    path = tmp_path/'structure_only.pt'
    torch.save({'state_dict': {'varnet.'+k:v for k,v in net.state_dict().items()}}, path)
    load_checkpoint(net,path,'cpu')
    data = torch.randn(1,2,20,28,2)
    mask = torch.ones(1,1,1,28,1,dtype=torch.bool)
    maps = net.sens_net(data,mask,8)
    torch.testing.assert_close(cm.rss_complex(maps,1),torch.ones(1,20,28))
    assert torch.isfinite(net(data,mask,8)).all()
    state = net.state_dict();state.pop(next(iter(state)));torch.save(state,path)
    with pytest.raises(RuntimeError): load_checkpoint(net,path,'cpu')


def test_missing_checkpoint_is_explicit():
    shape = (32,48)
    r = run_mri_task(dict(algorithm='varnet',inputs=dict(kspace=np.ones((1,2)+shape,complex),sampling_mask=np.ones((1,2)+shape)),
        model=dict(axes=['batch','coil','y','x'],spatial_shape=list(shape),sampling='cartesian')))
    assert r.status=='failed' and r.error['code']=='missing_checkpoint'
    assert not r.outputs
