import pickle
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.generate import generate_dataset,generate_split
from src.data.filters import load_filters
from src.dataset import MixDataset
from src.factory import make_kernels,make_model
from src.models.unet import ScoreUNet
from src.sde.rfft_noise import standard_rfft_noise_like


def test_data_convolution_cache_snr_and_worker_loading(tiny_config):
    path=generate_dataset(tiny_config)
    ds=MixDataset(path/'train.npz',load_time_domain=True)
    H=make_kernels(load_filters(path/'filters.npz'),ds.n_fft).numpy()
    images_fft=np.fft.rfft(ds.source_images,n=ds.n_fft,axis=-1,norm='ortho')
    np.testing.assert_allclose(images_fft,H[None]*ds.sources_fft,rtol=2e-5,atol=2e-6)
    clean=ds.source_images.astype(float).sum(1);noise=ds.mixtures-clean
    measured=10*np.log10(np.sum(clean**2,-1)/np.sum(noise**2,-1))
    np.testing.assert_allclose(measured,ds.snr_db,atol=2e-5)
    # No open ZipFile in the pickled dataset, and actual worker iteration.
    restored=pickle.loads(pickle.dumps(ds))
    batch=next(iter(DataLoader(restored,batch_size=2,num_workers=2)))
    assert batch[0].shape==(2,2,ds.num_frequency_bins)
    with pytest.raises(ValueError,match='normalization'):
        MixDataset(path/'train.npz',fft_norm='backward')
    with pytest.raises(FileExistsError):
        generate_dataset(tiny_config)


@pytest.mark.parametrize('n_fft',[30,31])
def test_noise_endpoint_variance(n_fft):
    torch.manual_seed(0)
    x=torch.zeros(5000,2,n_fft//2+1,dtype=torch.complex64)
    z=standard_rfft_noise_like(x,n_fft=n_fft)
    assert z[...,0].imag.abs().max()==0
    if n_fft%2==0: assert z[...,-1].imag.abs().max()==0
    else: assert z[...,-1].imag.abs().mean()>.3
    assert abs(z[...,0].real.square().mean().item()-1)<.05
    assert abs(z[...,1].real.square().mean().item()-.5)<.03
    assert abs(z[...,1].imag.square().mean().item()-.5)<.03


@pytest.mark.parametrize('n_fft',[30,31,2012])
def test_unet_backward_and_fixed_embedding(n_fft):
    torch.manual_seed(0)
    m=ScoreUNet(2,base_channels=8,time_dim=16,num_heads=2,n_fft=n_fft)
    B,F=2,n_fft//2+1
    x=torch.fft.rfft(torch.randn(B,2,n_fft),norm='ortho')
    y=x.sum(1); t=torch.tensor([.1,.8])
    before=m.time_embedding.frequencies.clone()
    out=m(x,y,t);loss=out.abs().square().mean();loss.backward()
    assert out.shape==x.shape and out[...,0].imag.abs().max()==0
    if n_fft%2==0: assert out[...,-1].imag.abs().max()==0
    else: assert out[...,-1].imag.abs().max()>0
    assert not list(m.time_embedding.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    opt=torch.optim.AdamW(m.parameters(),lr=1e-3);opt.step()
    assert torch.equal(before,m.time_embedding.frequencies)
    m.eval()
    with torch.no_grad():
        a=m(x,y,torch.full((B,),.1));b=m(x,y,torch.full((B,),.9));c=m(x,torch.zeros_like(y),torch.full((B,),.1))
    assert (a-b).abs().mean()>0 and (a-c).abs().mean()>0


def build_model_and_batch(cfg):
    data=generate_dataset(cfg)
    ds=MixDataset(data/'train.npz')
    kernels=make_kernels(load_filters(data/'filters.npz'),ds.n_fft)
    m=make_model(cfg,n_fft=ds.n_fft,kernels=kernels)
    x=torch.stack([ds[i][0] for i in range(2)]);y=torch.stack([ds[i][1] for i in range(2)])
    return m,x,y


def test_mismatch_input_and_target_identity(tiny_config):
    m,x,y=build_model_and_batch(tiny_config)
    t=x.real.new_full((2,),m.sde.T)
    epsilon=standard_rfft_noise_like(x,n_fft=m.n_fft)
    mask=torch.ones(2,dtype=torch.bool)
    terms=m.loss_terms(x,y,t,mismatch_mask=mask,epsilon=epsilon)
    direct=m.sde.apply_std_inv(terms['std'],terms['xt'].reshape(2,-1)-terms['mean'])
    torch.testing.assert_close(direct,terms['target_noise'],rtol=3e-5,atol=3e-6)
    yrep=y[:,None].expand_as(x).reshape(2,-1)
    expected=yrep+m.sde.apply_std(terms['std'],epsilon.reshape(2,-1))
    torch.testing.assert_close(terms['xt'].reshape(2,-1),expected)
    # The historical bug is an explicit, separate experiment.
    m.mismatch_input='forward_legacy'
    old=m.loss_terms(x,y,t,mismatch_mask=mask,epsilon=epsilon)
    assert not torch.allclose(old['xt'],terms['xt'])


def test_lightning_loss_ema_and_no_dense(tiny_config):
    m,x,y=build_model_and_batch(tiny_config)
    m.train()
    assert not m.ema_score_model.training
    assert not any(p.requires_grad for p in m.ema_score_model.parameters())
    opt=m.configure_optimizers()
    old={n:p.clone() for n,p in m.ema_score_model.named_parameters()}
    loss=m.compute_score_loss(x,y);loss.backward();opt.step()
    assert int(m.ema_updates)==1
    for name,p in m.score_model.named_parameters():
        expected=m.ema_decay*old[name]+(1-m.ema_decay)*p
        torch.testing.assert_close(dict(m.ema_score_model.named_parameters())[name],expected)
    assert m.sde._dense_cache=={}
    a=m.validation_step((x,y),0);b=m.validation_step((x,y),0)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
