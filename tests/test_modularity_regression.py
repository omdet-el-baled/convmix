"""Preserve the previous numerical implementations while changing construction."""
from __future__ import annotations
import copy
from functools import partial
import importlib
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from src.models.unet import ScoreUNet, BasicScoreUNet
from src.models.attention import IdentityAttention
from src.models.embeddings import GaussianFourierProjection
from src.models.features import SpectralFeatures
from src.pl_model import DCSModel
from src.losses import CovarianceWeightedDSM
from src.sampling.pipeline import PredictorCorrectorSampler
from src.sampling.predictors import (EulerProbabilityFlow, HeunProbabilityFlow,
                                    RK4ProbabilityFlow, EulerMaruyama)
from src.sampling.correctors import LangevinCorrector, NoCorrector
from src.sampling.samplers import sample, SamplingConfig
from src.sde.rfft_noise import standard_rfft_noise_like
from tests.reference.baseline_unet import ScoreUNet as PreviousScoreUNet
from tests.reference.baseline_unet import BasicScoreUNet as PreviousBasicScoreUNet
from tests.reference.baseline_pl_model import DCSModel as PreviousDCSModel
from tests.test_sampling import GaussianScore
from tests.test_sde import make_sde


@pytest.mark.parametrize('sources,n_fft,basic', [(2,30,False),(3,31,False),(2,62,True)])
def test_default_unet_weights_output_gradients_identical(sources,n_fft,basic):
    args = dict(num_sources=sources, base_channels=8, time_dim=16,
                num_heads=2, n_fft=n_fft, dropout=0.05)
    old_cls,new_cls = (PreviousBasicScoreUNet,BasicScoreUNet) if basic else (PreviousScoreUNet,ScoreUNet)
    torch.manual_seed(101); old = old_cls(**args).train()
    torch.manual_seed(101); new = new_cls(**args).train()
    assert list(old.state_dict()) == list(new.state_dict())
    for key in old.state_dict():
        torch.testing.assert_close(old.state_dict()[key],new.state_dict()[key],rtol=0,atol=0)
    x=torch.fft.rfft(torch.randn(2,sources,n_fft),norm='ortho'); y=x.sum(1)
    t=torch.tensor([.1,.8])
    torch.manual_seed(55);a=old(x,y,t)
    torch.manual_seed(55);b=new(x,y,t)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.abs().square().mean().backward();b.abs().square().mean().backward()
    for (ka,pa),(kb,pb) in zip(old.named_parameters(),new.named_parameters()):
        assert ka==kb
        torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)


@pytest.mark.parametrize('solver,predictor',[
    ('euler',EulerProbabilityFlow),('heun',HeunProbabilityFlow),('rk4',RK4ProbabilityFlow),
    ('em',EulerMaruyama),('pc',EulerMaruyama)])
def test_composed_sampler_equals_previous_solver(solver,predictor):
    sde=make_sde(n_fft=8);model=GaussianScore(sde).eval()
    y=torch.zeros(2,5,dtype=torch.complex128)
    cfg=SamplingConfig(solver=solver,num_steps=9,seed=141,corrector_step_size=.001)
    old=sample(model,sde,y,K=2,n_fft=8,t_eps=.05,config=cfg,example_ids=[2,7])
    sampler=PredictorCorrectorSampler(predictor=predictor,
        corrector=LangevinCorrector(step_size=.001) if solver=='pc' else NoCorrector(),
        num_steps=9,seed=141,solver=solver)
    new=sampler(model,sde,y,K=2,n_fft=8,t_eps=.05,example_ids=[2,7])
    torch.testing.assert_close(old.x,new.x,rtol=0,atol=0)
    torch.testing.assert_close(old.initial_x,new.initial_x,rtol=0,atol=0)
    assert old.nfe==new.nfe and old.score_times==new.score_times
    assert old.final_time==new.final_time==.05
    assert old.corrector_clipped_steps==new.corrector_clipped_steps


@pytest.mark.parametrize('geometry', ['diffusion','identity'])
def test_adaptive_pc_exact_equivalence(geometry):
    sde=make_sde(n_fft=8);model=GaussianScore(sde).eval()
    y=torch.zeros(1,5,dtype=torch.complex128)
    cfg=SamplingConfig(solver='pc',num_steps=7,seed=99,corrector_steps=2,
                       corrector_snr=.08,corrector_geometry=geometry,corrector_max_step=.2)
    old=sample(model,sde,y,K=2,n_fft=8,t_eps=.05,config=cfg)
    new=PredictorCorrectorSampler(predictor=EulerMaruyama,
        corrector=LangevinCorrector(num_steps=2,snr=.08,geometry=geometry,max_step=.2),
        num_steps=7,seed=99,solver='pc')(model,sde,y,K=2,n_fft=8,t_eps=.05)
    torch.testing.assert_close(old.x,new.x,rtol=0,atol=0)
    assert old.corrector_clipped_steps==new.corrector_clipped_steps


@pytest.mark.parametrize('input_mode', ['terminal','forward_legacy'])
def test_extracted_loss_and_gradient_identical(input_mode):
    from src.models.examples import TinyScoreNet
    sde=make_sde(n_fft=8)
    torch.manual_seed(6); net=TinyScoreNet(2,8,channels=8,time_dim=16).double()
    old=PreviousDCSModel(copy.deepcopy(net),sde,n_fft=8,mismatch_input=input_mode)
    new=DCSModel(copy.deepcopy(net),sde,n_fft=8,mismatch_input=input_mode)
    x=torch.fft.rfft(torch.randn(3,2,8,dtype=torch.float64),norm='ortho');y=x.sum(1)
    eps=standard_rfft_noise_like(x,n_fft=8);t=torch.tensor([.08,.4,.8],dtype=torch.float64)
    mask=torch.tensor([False,True,True])
    a=old.loss_terms(x,y,t,epsilon=eps,mismatch_mask=mask)
    b=new.loss_terms(x,y,t,epsilon=eps,mismatch_mask=mask)
    for key in a:
        torch.testing.assert_close(a[key],b[key],rtol=0,atol=0)
    a['loss'].backward();b['loss'].backward()
    for pa,pb in zip(old.score_model.parameters(),new.score_model.parameters()):
        torch.testing.assert_close(pa.grad,pb.grad,rtol=0,atol=0)


def test_variable_depth_and_alternate_components():
    model=ScoreUNet(2,base_channels=8,time_dim=16,num_heads=2,n_fft=30,
        channel_multipliers=(1,2,3,4),encoder_dilations=((1,),(1,2),(1,2,4),(1,)),
        decoder_dilations=((1,),(1,),(1,2),(1,)),bottleneck_dilations=(1,2),attention_after=1,
        attention=IdentityAttention,time_embedding=GaussianFourierProjection,
        features=partial(SpectralFeatures,log_magnitude=False))
    x=torch.fft.rfft(torch.randn(2,2,30));y=x.sum(1);t=torch.rand(2)
    z=model(x,y,t);z.abs().square().mean().backward()
    assert z.shape==x.shape and torch.isfinite(z).all()
    assert isinstance(model.attention,IdentityAttention)
    assert not list(model.time_embedding.parameters())
    assert model.input_conv.in_channels==6
    assert len(model.encoder_names)==4


def test_trainable_time_embedding_is_rejected():
    class Learned(nn.Module):
        def __init__(self,embedding_size):
            super().__init__();self.weight=nn.Parameter(torch.ones(embedding_size))
        def forward(self,t):return t[:,None]*self.weight
    with pytest.raises(ValueError,match='fixed'):
        ScoreUNet(2,base_channels=8,time_dim=16,num_heads=2,n_fft=30,time_embedding=Learned)


def test_optimizer_factory_does_not_require_lightning_edit():
    from src.models.examples import TinyScoreNet
    m=DCSModel(TinyScoreNet(2,8,8,16),make_sde(n_fft=8),n_fft=8,
               optimizer=partial(torch.optim.SGD,lr=.03,momentum=.9))
    opt=m.configure_optimizers()
    assert isinstance(opt,torch.optim.SGD) and opt.param_groups[0]['lr']==.03


def test_all_config_target_imports_resolve():
    # Imports/type discovery only; this is not a Hydra composition test.
    root=Path(__file__).resolve().parents[1]/'config'
    targets=set()
    def visit(v):
        if isinstance(v,dict):
            if '_target_' in v:targets.add(v['_target_'])
            for child in v.values():visit(child)
        elif isinstance(v,list):
            for child in v:visit(child)
    for path in root.rglob('*.yaml'):visit(yaml.safe_load(path.read_text()))
    for target in targets:
        module,name=target.rsplit('.',1)
        assert callable(getattr(importlib.import_module(module),name)),target
