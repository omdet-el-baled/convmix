import numpy as np
import pytest
import torch
from torch import nn

from src.sampling.samplers import sample,SamplingConfig
from src.sde.rfft_noise import standard_rfft_noise_like
from tests.test_sde import make_sde


class GaussianScore(nn.Module):
    """Exact marginal score for Gaussian x0 in the modal basis, not a DSM target."""
    def __init__(self,sde):
        super().__init__();self.sde=sde

    def variance(self,t):
        return .7*self.sde.get_Ft(t).abs().square()+self.sde._var(t)

    def forward(self,x,y,t,use_ema=True):
        B=x.shape[0]
        modal=self.sde.apply_V_inv(x.reshape(B,-1))
        # V^-H score in modal coordinates, unlike V^-1 acting on states.
        value=self.sde.apply_V_inv_H(-modal/self.variance(t))
        return value.reshape_as(x)


@pytest.mark.parametrize('solver,nfe_factor',[('euler',1),('heun',2),('rk4',4),('em',1),('pc',2)])
def test_endpoint_nfe_finite(solver,nfe_factor):
    torch.manual_seed(0)
    s=make_sde(n_fft=8);net=GaussianScore(s).eval()
    x=standard_rfft_noise_like(torch.zeros(2,2,5,dtype=torch.complex128),n_fft=8)
    y=torch.zeros(2,5,dtype=torch.complex128)
    result=sample(net,s,y,K=2,n_fft=8,t_eps=.05,x_T=x,
                  config=SamplingConfig(solver=solver,num_steps=20,corrector_step_size=.001))
    assert result.final_time==.05 and len(result.times)==21
    assert min(result.score_times)>=.05 and max(result.score_times)<=1
    assert result.nfe==20*nfe_factor
    assert result.x[...,0].imag.abs().max()==0 and result.x[...,-1].imag.abs().max()==0
    assert torch.isfinite(result.x).all()


def test_ode_numerical_convergence_against_exact_gaussian_flow():
    torch.manual_seed(1)
    s=make_sde(n_fft=8);net=GaussianScore(s).eval();eps=.1
    z=standard_rfft_noise_like(torch.zeros(3,2,5,dtype=torch.complex128),n_fft=8).reshape(3,-1)
    vT=net.variance(1.);ve=net.variance(eps)
    cT=z*vT.sqrt();xT=s.apply_V(cT).reshape(3,2,5)
    rotation=torch.exp(1j*(s._phi(eps).imag-s._phi(1.).imag))
    exact=s.apply_V(cT*(ve/vT).sqrt()*rotation).reshape_as(xT)
    y=torch.zeros(3,5,dtype=torch.complex128)
    errors=[]
    for N in [25,50,100,200]:
        r=sample(net,s,y,K=2,n_fft=8,t_eps=eps,x_T=xT,config=SamplingConfig(num_steps=N))
        errors.append(float(torch.linalg.vector_norm(r.x-exact)))
    assert all(b<a for a,b in zip(errors,errors[1:])),errors
    assert errors[-1]<errors[0]/3


def test_pc_zero_corrector_equals_em_and_shared_xT():
    s=make_sde(n_fft=8);net=GaussianScore(s).eval()
    y=torch.zeros(2,5,dtype=torch.complex128)
    common=dict(num_steps=10,seed=7,corrector_steps=0)
    em=sample(net,s,y,K=2,n_fft=8,t_eps=.1,example_ids=[5,9],config=SamplingConfig(solver='em',**common))
    pc=sample(net,s,y,K=2,n_fft=8,t_eps=.1,example_ids=[5,9],config=SamplingConfig(solver='pc',**common))
    torch.testing.assert_close(em.x,pc.x,rtol=0,atol=0)
    a=sample(net,s,y,K=2,n_fft=8,t_eps=.1,example_ids=[5,9],config=SamplingConfig(num_steps=5,seed=7))
    b=sample(net,s,y,K=2,n_fft=8,t_eps=.1,example_ids=[5,9],config=SamplingConfig(num_steps=20,seed=7))
    torch.testing.assert_close(a.initial_x,b.initial_x,rtol=0,atol=0)
    single=sample(net,s,y[:1],K=2,n_fft=8,t_eps=.1,example_ids=[9],config=SamplingConfig(num_steps=5,seed=7))
    torch.testing.assert_close(a.initial_x[1],single.initial_x[0],rtol=0,atol=0)


class StationarySDE:
    T=1.;ndim=10
    def send_to(self,device):return self
    def apply_drift(self,x,t):return -.5*x
    def apply_BBH(self,x,t):return x
    def apply_B(self,z,t):return z


class StationaryScore(nn.Module):
    def forward(self,x,y,t,use_ema=True):return -x


@pytest.mark.slow
def test_pc_noise_score_scaling_preserves_unit_gaussian_approximately():
    torch.manual_seed(45)
    N=2000
    x=standard_rfft_noise_like(torch.zeros(N,2,5,dtype=torch.complex64),n_fft=8)
    model=StationaryScore().eval()
    noise=lambda z:standard_rfft_noise_like(z,n_fft=8)
    result=sample(model,StationarySDE(),torch.zeros(N,5,dtype=torch.complex64),K=2,n_fft=8,
        t_eps=.1,x_T=x,noise_source=noise,
        config=SamplingConfig(solver='pc',num_steps=40,corrector_step_size=.002,corrector_geometry='identity'))
    # Complex interior unit variance has real/imag variance .5. Endpoints have
    # real variance 1. Tests the stochastic convention, not final SI-SDR.
    assert abs(result.x[...,1:4].abs().square().mean().item()-1)<.06
    assert abs(result.x[...,0].real.square().mean().item()-1)<.09
