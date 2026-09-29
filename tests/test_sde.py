import math
import numpy as np
import pytest
import torch
from scipy.integrate import solve_ivp

from src.sde.freqsde import ConvMixSDE


def make_sde(K=2, n_fft=16, mode='mixing', alpha=3., beta=2., T=1., dtype=torch.complex128):
    re = torch.float64 if dtype==torch.complex128 else torch.float32
    h = torch.stack([torch.tensor([1.+.2*k,.12,.02],dtype=re) for k in range(K)])
    H = torch.fft.rfft(h, n=n_fft)
    sde=ConvMixSDE(K*H.shape[-1],alpha,beta,.2,5,1024,T,drift_mode=mode)
    sde.get_mix_mat(H,H.shape[-1]); sde.set_rfft(n_fft)
    return sde


@pytest.mark.parametrize('K',[2,3,5])
def test_eigenbasis_and_adjoint(K):
    sde=make_sde(K)
    A,V,Vi=sde.A,sde.eigvecs,sde.eigvecs_inv
    torch.testing.assert_close(A@V,V@torch.diag(sde.eigvals),rtol=1e-11,atol=1e-11)
    x=torch.randn(3,sde.ndim,dtype=torch.complex128)
    for name,mat in [('apply_V',V),('apply_V_inv',Vi),('apply_VH',V.mH),('apply_V_inv_H',Vi.mH)]:
        actual=getattr(sde,name)(x)
        torch.testing.assert_close(actual,(mat@x[...,None]).squeeze(-1),rtol=1e-11,atol=1e-11)


def test_get_Gt_is_derivative_and_mean_matches_integrated_drift():
    s=make_sde(); t=.31; h=1e-6
    derivative=(s.get_Ft(t+h)-s.get_Ft(t-h))/(2*h)
    torch.testing.assert_close(derivative,s.get_Ft(t)*s.get_Gt(t),rtol=1e-8,atol=1e-8)
    x=torch.randn(2,s.ndim,dtype=torch.complex128)
    dmean=(s._mean(x,t+h)-s._mean(x,t-h))/(2*h)
    torch.testing.assert_close(dmean,s.apply_drift(s._mean(x,t),t),rtol=1e-7,atol=1e-8)
    fast=make_sde(alpha=50,beta=50)
    torch.testing.assert_close(fast._mean(x,1),(fast.A@x[...,None]).squeeze(-1),rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('mode',['mixing','none','instantaneous'])
def test_covariance_vs_independent_block_lyapunov(mode):
    s=make_sde(n_fft=8,mode=mode)
    F,K=s.N_fft,s.K
    V,Vi=s.V.numpy(),s.V_inv.numpy(); C=V@V.conj().transpose(0,2,1)
    P=np.ones((K,K))/K; Q=np.eye(K)-P
    loglam=np.log(s.kernels.numpy().sum(0))
    rate=2*math.log(s.sigma_max/s.sigma_min)/s.T
    def rhs(t,q):
        q=q.reshape(F,K,K)
        if mode=='none': drift=np.zeros((F,K,K),complex)
        elif mode=='instantaneous': drift=np.broadcast_to(-s.beta*Q,(F,K,K))
        else:
            rates=np.full((F,K),-s.beta,complex)
            rates[:,0]=s.alpha*np.exp(-s.alpha*t)*loglam
            drift=(V*rates[:,None,:])@Vi
        g2=s.sigma_min**2*rate*np.exp(rate*t)
        return (drift@q+q@drift.conj().transpose(0,2,1)+g2*C).ravel()
    for t in [.005,.4,1.]:
        expected=solve_ivp(rhs,(0,t),np.zeros(F*K*K,complex),rtol=1e-10,atol=1e-12).y[:,-1].reshape(F,K,K)
        S=s.factor_blocks(t)[0]; actual=(S@S.mH).numpy()
        np.testing.assert_allclose(actual,expected,rtol=3e-5,atol=1e-8)
        # Inverse and adjoint actions, including complex phase.
        x=torch.randn(2,s.ndim,dtype=torch.complex128)
        std=s.fast_factor(torch.tensor([t,t],dtype=torch.float64))
        torch.testing.assert_close(s.apply_std_inv(std,s.apply_std(std,x)),x,rtol=1e-10,atol=1e-10)
        z=torch.randn_like(x)
        left=(s.apply_std(std,x).conj()*z).sum()
        right=(x.conj()*s.apply_std_h(std,z)).sum()
        torch.testing.assert_close(left,right,rtol=1e-11,atol=1e-11)


def test_covariance_zero_and_T_scaling():
    s=make_sde(T=2.,mode='none')
    assert s._std(0).abs().max()==0
    q=s._var(2.)
    torch.testing.assert_close(q,torch.full_like(q,s.sigma_max**2-s.sigma_min**2))


def test_no_dense_allocation_for_normal_path():
    s=make_sde(); x=torch.randn(2,s.ndim,dtype=torch.complex128)
    t=torch.tensor([.2,.3],dtype=torch.float64)
    s._mean(x,t); s.apply_drift(x,t); s.apply_std(s.fast_factor(t),x); s.apply_BBH(x,t)
    assert s._dense_cache=={}


def test_rejection_of_invalid_rfft_log_endpoints():
    s=ConvMixSDE(6,3,3,.2,5,32,1)
    s.get_mix_mat(torch.full((2,3),-1+0j,dtype=torch.complex128),3)
    with pytest.raises(ValueError,match='positive'):
        s.set_rfft(4)


def test_reverse_uses_matrix_covariance_not_elementwise_square():
    s=make_sde(); x=torch.randn(2,s.ndim,dtype=torch.complex128);t=torch.tensor([.1,.2],dtype=torch.float64)
    score=lambda x,y,t: -x
    reverse=s.reverse(score,True)
    actual,_=reverse.sde(x,None,t)
    expected=s.apply_drift(x,t)+.5*s.apply_BBH(x,t)
    torch.testing.assert_close(actual,expected)
