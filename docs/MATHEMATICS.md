# Conventions, equations, and limitations

## Dimensions and Fourier normalization

The problem is one observed sensor and K>=2 latent sources:

    y[n] = sum_k (h_k * s_k)[n] + measurement_noise[n].

Linear convolution is retained in full. n_fft >= source_length + max_filter_length - 1.
Signals use the configured forward FFT normalization; kernels use the unscaled
forward FFT. With ortho signals:

    rFFT_ortho(h*s) = rFFT_backward(h) * rFFT_ortho(s).

State tensors are [B,K,F], F=n_fft//2+1. The ConvMixSDE compatibility API uses
[B,KF] source-major flattening. Eigenspace blocks use [B,F,K]. The first modal
channel is the nonzero mode, followed by K-1 null channels, each containing F bins.

## Mean and diffusion

For each frequency, A=1 h^T (transpose, not conjugate transpose), lambda=sum H_k.
The original right eigenvectors are v0=1 and vj=e_j-(H_j/H1)e_1.

    phi_lambda(t) = log(lambda)*(1-exp(-alpha*t))
    phi_null(t)   = -beta*t
    get_Ft(t)    = exp(phi(t))
    get_Gt(t)    = phi'(t)
    mu(t)        = V diag(exp(phi(t))) V^-1 x0

The nonzero derivative is alpha*log(lambda)*exp(-alpha*t). A fixed V makes the
time-dependent drift matrices commute. Positive alpha/beta give asymptotic mean
convergence to A; finite T does not make that equality exact.

    sigma(t) = sigma_min * (sigma_max/sigma_min)^(t/T)
    g²(t) = (2 log(sigma_max/sigma_min)/T) * sigma(t)²
    B(t) = g(t) V
    q_i(t) = integral_0^t exp(2 Re(phi_i(t)-phi_i(s))) g²(s) ds
    s_i(t) = sqrt(q_i(t)) exp(i Im(phi_i(t)))
    S(t) = V diag(s_i(t)); Sigma=S S^H

The covariance-factor phase is a chosen factor gauge. The implementation keeps
our final shared gauge; q alone does not determine that phase. Rescaling V would
change B and is not a harmless implementation optimization here.

The original basis is singular at H1=0 and the log construction fails when
lambda=0. Those cases are rejected rather than regularized invisibly. Physical
rFFT use also requires real endpoint kernels and positive lambda at DC/Nyquist
for this particular principal-log interpolation. No theorem covering arbitrary
filters is claimed.

## Complex score and stochastic factors

Interior complex noise z has E|z|²=1, Re/Im variance 1/2. DC and even Nyquist
use real N(0,1). These endpoints are not proper complex random variables.

The implemented score convention is:

- Interior Gaussian: score = -Sigma^-1 (x-mu), i.e. conjugate-Wirtinger score.
- Real endpoint Gaussian: score = -Sigma^-1 (x-mu), the ordinary real score.

With these conventions the same algebraic DSM residual applies:

    r = S^H score + epsilon.

The complex interior's real-coordinate Brownian covariance and real gradient
have compensating factors of 1/2 and 2. The reverse and PF fields are consequently:

    reverse SDE: F x - B B^H score
    PF ODE:      F x - (1/2) B B^H score.

For decreasing t, use negative dt for drift and sqrt(-dt) for Brownian noise.
Tests include an analytic Gaussian marginal score, not just a comparison between
two calls to the same drift formula.

Langevin with fixed preconditioner M=B B^H uses

    x <- x + eta M score + sqrt(2 eta) B z.

The same mixed real/complex convention makes this correct at infinitesimal step
size for the corresponding Gaussian score. Adaptive state-dependent eta,
clipping, finite step size, and approximate neural scores introduce bias; PC is
not described as an exact invariant sampler. Noise and source dimensions are
never treated as scalar diffusion when the diffusion is matrix-valued.

## Mismatch branch

Ordinary perturbation:

    x_t = mu_t(x0) + S_t epsilon
    r_t = S_t^H score(x_t,y,t) + epsilon.

Intended recentered terminal augmentation:

    x_T_tilde = y_rep + S_T epsilon
    r_T = S_T^H score(x_T_tilde,y,T)
          + epsilon + S_T^-1(y_rep-mu_T(x0)).

The second noise target is exactly S_T^-1(x_T_tilde-mu_T(x0)). This identity is
covered by a test. Evaluating the network instead at mu_T+S_T epsilon is the
previous input-centering variant, available as forward_legacy, not silently mixed
with the corrected variant.

The recentered augmentation still changes the distribution of score-training
pairs. It is an empirical terminal correction, not ordinary DSM for the same
marginal law and not a proof of correct posterior inference.

## What diagnostics do NOT establish

The per-perturbation kernel score is not the unknown conditional marginal score.
For squared-error prediction, the expected squared target error contains both
model error relative to the conditional expectation and irreducible conditional
target variance. A DSM value near 1 at small t, by itself, does not prove an
untrained/incorrect score; dividing it by a small covariance scale does not remove
that irreducible term.

More integration steps can reduce numerical error relative to a given learned
field without improving SI-SDR. The forward mean converging to the observation
also does not establish exact reverse initialization or learned separation.
Unit tests validate specific numerical/API properties, not final research claims.

## Metrics

Primary source IDs are fixed; no PIT or delay alignment is applied. SI-SDR uses
zero-mean waveforms and optimal scalar projection onto each reference. nMSE and
nMAE use sums over the stored one-sided spectral bins, divided by corresponding
reference sums. These are the earlier unweighted rFFT metrics, not a claim of
Parseval-equivalent full time energy. Source-image metrics explicitly convolve
the cropped latent reconstruction in time. The un-cropped state tail is reported.
