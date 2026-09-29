"""ConvMixSDE with the names and eigenbasis ordering used in the shared code.

State order: [source_1(all F bins), ..., source_K(all F bins)].
Modal order: [nonzero_mode(all F bins), null_1(all F bins), ...].

get_Ft(t) = exp(phi(t)); get_Gt(t) = phi'(t), NOT the same object.
B_t = g(t) V. S_t = V diag(s_t), s_t = sqrt(q_t) exp(i Im(phi_t)).
All normal operations use per-frequency KxK blocks. Dense attributes remain
available as lazy compatibility views, but are not used by training/sampling.

Complex score convention (see docs/MATHEMATICS.md): interior rFFT bins use
Wirtinger score -Sigma^{-1}(x-mu) for a proper complex Gaussian; real endpoints
use the real Gaussian score. Reverse coefficients are 1 (SDE) and 1/2 (PF).
"""
from __future__ import annotations

import abc
import math
from numbers import Integral

import torch
from torch import Tensor


class SDE(abc.ABC):
    def __init__(self, T, N):
        if not math.isfinite(T) or T <= 0 or not isinstance(N, Integral) or N < 1:
            raise ValueError("Require finite T>0 and integer N>=1")
        self.T, self.N = float(T), int(N)

    @abc.abstractmethod
    def sde(self, x, y, t):
        raise NotImplementedError

    @abc.abstractmethod
    def marginal_prob(self, x0, t):
        raise NotImplementedError

    def discretize(self, x, y, t, dt=None):
        """Positive forward increments (f,G); reverse users SUBTRACT rev_f.

        Supply the actual interval dt when stopping at t_eps. T/N is retained
        only as a backwards-compatible default.
        """
        dt = self.T / self.N if dt is None else float(dt)
        if dt <= 0:
            raise ValueError("dt must be a positive interval length")
        drift, diffusion = self.sde(x, y, t)
        return drift * dt, diffusion * math.sqrt(dt)

    def reverse(self, score_model, probability_flow=False):
        return _ReverseSDE(self, score_model, probability_flow)


class _ReverseSDE:
    """Delegating wrapper rather than an incompletely initialized subclass."""
    def __init__(self, forward, score_model, probability_flow):
        self.forward, self.score_model = forward, score_model
        self.probability_flow = bool(probability_flow)
        self.N, self.T = forward.N, forward.T

    def rsde_parts(self, x, y, t):
        drift, diffusion = self.forward.sde(x, y, t)
        score = self.score_model(x, y, t)
        coefficient = 0.5 if self.probability_flow else 1.0
        score_drift = -coefficient * self.forward.apply_BBH(score, t)
        return {"total_drift": drift + score_drift,
                "diffusion": torch.zeros_like(diffusion) if self.probability_flow else diffusion,
                "sde_drift": drift, "sde_diffusion": diffusion,
                "score_drift": score_drift, "score": score}

    def sde(self, x, y, t):
        parts = self.rsde_parts(x, y, t)
        return parts["total_drift"], parts["diffusion"]

    def discretize(self, x, y, t, dt=None):
        dt = self.T / self.N if dt is None else float(dt)
        drift, diffusion = self.sde(x, y, t)
        if dt <= 0:
            raise ValueError("Use a positive interval length and x_next=x-rev_f+rev_G*z")
        return drift * dt, diffusion * math.sqrt(dt)

    def __getattr__(self, name):
        return getattr(self.forward, name)


class ConvMixSDE(SDE):
    def __init__(self, ndim, alpha, beta, sigma_min, sigma_max, steps, T, N=1000,
                 *, drift_mode="mixing"):
        super().__init__(T, N)
        if not all(math.isfinite(v) and v > 0 for v in (alpha, beta, sigma_min, sigma_max)):
            raise ValueError("alpha, beta, sigma_min and sigma_max must be finite and positive")
        if sigma_max <= sigma_min or not isinstance(steps, Integral) or steps < 2:
            raise ValueError("Require sigma_max>sigma_min and integer steps>=2")
        if not isinstance(ndim, Integral) or ndim < 2:
            raise ValueError("ndim must be the integer flattened state dimension")
        if drift_mode not in {"mixing", "none", "instantaneous"}:
            raise ValueError("drift_mode must be mixing, none or instantaneous")
        self.ndim, self.alpha, self.beta = int(ndim), float(alpha), float(beta)
        self.sigma_min, self.sigma_max = float(sigma_min), float(sigma_max)
        self.ratiosig = self.sigma_max / self.sigma_min
        self.logsig, self.steps = math.log(self.ratiosig), int(steps)
        self.drift_mode = drift_mode
        self.kernels = self.eigvals = self.V = self.V_inv = None
        self.N_fft = self.n_fft = self.K = None
        self._dense_cache = {}

    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sde-n", type=int, default=1000)
        parser.add_argument("--theta", type=float, default=1.5)  # historical CLI name
        parser.add_argument("--sigma-min", type=float, default=0.05)
        parser.add_argument("--sigma-max", type=float, default=0.5)
        return parser

    def get_mix_mat(self, kernels, N_fft):
        if not torch.is_tensor(kernels) or kernels.ndim != 2:
            raise ValueError("kernels must be [K,F]")
        if not kernels.is_complex():
            kernels = kernels.to(torch.complex128 if kernels.dtype == torch.float64 else torch.complex64)
        if kernels.dtype not in (torch.complex64, torch.complex128) or not torch.isfinite(kernels).all():
            raise ValueError("kernels must be finite complex64/complex128")
        K, F = kernels.shape
        if K < 2 or F != N_fft or K * F != self.ndim:
            raise ValueError(f"Require K>=2, N_fft=F={F} and ndim=K*F={K*F}")
        # Do NOT switch eigenvector normalization/pivot: B=gV depends on V.
        self.kernels = kernels.detach().clone()
        self.K, self.N_fft = K, F
        eigvals_nonzero = kernels.sum(0)
        tolerance = 10 * torch.finfo(kernels.real.dtype).eps
        scale = kernels.abs().sum(0).clamp_min(torch.finfo(kernels.real.dtype).tiny)
        if torch.any(eigvals_nonzero.abs() <= tolerance * scale):
            raise ValueError("sum_k H_k is zero/near cancellation; the prescribed eigensystem is singular")
        if torch.any(kernels[0].abs() <= tolerance * kernels.abs().amax(0)):
            raise ValueError("Original eigenbasis divides by H1, which is zero/near-zero")
        V = torch.zeros(F, K, K, device=kernels.device, dtype=kernels.dtype)
        V[:, :, 0] = 1
        for j in range(1, K):
            V[:, j, j] = 1
            V[:, 0, j] = -kernels[j] / kernels[0]
        self.V, self.V_inv = V, torch.linalg.inv(V)
        self.eigvals = torch.cat([eigvals_nonzero, kernels.new_zeros(F * (K - 1))])
        self._dense_cache.clear()
        if self.n_fft is not None:
            self.set_rfft(self.n_fft)

    def set_rfft(self, n_fft: int):
        """Check, do not silently project, validity of the logarithmic mean at endpoints."""
        self._require_operator()
        if n_fft // 2 + 1 != self.N_fft:
            raise ValueError("n_fft does not match the number of rFFT bins")
        endpoints = [0] + ([-1] if n_fft % 2 == 0 else [])
        H = self.kernels[:, endpoints]
        tol = 50 * torch.finfo(H.real.dtype).eps
        if H.imag.abs().max() > tol * H.abs().max().clamp_min(1):
            raise ValueError("Real-signal FIR transfer functions must be real at DC/Nyquist")
        if torch.any(H.sum(0).real <= 0):
            raise ValueError("Principal-log mean requires positive sum(H) at real rFFT endpoints. "
                             "A negative endpoint would leave the real-signal subspace.")
        self.n_fft = int(n_fft)

    def _require_operator(self):
        if self.V is None:
            raise RuntimeError("Call get_mix_mat(kernels, N_fft=F) first")

    def send_to(self, device):
        self._require_operator()
        device = torch.device(device)
        if self.kernels.device != device:
            for name in ("kernels", "eigvals", "V", "V_inv"):
                setattr(self, name, getattr(self, name).to(device))
            self._dense_cache.clear()
        return self

    def _time(self, t, batch=None, covariance=False):
        self._require_operator()
        t = torch.as_tensor(t, device=self.kernels.device, dtype=self.kernels.real.dtype)
        if t.ndim == 0:
            t = t[None]
        if t.ndim != 1 or not torch.isfinite(t).all() or torch.any(t < 0):
            raise ValueError("t must be a finite nonnegative scalar or [B]")
        if covariance and torch.any(t > self.T + 1e-6 * self.T):
            raise ValueError("Covariance/diffusion requested outside [0,T]")
        if batch is not None:
            if t.numel() == 1:
                t = t.expand(batch)
            elif t.numel() != batch:
                raise ValueError("t must be scalar or have the same batch size as x")
        return t

    def _state(self, x):
        self._require_operator()
        if x.ndim != 2 or x.shape[1] != self.ndim:
            raise ValueError(f"Expected flattened complex [B,{self.ndim}], got {tuple(x.shape)}")
        if x.dtype != self.kernels.dtype or x.device != self.kernels.device:
            raise ValueError("SDE/state dtype or device mismatch; call send_to(x.device)")
        return x.reshape(x.shape[0], self.K, self.N_fft).transpose(1, 2)

    def _block_apply(self, blocks, x):
        out = torch.einsum("fij,bfj->bfi", blocks, self._state(x))
        return out.transpose(1, 2).reshape(x.shape)

    def apply_V(self, x):
        return self._block_apply(self.V, x)

    def apply_V_inv(self, x):
        return self._block_apply(self.V_inv, x)

    def apply_VH(self, x):
        return self._block_apply(self.V.mH, x)

    def apply_V_inv_H(self, x):
        return self._block_apply(self.V_inv.mH, x)

    def _phi(self, t):
        t = self._time(t)
        # Compute only genuinely nonzero logarithms (never evaluate log(0)).
        log_lambda = self.eigvals[:self.N_fft].log()
        nonzero = -torch.expm1(-self.alpha * t[:, None]) * log_lambda[None]
        null = (-self.beta * t[:, None]).expand(-1, (self.K - 1) * self.N_fft)
        if self.drift_mode == "none":
            return self.eigvals.new_zeros(t.numel(), self.ndim)
        return torch.cat([nonzero, null.to(nonzero.dtype)], dim=-1)

    def get_Ft(self, t):
        """Eigenvalues of the integrated mean propagator; name retained."""
        if self.drift_mode == "instantaneous":
            raise NotImplementedError("Instantaneous drift is not diagonal in the fixed V basis; use _mean().")
        return self._phi(t).exp()

    def get_Gt(self, t):
        """Instantaneous eigenvalues: alpha*log(lambda)*exp(-alpha*t), -beta."""
        if self.drift_mode == "instantaneous":
            raise NotImplementedError("Use apply_drift() for the instantaneous ablation.")
        t = self._time(t)
        nz = self.alpha * torch.exp(-self.alpha * t[:, None]) * self.eigvals[:self.N_fft].log()[None]
        null = torch.full((t.numel(), (self.K - 1) * self.N_fft), -self.beta,
                          device=t.device, dtype=nz.dtype)
        if self.drift_mode == "none":
            return torch.zeros(t.numel(), self.ndim, dtype=nz.dtype, device=t.device)
        return torch.cat([nz, null], dim=-1)

    def _instantaneous_mean(self, x0, t):
        # Matched instantaneous-projector ablation: normalized equal-weight P.
        # alpha does not act on P; only beta acts on (I-P). B=gV is held fixed.
        x = x0.reshape(-1, self.K, self.N_fft)
        p = x.mean(1, keepdim=True)
        return (p + torch.exp(-self.beta * t[:, None, None]) * (x - p)).reshape_as(x0)

    def _mean(self, x0, t):
        self.send_to(x0.device)
        t = self._time(t, x0.shape[0])
        if self.drift_mode == "instantaneous":
            return self._instantaneous_mean(x0, t)
        return self.apply_V(self.get_Ft(t) * self.apply_V_inv(x0))

    def apply_drift(self, x, t):
        self.send_to(x.device)
        t = self._time(t, x.shape[0])
        if self.drift_mode == "instantaneous":
            X = x.reshape(-1, self.K, self.N_fft)
            return (-self.beta * (X - X.mean(1, keepdim=True))).reshape_as(x)
        return self.apply_V(self.get_Gt(t) * self.apply_V_inv(x))

    def g_squared(self, t):
        t = self._time(t, covariance=True)
        rate = 2 * self.logsig / self.T
        return (self.sigma_min**2 * rate) * torch.exp(rate * t)

    def apply_B(self, z, t):
        t = self._time(t, z.shape[0], covariance=True)
        return self.g_squared(t).sqrt()[:, None] * self.apply_V(z)

    def apply_BBH(self, x, t):
        t = self._time(t, x.shape[0], covariance=True)
        return self.g_squared(t)[:, None] * self.apply_V(self.apply_VH(x))

    def sde(self, x, y, t):
        del y
        drift = self.apply_drift(x, t)
        t = self._time(t, x.shape[0], covariance=True)
        diffusion = self.g_squared(t).sqrt()[:, None, None] * self.eigvecs[None]
        return drift, diffusion

    def _std(self, t):
        """Modal S entries [B,KF]. Never constructs a batch of KF x KF factors.

        Quadrature uses exp(2 Re(phi(t)-phi(s))) g(s)^2, avoiding an overflow
        followed by an underflow. Null-mode variance has an analytic formula.
        The phase/gauge matches the final shared _std implementation.
        """
        if self.drift_mode == "instantaneous":
            raise NotImplementedError("Instantaneous drift needs non-diagonal covariance; use factor_blocks().")
        t = self._time(t, covariance=True)
        rate = 2 * self.logsig / self.T
        if self.drift_mode == "none":
            q = self.sigma_min**2 * torch.expm1(rate * t)
            return q.sqrt()[:, None].expand(-1, self.ndim).to(self.kernels.dtype)
        u = torch.linspace(0, 1, self.steps, device=t.device, dtype=t.dtype)
        s = t[:, None] * u[None]
        g2 = self.sigma_min**2 * rate * torch.exp(rate * s)
        phi_scale_s = -torch.expm1(-self.alpha * s)
        phi_scale_t = -torch.expm1(-self.alpha * t)
        chunks = []
        for start in range(0, self.N_fft, 64):
            loglam = self.eigvals[start:min(start + 64, self.N_fft)].log()
            phi_t = phi_scale_t[:, None] * loglam[None]
            delta = (phi_scale_t[:, None, None] - phi_scale_s[:, None, :]) * loglam.real[None, :, None]
            values = (2 * delta).exp() * g2[:, None, :]
            q = t[:, None] * torch.trapezoid(values, x=u, dim=-1)
            chunks.append(q.clamp_min(0).sqrt().to(self.kernels.dtype) * torch.exp(1j * phi_t.imag))
        rate0 = rate + 2 * self.beta
        qzero = self.sigma_min**2 * rate * torch.exp(rate * t) * (-torch.expm1(-rate0 * t)) / rate0
        null = qzero.clamp_min(0).sqrt()[:, None].expand(-1, (self.K - 1) * self.N_fft)
        result = torch.cat([torch.cat(chunks, -1), null.to(self.kernels.dtype)], -1)
        if not torch.isfinite(result).all():
            raise FloatingPointError("Covariance factor is nonfinite; inspect filters/schedule")
        return result

    def factor_blocks(self, t):
        """[B,F,K,K] factor for all modes, including instantaneous ablation.

        For mixing/none this is V diag(s). For instantaneous we compute its
        covariance independently in frequency blocks and use Cholesky.
        """
        t = self._time(t, covariance=True)
        if self.drift_mode != "instantaneous":
            std = self._std(t).reshape(-1, self.K, self.N_fft).transpose(1, 2)
            return self.V[None] * std[:, :, None, :]
        # Gamma(t,s)=P+exp(-beta*(t-s))Q for instantaneous projector drift.
        P = torch.ones(self.K, self.K, device=t.device, dtype=self.kernels.dtype) / self.K
        Q = torch.eye(self.K, device=t.device, dtype=self.kernels.dtype) - P
        C = self.V @ self.V.mH
        rate = 2 * self.logsig / self.T
        def integral(decay):
            return self.sigma_min**2 * rate * torch.exp(rate * t) * (-torch.expm1(-(rate + decay) * t)) / (rate + decay)
        q0, q1, q2 = integral(0), integral(self.beta), integral(2 * self.beta)
        Sigma = (q0[:, None, None, None] * (P @ C @ P)[None]
            + q1[:, None, None, None] * (P @ C @ Q + Q @ C @ P)[None]
            + q2[:, None, None, None] * (Q @ C @ Q)[None])
        Sigma = (Sigma + Sigma.mH) / 2
        # t=0 factor is exactly zero, not an artificial variance floor.
        positive = t > 0
        S = torch.zeros_like(Sigma)
        if positive.any():
            S[positive] = torch.linalg.cholesky(Sigma[positive])
        return S

    def fast_factor(self, t):
        """Compact factor representation, local to one loss/sampling call."""
        return self.factor_blocks(t) if self.drift_mode == "instantaneous" else self._std(t)

    def apply_std(self, std, x):
        if std.ndim == 2:
            return self.apply_V(std * x)
        return torch.einsum("bfij,bfj->bfi", std, self._state(x)).transpose(1, 2).reshape_as(x)

    def apply_std_h(self, std, x):
        if std.ndim == 2:
            return std.conj() * self.apply_VH(x)
        return torch.einsum("bfij,bfj->bfi", std.mH, self._state(x)).transpose(1, 2).reshape_as(x)

    def apply_std_inv(self, std, x):
        if std.ndim == 2:
            if torch.any(std.abs() == 0):
                raise ValueError("S is singular at t=0")
            return self.apply_V_inv(x) / std
        out = torch.linalg.solve(std, self._state(x).unsqueeze(-1)).squeeze(-1)
        return out.transpose(1, 2).reshape_as(x)

    def _var(self, t):
        return self._std(t).abs().square()

    def _blocks_to_dense(self, blocks):
        # [...,F,K,K] -> [...,KF,KF] in the historical source-major order.
        shape = blocks.shape[:-3]
        eye = torch.eye(self.N_fft, device=blocks.device, dtype=blocks.dtype)
        return torch.einsum("...fij,fg->...ifjg", blocks, eye).reshape(*shape, self.ndim, self.ndim)

    @property
    def A(self):
        if "A" not in self._dense_cache:
            self._dense_cache["A"] = self._blocks_to_dense(self.kernels.T[:, None, :].expand(-1, self.K, -1))
        return self._dense_cache["A"]

    @property
    def eigvecs(self):
        if "V" not in self._dense_cache:
            self._dense_cache["V"] = self._blocks_to_dense(self.V)
        return self._dense_cache["V"]

    @property
    def eigvecs_inv(self):
        if "V_inv" not in self._dense_cache:
            self._dense_cache["V_inv"] = self._blocks_to_dense(self.V_inv)
        return self._dense_cache["V_inv"]

    def _cov(self, t):
        S = self.factor_blocks(t)
        return self._blocks_to_dense(S @ S.mH)

    def marginal_prob(self, x0, t):
        """Legacy dense output retained for old tests; training uses fast_factor."""
        t = self._time(t, x0.shape[0], covariance=True)
        return self._mean(x0, t), self._blocks_to_dense(self.factor_blocks(t))

    @staticmethod
    def mult_std(std, x):
        return std @ x

    @staticmethod
    def mult_std_inv(std, x):
        return torch.linalg.solve(std, x)

    def sample_time_varprop(self, n, t_eps=0.0, device=None):
        """Times proportional to RMS modal std, using a conservative envelope."""
        if self.drift_mode == "instantaneous":
            raise NotImplementedError("varprop not exposed for instantaneous ablation")
        if n < 1 or not 0 <= t_eps < self.T:
            raise ValueError("Invalid sample count/time interval")
        if device is not None:
            self.send_to(device)
        # Absolute modal variance bound on [0,T]; no monotonicity assumption.
        scale = max(1.0, float(self.eigvals.abs().max()))
        bound = math.sqrt(self.sigma_max**2 - self.sigma_min**2) * scale
        output = []
        count = 0
        for _ in range(10000):
            size = min(256, max(8, 2 * (n - count)))
            t = torch.rand(size, device=self.kernels.device, dtype=self.kernels.real.dtype) * (self.T - t_eps) + t_eps
            weight = self._var(t).mean(-1).sqrt()
            take = t[torch.rand_like(t) * bound < weight]
            output.append(take)
            count += take.numel()
            if count >= n:
                return torch.cat(output)[:n]
        raise RuntimeError("Time rejection sampler exhausted its proposal budget")

    def copy(self):
        result = ConvMixSDE(self.ndim, self.alpha, self.beta, self.sigma_min, self.sigma_max,
                            self.steps, self.T, self.N, drift_mode=self.drift_mode)
        if self.kernels is not None:
            result.get_mix_mat(self.kernels, self.N_fft)
            if self.n_fft is not None:
                result.set_rfft(self.n_fft)
        return result

    def prior_sampling(self, shape, y):
        raise NotImplementedError("Use sampling.initialize_terminal_state: finite-T prior is approximate")

    def prior_logp(self, z):
        raise NotImplementedError("No exact observation-conditioned terminal density is assumed")
