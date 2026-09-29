"""Ordered-source metrics. No unreported permutation or delay alignment."""
from __future__ import annotations
import numpy as np


def si_sdr(estimate, reference, eps=1e-12):
    estimate, reference = np.asarray(estimate, np.float64), np.asarray(reference, np.float64)
    if estimate.shape != reference.shape:
        raise ValueError("SI-SDR inputs must have matching shape")
    estimate = estimate - estimate.mean(-1, keepdims=True)
    reference = reference - reference.mean(-1, keepdims=True)
    ref_energy = np.sum(reference**2, axis=-1, keepdims=True)
    if np.any(ref_energy <= eps):
        raise ValueError("SI-SDR is undefined for a silent reference")
    scale = np.sum(estimate * reference, axis=-1, keepdims=True) / ref_energy
    target = scale * reference
    num = np.sum(target**2, axis=-1)
    den = np.sum((estimate - target)**2, axis=-1)
    # Avoid spuriously giving 0 dB to an all-zero estimate: use a reference-scaled
    # denominator floor rather than adding identical eps to numerator/denominator.
    floor = eps * ref_energy[..., 0]
    ratio = np.maximum(num, floor * eps) / np.maximum(den, floor)
    return 10 * np.log10(ratio)


def nmse(estimate, reference, eps=1e-12):
    estimate, reference = np.asarray(estimate), np.asarray(reference)
    if estimate.shape != reference.shape:
        raise ValueError("nMSE inputs must have matching shape")
    return np.sum(np.abs(estimate-reference)**2, axis=-1) / np.maximum(
        np.sum(np.abs(reference)**2, axis=-1), eps)


def nmae(estimate, reference, eps=1e-12):
    return np.sum(np.abs(estimate-reference), axis=-1) / np.maximum(np.sum(np.abs(reference), axis=-1), eps)
