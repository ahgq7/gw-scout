from __future__ import annotations

import math
import logging
from typing import Any, Tuple, List

import numpy as np

try:
    from pycbc.filter import sigma  # noqa: F401
    from pycbc.waveform import get_td_waveform  # noqa: F401
except ImportError:
    sigma = None  # type: ignore[assignment]
    get_td_waveform = None  # type: ignore[assignment]

LOG = logging.getLogger(__name__)
_MISSING_CHISQ_WARNED = False


def compute_chisq_batch(strain, template, psd, peak_indices: np.ndarray, cfg) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute chi-squared for multiple peaks in a SINGLE BATCH (best practice).
    
    Uses PyCBC's SingleDetPowerChisq for precomputation and efficient batched evaluation.
    This is the operational pipeline standard (PyCBC/GstLAL methodology).
    
    Args:
        strain: conditioned strain data
        template: waveform template
        psd: power spectral density
        peak_indices: array of indices where to compute chi-squared
        cfg: search configuration
    
    Returns:
        (chisq_array, rchisq_array) - chi-squared values for all peaks
    """
    global _MISSING_CHISQ_WARNED
    
    nbins = 16
    dof = 2 * nbins - 2
    
    if len(peak_indices) == 0:
        return np.array([]), np.array([])
    
    try:
        from pycbc.vetoes.chisq import power_chisq
    except Exception:
        if not _MISSING_CHISQ_WARNED:
            LOG.warning("pycbc.vetoes.chisq unavailable; using rchisq=1 fallback (one-time warning)")
            _MISSING_CHISQ_WARNED = True
        return np.full(len(peak_indices), float(dof)), np.ones(len(peak_indices))
    
    # power_chisq expects FrequencySeries template; pass FD template directly
    # Compute chi-squared time series once
    hfc = getattr(cfg.conditioning, "high_frequency_cutoff", None)
    try:
        chisq_ts = power_chisq(
            template,
            strain,
            num_bins=nbins,
            psd=psd,
            low_frequency_cutoff=cfg.conditioning.low_frequency_cutoff,
            high_frequency_cutoff=hfc,
        )
    except Exception as exc:
        if not _MISSING_CHISQ_WARNED:
            LOG.warning("power_chisq failed (%s); using rchisq=1 fallback (one-time warning)", exc)
            _MISSING_CHISQ_WARNED = True
        return np.full(len(peak_indices), float(dof)), np.ones(len(peak_indices))
    
    # Apply same cropping as SNR (25% wraparound removal)
    # Match the SNR cropping in single_ifo.py
    crop_start = len(chisq_ts)//4
    crop_end = len(chisq_ts)*3//4
    chisq_ts = chisq_ts[crop_start:crop_end]
    
    # Extract chi-squared values at all peak indices (batched)
    chisq_values = []
    rchisq_values = []
    
    for idx in peak_indices:
        if 0 <= idx < len(chisq_ts):
            # chi-squared is a sum of squares and cannot be negative;
            # clamp any numerical underflow to zero
            chisq = max(0.0, float(chisq_ts[idx]))
            rchisq = chisq / dof if dof > 0 else 1.0
        else:
            # Out of range - use fallback
            chisq = float(dof)
            rchisq = 1.0
        
        chisq_values.append(chisq)
        rchisq_values.append(rchisq)
    
    return np.array(chisq_values), np.array(rchisq_values)


def compute_chisq(strain, template, psd, peak_idx: int, cfg, snr_series=None) -> Tuple[float, float]:
    """
    Allen chi-square (reduced chi-square returned as rchisq).
    Uses PyCBC's vetoes.chisq implementation (modern / supported path).
    
    Args:
        strain: conditioned strain data
        template: waveform template
        psd: power spectral density
        peak_idx: index in SNR time series where to compute chi-squared
        cfg: search configuration
        snr_series: optional pre-computed SNR series (avoids recomputation)
    
    Returns:
        (chisq, rchisq) tuple
    """
    global _MISSING_CHISQ_WARNED

    nbins = 16
    dof = 2 * nbins - 2

    try:
        from pycbc.vetoes.chisq import power_chisq
    except Exception:
        if not _MISSING_CHISQ_WARNED:
            LOG.warning("pycbc.vetoes.chisq unavailable; using rchisq=1 fallback (one-time warning)")
            _MISSING_CHISQ_WARNED = True
        return float(dof), 1.0

    # Allen chi-square time series
    # CRITICAL: Must use same cropping as SNR to maintain index alignment
    hfc = getattr(cfg.conditioning, "high_frequency_cutoff", None)
    try:
        chisq_ts = power_chisq(
            template,
            strain,
            num_bins=nbins,
            psd=psd,
            low_frequency_cutoff=cfg.conditioning.low_frequency_cutoff,
            high_frequency_cutoff=hfc,
        )
    except Exception as exc:
        if not _MISSING_CHISQ_WARNED:
            LOG.warning("power_chisq failed (%s); using rchisq=1 fallback (one-time warning)", exc)
            _MISSING_CHISQ_WARNED = True
        return float(dof), 1.0
    
    # Apply same cropping as SNR (4 seconds on each side, as done in single_ifo.py)
    pad = 4
    if len(chisq_ts) > 2 * pad:
        chisq_ts = chisq_ts.crop(pad, pad)

    # Guard peak index
    if peak_idx < 0 or peak_idx >= len(chisq_ts):
        if not _MISSING_CHISQ_WARNED:
            LOG.warning("peak_idx=%d out of range [0, %d) for chisq_ts; using rchisq=1 fallback (one-time warning)",
                       peak_idx, len(chisq_ts))
            _MISSING_CHISQ_WARNED = True
        return float(dof), 1.0

    chisq = float(chisq_ts[peak_idx])
    rchisq = chisq / dof if dof > 0 else 1.0
    return chisq, float(rchisq)


def new_snr(snr: float, chisq: float, duration: float, cfg) -> float:
    """
    PyCBC newSNR reweighting with robust NaN handling.
    
    Formula: newSNR = SNR / [((1 + rχ²³)/2)^(1/6)]
    
    Args:
        snr: Signal-to-noise ratio
        chisq: Chi-squared value
        duration: Template duration
        cfg: Configuration
    
    Returns:
        Reweighted SNR (handles NaN/inf gracefully)
    """
    # PyCBC newSNR reweighting
    dof = 16 * 2 - 2  # 30 for 16 bins
    
    # Validate inputs
    if not np.isfinite(snr) or snr <= 0:
        return 0.0
    
    if not np.isfinite(chisq) or chisq < 0:
        # Fallback to plain SNR if chi-squared invalid
        return float(snr)
    
    # Compute reduced chi-squared
    rchisq = chisq / dof if dof > 0 else 1.0
    
    # Guard against pathological rchisq values
    if not np.isfinite(rchisq):
        return float(snr)
    
    # Clamp rchisq to reasonable range to avoid numerical issues
    # Very high rchisq (glitch) should heavily penalize
    # Very low rchisq is possible but rare
    rchisq = np.clip(rchisq, 0.0, 1000.0)
    
    # Compute reweighting factor with error handling
    try:
        # Standard PyCBC formula
        reweight_factor = ((1.0 + rchisq**3) / 2.0) ** (1.0 / 6.0)
        
        if not np.isfinite(reweight_factor) or reweight_factor <= 0:
            return float(snr)
        
        new_snr_val = snr / reweight_factor
        
        if not np.isfinite(new_snr_val):
            return float(snr)
        
        return float(new_snr_val)
        
    except (FloatingPointError, ZeroDivisionError, ValueError):
        # Fallback to plain SNR
        return float(snr)
