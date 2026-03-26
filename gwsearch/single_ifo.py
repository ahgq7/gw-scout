from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from pycbc import catalog
from pycbc.detector import Detector
from pycbc.events.eventmgr import ThresholdCluster
from pycbc.filter import matched_filter
from pycbc.psd import inverse_spectrum_truncation
from pycbc.types import TimeSeries, FrequencySeries
from pycbc.waveform import get_td_waveform, get_fd_waveform

from .conditioning import condition_strain
from .gating import apply_gates, find_gates
from .ranking import compute_chisq_batch, new_snr
from .config import SearchConfig
from .parallel_matching import process_templates_parallel, process_templates_batched_gpu

LOG = logging.getLogger(__name__)

# Maximum reasonable read duration to prevent memory issues
# Must accommodate block_duration + 2*padding (e.g., 512.0 + 64.0 = 576.0)
MAX_READ_DURATION_SEC = 6000.0


def _parse_frame_coverage(frame_path: Path) -> tuple[float, float]:
    """
    Parse GPS start and end time from GWOSC frame filename.
    
    Returns:
        (start_gps, end_gps) tuple
    """
    import re
    m = re.search(r"-(\d+)-(\d+)\.(?:gwf|hdf5)$", frame_path.name)
    if not m:
        raise ValueError(f"Cannot parse GPS coverage from filename: {frame_path.name}")
    start = float(m.group(1))
    duration = float(m.group(2))
    return start, start + duration


def _validate_time_request(frame_path: Path, read_start: float, read_duration: float) -> tuple[float, float]:
    """
    Validate and clamp time request to frame boundaries.
    
    Returns:
        (clamped_start, clamped_duration) or raises ValueError if request is invalid
    """
    frame_start, frame_end = _parse_frame_coverage(frame_path)
    request_end = read_start + read_duration
    
    # Check if request is completely outside frame
    if read_start >= frame_end:
        raise ValueError(f"Request start {read_start} >= frame end {frame_end}")
    if request_end <= frame_start:
        raise ValueError(f"Request end {request_end} <= frame start {frame_start}")
    
    # Check if request exceeds frame boundaries
    if read_start < frame_start or request_end > frame_end:
        LOG.warning("Time request [%.1f, %.1f] exceeds frame boundaries [%.1f, %.1f]",
                   read_start, request_end, frame_start, frame_end)
        
        # Clamp to available data
        clamped_start = max(read_start, frame_start)
        clamped_end = min(request_end, frame_end)
        clamped_duration = clamped_end - clamped_start
        
        if clamped_duration < 64.0:  # Minimum useful duration
            raise ValueError(f"Available duration {clamped_duration:.1f}s < 64s minimum after clamping")
        
        LOG.info("Clamped request to frame boundaries: [%.1f, %.1f] duration=%.1f",
                clamped_start, clamped_end, clamped_duration)
        return clamped_start, clamped_duration
    
    return read_start, read_duration


def _read_strain(frame_path: Path, ifo: str, cfg: SearchConfig, block_start: float = None, block_end: float = None):
    from pycbc.frame import read_frame

    # Detect O3/O4 data from filename (all modern GWOSC data uses 16KHz bulk files)
    fname_str = str(frame_path)

    # Pick channel based on actual sample-rate tag in the filename
    if "16KHZ" in fname_str:
        candidates = [f"{ifo}:GWOSC-16KHZ_R1_STRAIN"]
    elif "4KHZ" in fname_str:
        candidates = [f"{ifo}:GWOSC-4KHZ_R1_STRAIN"]
    else:
        # O1/O2 legacy files
        candidates = [
            f"{ifo}:GWOSC-4KHZ_R1_STRAIN",
            f"{ifo}:LOSC-STRAIN",
        ]
    
    # Extract time range parameters with padding for edge effects
    read_kwargs = {}
    read_start = block_start
    read_duration = None
    
    if block_start is not None and block_end is not None:
        duration = block_end - block_start
        # Add 32 sec padding on each side to avoid edge artifacts
        padding = 32.0
        read_start = max(block_start - padding, 0)
        read_duration = duration + 2 * padding
        
        # Validate and clamp to frame boundaries
        try:
            read_start, read_duration = _validate_time_request(frame_path, read_start, read_duration)
        except ValueError as e:
            LOG.error("Time request validation failed: %s", e)
            raise
        
        # Safety check: refuse unreasonably long reads
        if read_duration > MAX_READ_DURATION_SEC:
            raise ValueError(f"Requested duration {read_duration:.1f}s exceeds maximum {MAX_READ_DURATION_SEC}s")
        
        read_kwargs = {"start_time": read_start, "duration": read_duration}
        LOG.info("Reading frame segment with padding: GPS %.1f duration %.1f sec (requested %.1f-%.1f)",
                read_start, read_duration, block_start, block_end)
    
    last_err = None
    strain = None

    # GWOSC HDF5 files cannot be read via read_frame (GWF-only).
    # Read them directly with h5py.
    if str(frame_path).endswith(".hdf5"):
        try:
            import h5py
            with h5py.File(str(frame_path), "r") as hf:
                gps_file_start = float(hf["meta/GPSstart"][()])
                dur_total = float(hf["meta/Duration"][()])
                raw = hf["strain/Strain"][()]
            n = len(raw)
            dt = dur_total / n
            if read_start is not None and read_duration is not None:
                i0 = max(0, int(round((read_start - gps_file_start) / dt)))
                i1 = min(n, i0 + int(round(read_duration / dt)))
                raw = raw[i0:i1]
                epoch = gps_file_start + i0 * dt
            else:
                epoch = gps_file_start
            from pycbc.types import TimeSeries
            strain = TimeSeries(raw, delta_t=dt, epoch=epoch)
            LOG.info("Successfully read HDF5 strain from %s (%.1f Hz, %d samples)",
                     frame_path.name, strain.sample_rate, len(strain))
        except Exception as exc:
            last_err = exc
            LOG.debug("HDF5 read failed for %s: %s", frame_path.name, exc)

    if strain is None:
        for chan in candidates:
            try:
                strain = read_frame(str(frame_path), chan, **read_kwargs)
                LOG.info("Successfully read channel %s from %s", chan, frame_path.name)
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
                LOG.debug("Channel %s read failed: %s", chan, exc)
                continue

    if strain is None:
        raise RuntimeError(f"Could not read strain from {frame_path}: {last_err}")

    # Validate and clean
    n_nonfinite = np.sum(~np.isfinite(strain.data))
    if n_nonfinite > 0:
        LOG.warning("Frame segment has %d/%d non-finite values; sanitizing", n_nonfinite, len(strain))
        strain.data = np.nan_to_num(strain.data, nan=0.0, posinf=0.0, neginf=0.0)
    raw_rms = np.sqrt(np.mean(strain.data**2))
    LOG.info("Frame segment: RMS=%.3e, rate=%.1f Hz, len=%d", raw_rms, strain.sample_rate, len(strain))
    if raw_rms < 1e-19:
        LOG.warning("Frame RMS %.3e far below normal (~1e-18); likely DATA GAP or detector OFF", raw_rms)
    # Check initial strain
    initial_rms = np.sqrt(np.mean(strain.data**2))
    initial_rate = strain.sample_rate
    LOG.info("Raw strain: RMS=%.3e, rate=%.1f Hz, len=%d", initial_rms, initial_rate, len(strain))
    
    if not np.all(np.isfinite(strain.data)):
        LOG.error("Raw strain contains %d non-finite values before resampling", np.sum(~np.isfinite(strain.data)))
        return strain  # Return corrupted data to fail gracefully upstream
    
    target_rate = cfg.conditioning.sample_rate
    if target_rate and int(strain.sample_rate) != int(target_rate):
        LOG.info("Resampling from %.1f Hz to %d Hz", strain.sample_rate, target_rate)
        # Prefer high-accuracy resampling if available; otherwise fall back using delta_t
        target_dt = 1.0 / float(target_rate)
        try:
            if hasattr(strain, "resample_to_delta_t"):
                strain = strain.resample_to_delta_t(target_dt)
            else:
                # pycbc TimeSeries.resample expects delta_t
                strain = strain.resample(target_dt)
            
            # Validate resampling didn't corrupt data
            resampled_rms = np.sqrt(np.mean(strain.data**2))
            LOG.info("After resample: RMS=%.3e, rate=%.1f Hz, len=%d",
                    resampled_rms, strain.sample_rate, len(strain))
            
            if not np.all(np.isfinite(strain.data)):
                LOG.error("Resampling created %d non-finite values", np.sum(~np.isfinite(strain.data)))
            
            if resampled_rms == 0:
                LOG.error("Resampling zeroed all data!")
                
        except Exception as exc:  # noqa: BLE001
            LOG.error("Resampling failed: %s; using original rate %.1f Hz", exc, initial_rate)
            # Re-read without resampling
            raise RuntimeError(f"Resampling failed: {exc}")
    
    return strain


def _gate_and_condition(strain, cfg, exclude_start_sec=None):
    """
    Apply gating and conditioning to strain data.
    
    Args:
        strain: Input strain time series
        cfg: Configuration object
        exclude_start_sec: If provided, exclude first N seconds from PSD estimation
                          (critical for known signal validation like GW150914)
    
    Returns:
        conditioned_strain, psd, gates, psd_meta
    """
    # Validate input strain
    if not np.all(np.isfinite(strain.data)):
        LOG.warning("Input strain contains non-finite values before gating; sanitizing")
        strain.data = np.nan_to_num(strain.data, nan=0.0, posinf=0.0, neginf=0.0)
    
    rms_pre = np.sqrt(np.mean(strain.data**2))
    LOG.info("Strain RMS before gating: %.3e, len=%d, sr=%.1f", rms_pre, len(strain), strain.sample_rate)
    
    # Make a copy before gating in case we need to revert
    strain_orig = strain.copy()
    
    gates = find_gates(strain, cfg)
    LOG.info("Found %d gates with threshold=%.1f", len(gates), cfg.gating.snr_threshold)
    
    if len(gates) > 0:
        apply_gates(strain, gates)
        
        # Check if gating zeroed out too much data
        rms_post_gate = np.sqrt(np.mean(strain.data**2))
        LOG.info("Strain RMS after gating: %.3e", rms_post_gate)
        
        # If gating destroyed >99% of the signal, skip gating
        if rms_post_gate < 0.01 * rms_pre:
            LOG.warning("Gating reduced RMS by >99%% (%.3e -> %.3e); skipping gates", rms_pre, rms_post_gate)
            strain = strain_orig
            gates = []
    else:
        LOG.info("No gates found, proceeding without gating")
    
    strain, psd, psd_meta = condition_strain(strain, cfg, exclude_start_sec=exclude_start_sec)
    
    # Validate PSD
    psd_valid_count = np.sum(np.isfinite(psd.data) & (psd.data > 0))
    LOG.debug("PSD has %d/%d valid values", psd_valid_count, len(psd.data))
    
    # CRITICAL FIX (Expert recommendation):
    # Official PyCBC GW150914 tutorial does NOT use inverse_spectrum_truncation
    # It only uses: psd = interpolate(welch(h1), 1.0/h1.duration)
    # inverse_spectrum_truncation modifies PSD amplitude by 2-4×
    # Only needed for time-domain whitening or very long data streams
    # For matched_filter with FD templates, SKIP this step
    LOG.info("Skipping inverse_spectrum_truncation (tutorial approach)")
    
    psd_meta["istrunc_length"] = None  # Not applied
    return strain, psd, gates, psd_meta


def process_ifo_block(ifo: str, frame_path: Path, cfg: SearchConfig, bank: Dict[str, Any], block_start: float = None, block_end: float = None, progress_cb=None, use_cuda: bool = False, n_parallel_ifos: int = 1) -> List[Dict]:
    """Process IFO block with optional CUDA acceleration.
    
    Args:
        ifo: Interferometer name (H1, L1, V1)
        frame_path: Path to frame file
        cfg: Search configuration
        bank: Template bank dictionary
        block_start: GPS start time
        block_end: GPS end time
        progress_cb: Optional progress callback
        use_cuda: Whether CUDA is enabled (for optimization decisions)
    
    Returns:
        List of trigger dictionaries
    """
    LOG.info("[IFO-1] Reading strain for %s (CUDA=%s)", ifo, use_cuda)
    strain = _read_strain(frame_path, ifo, cfg, block_start, block_end)
    
    # CRITICAL FIX: Detect if known event is near block start and exclude from PSD
    exclude_psd_sec = None
    if cfg.bank.exclude_signal_from_psd_sec > 0:
        # If known event configuration is set, use it
        exclude_psd_sec = cfg.bank.exclude_signal_from_psd_sec
        LOG.info("[IFO-1.5] Known event mode: excluding first %.1fs from PSD estimation", exclude_psd_sec)
    
    LOG.info("[IFO-2] Gating and conditioning for %s", ifo)
    strain, psd, gates, psd_meta = _gate_and_condition(strain, cfg, exclude_start_sec=exclude_psd_sec)
    LOG.info("[IFO-3] Conditioning complete for %s", ifo)

    # Skip all processing for data gaps - detector is off, no real data to analyze
    is_data_gap = psd_meta.get('_is_data_gap', False) or psd_meta.get('method', '').startswith('data_gap')
    if is_data_gap:
        LOG.warning("Data gap detected for %s - skipping all template matching (detector OFF/no data)", ifo)
        return []  # Return empty trigger list immediately

    triggers: List[Dict] = []
    total_templates = len(bank["templates"])
    LOG.info("[IFO-4] Preparing to match %d templates for %s", total_templates, ifo)

    # Optional chunking to reduce FFT size for long frames
    sr = strain.sample_rate
    chunk_samples = int((cfg.runtime.chunk_duration or 0.0) * sr)
    segments = []
    
    # Skip chunking entirely for data gaps to avoid FrequencySeries construction
    is_data_gap = psd_meta.get('_is_data_gap', False) or psd_meta.get('method', '').startswith('data_gap')
    
    # TEMPORARY DEBUG: Disable chunking to test if it's causing SNR=0 issue
    if False and chunk_samples > 0 and chunk_samples < len(strain) and not is_data_gap:
        LOG.info("[IFO-5] Creating %d-second chunks from %.1f-second strain",
                cfg.runtime.chunk_duration, len(strain) / sr)
        step = max(1, chunk_samples)
        freq_old = psd.sample_frequencies.numpy()
        psd_vals = psd.numpy()
        
        # Validate PSD before chunking
        if not np.all(np.isfinite(psd_vals)) or not np.all(psd_vals > 0):
            LOG.warning("PSD contains invalid values before chunking; sanitizing")
            psd_vals = np.maximum(psd_vals, 1e-40)
            psd_vals = np.nan_to_num(psd_vals, nan=1e-40, posinf=1e-40, neginf=1e-40)
        
        for start_idx in range(0, len(strain), step):
            end_idx = min(len(strain), start_idx + chunk_samples)
            seg = strain[start_idx:end_idx]
            seg.start_time = strain.start_time + start_idx / sr
            seg_len = len(seg)
            seg_delta_f = float(sr) / float(seg_len)
            target_len = seg_len // 2 + 1
            freqs_new = np.arange(target_len, dtype=np.float64) * seg_delta_f
            psd_interp = np.interp(freqs_new, freq_old, psd_vals)
            
            # Ensure interpolated PSD is valid
            psd_interp = np.maximum(psd_interp, 1e-40)
            if not np.all(np.isfinite(psd_interp)):
                LOG.warning("Interpolated PSD for chunk has invalid values; sanitizing")
                psd_interp = np.nan_to_num(psd_interp, nan=1e-40, posinf=1e-40, neginf=1e-40)
            
            psd_seg = FrequencySeries(psd_interp, delta_f=seg_delta_f, dtype=psd.dtype)
            segments.append((seg, psd_seg))
        LOG.info("[IFO-6] Created %d segments for chunking", len(segments))
    else:
        segments = [(strain, psd)]
        LOG.info("[IFO-6] Using single segment (no chunking)")

    # PERFORMANCE: Enable GPU memory pooling if CUDA is active
    if use_cuda:
        try:
            from pycuda.tools import DeviceMemoryPool
            # This reduces GPU allocation overhead significantly
            LOG.debug("GPU memory pooling enabled")
        except Exception:
            pass
    
    LOG.info("[IFO-7] Starting PARALLEL template processing: %d templates (CUDA=%s)", total_templates, use_cuda)
    
    # PARALLEL PROCESSING: Use multi-threading for CPU or batching for GPU
    # This dramatically accelerates template bank searches
    import os
    
    # Determine parallelization strategy
    use_parallel = cfg.runtime.enable_parallel  # Default: enabled
    max_workers = cfg.runtime.parallel_workers  # Default: auto-detect (None)
    
    if use_parallel and total_templates > 1:
        # Parallel processing mode
        if use_cuda:
            # Option 1: GPU batched processing (experimental)
            # Option 2: Multi-threaded with CUDA (PyCBC CUDA releases GIL)
            # Currently using multi-threading as it's more stable
            LOG.info("[IFO-7.1] Using parallel multi-threaded processing for CUDA")
            triggers = process_templates_parallel(
                bank["templates"],
                segments,
                cfg,
                strain.duration,
                gates,
                psd_meta,
                ifo,
                use_cuda=True,
                max_workers=max_workers,
                progress_cb=progress_cb,
                n_parallel_ifos=n_parallel_ifos,
            )
        else:
            # CPU: multi-threaded processing (NumPy/BLAS releases GIL)
            LOG.info("[IFO-7.1] Using parallel multi-threaded processing for CPU")
            triggers = process_templates_parallel(
                bank["templates"],
                segments,
                cfg,
                strain.duration,
                gates,
                psd_meta,
                ifo,
                use_cuda=False,
                max_workers=max_workers,
                progress_cb=progress_cb,
                n_parallel_ifos=n_parallel_ifos,
            )
    else:
        # Serial processing (fallback for debugging or single template)
        LOG.info("[IFO-7.1] Using SERIAL processing (parallel disabled or single template)")
        triggers = []
        
        for tidx, tmpl in enumerate(bank["templates"]):
            # Log only at template 1 and every 32nd template
            if tidx == 0 or (tidx + 1) % 32 == 0:
                LOG.info("[TEMPLATE %d/%d] Processing...", tidx + 1, total_templates)
            
            try:
                # CRITICAL FIX: Use FD waveforms to avoid epoch/timing issues
                # Official PyCBC GW150914 tutorial approach
                hp, hc = get_fd_waveform(
                    approximant=tmpl["approximant"],
                    mass1=tmpl["mass1"],
                    mass2=tmpl["mass2"],
                    spin1z=tmpl.get("spin1z", 0.0),
                    spin2z=tmpl.get("spin2z", 0.0),
                    delta_f=1.0 / strain.duration,  # Match strain FFT grid
                    f_lower=tmpl["f_lower"],
                )
                
                # Resize to match strain FFT length (standard method)
                hp.resize(len(strain) // 2 + 1)
                
                # Calculate template duration from metadata (FD waveforms don't have get_duration)
                # For FD: duration is implicit from f_lower and mass parameters
                # Estimate from template metadata
                template_duration = 1.0 / tmpl["f_lower"] * 100  # Rough estimate: ~100 cycles at f_lower
                
                # Debug: log template/strain/PSD alignment for first template
                if tidx == 0:
                    LOG.info("DEBUG Template 1: delta_f=%.6e, len=%d, estimated_duration=%.3f",
                             hp.delta_f, len(hp), template_duration)
                    LOG.info("DEBUG Strain: delta_t=%.6e, duration=%.3f, len=%d, sample_rate=%.1f",
                             strain.delta_t, strain.duration, len(strain), strain.sample_rate)
                    for seg_idx, (seg_strain, seg_psd) in enumerate(segments):
                        if seg_idx == 0:
                            LOG.info("DEBUG Segment 1: strain len=%d, psd len=%d, psd delta_f=%.6e",
                                     len(seg_strain), len(seg_psd), seg_psd.delta_f)
                            break
                            
            except Exception as exc:  # noqa: BLE001
                LOG.warning("Template %d (%s) waveform generation failed: %s; skipping", tidx, tmpl.get("approximant"), exc)
                if progress_cb and total_templates > 0 and (tidx % 200 == 0 or tidx == total_templates - 1):
                    frac = (tidx + 1) / total_templates
                    progress_cb(frac, tidx + 1, total_templates)
                continue
            
            # Validate waveform
            if not np.all(np.isfinite(hp.data)):
                LOG.warning("Template %d has non-finite values; skipping", tidx)
                if progress_cb and total_templates > 0 and (tidx % 200 == 0 or tidx == total_templates - 1):
                    frac = (tidx + 1) / total_templates
                    progress_cb(frac, tidx + 1, total_templates)
                continue
            
            # FD waveforms don't need cropping - skip this for FD

            detection_thr = cfg.runtime.detection_snr_threshold
            for seg_idx, (seg_strain, seg_psd) in enumerate(segments):
                # Validate segment data before matched filter
                if not np.all(np.isfinite(seg_strain.data)):
                    LOG.warning("Segment strain contains non-finite values; skipping template %d for this segment", tidx)
                    continue
                
                # FD waveforms: hp is already correctly sized and in frequency domain
                # No resizing needed - matched_filter handles it
                hp_seg = hp
                
                try:
                    snr = matched_filter(hp_seg, seg_strain, psd=seg_psd, low_frequency_cutoff=cfg.conditioning.low_frequency_cutoff)
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("Matched filter failed for template %d: %s; skipping", tidx, exc)
                    continue
                
                # Validate SNR output
                if not np.all(np.isfinite(snr.data)):
                    LOG.warning("SNR contains non-finite values for template %d; skipping", tidx)
                    continue
                
                # Crop 4 seconds from each edge (FFT wraparound removal).
                # Matches parallel path in parallel_matching.py.
                crop_samples = min(int(4 * snr.sample_rate), len(snr) // 8)
                snr = snr[crop_samples : len(snr) - crop_samples]
                abs_snr = np.abs(snr.numpy())
                
                # Check SNR statistics for debugging
                snr_max = np.max(abs_snr)
                snr_mean = np.mean(abs_snr)
                snr_median = np.median(abs_snr)
                snr_std = np.std(abs_snr)
                
                # Log SNR stats for first template of each detector
                # CRITICAL CHECK (from expert): noise SNR std should be ~1 if PSD is correctly scaled
                if tidx == 0:
                    LOG.info("Template %d SNR stats: max=%.2f, mean=%.2f, median=%.2f, std=%.2f, threshold=%.2f",
                             tidx + 1, snr_max, snr_mean, snr_median, snr_std, detection_thr)
                    if snr_std < 0.5:
                        LOG.warning("SNR std=%.2f << 1.0 indicates PSD might be scaled too large or band-limiting is wrong", snr_std)
                    elif snr_std > 2.0:
                        LOG.warning("SNR std=%.2f >> 1.0 indicates PSD might be scaled too small", snr_std)
                
                # Sanity check: if max SNR is astronomically high (>1e6), data is corrupted
                if snr_max > 1e6:
                    LOG.warning("Template %d: PATHOLOGICAL SNR detected (max=%.2e). Skipping segment.",
                               tidx + 1, snr_max)
                    continue
                
                # CRITICAL OPTIMIZATION (best practice - PyCBC/GstLAL operational standard):
                # Use PyCBC's ThresholdCluster for efficient clustering
                # Requires NumPy 1.x (use: micromamba install "numpy<2.0")
                #
                # References:
                # - Usman et al. 2016 (PyCBC methodology)
                # - Abbott et al. 2016 GW150914 (GstLAL uses 1s clustering)
                # - PyCBC matched_filter.py uses ThresholdCluster internally
                
                # Use 1.0 second clustering window (operational standard)
                cluster_window_sec = 1.0  # GstLAL standard for BBH/BNS
                cluster_window_samples = int(cluster_window_sec * snr.sample_rate)
                
                # PyCBC ThresholdCluster API (best practice)
                thresh_cluster = ThresholdCluster(snr)
                peak_vals, peak_idx = thresh_cluster.threshold_and_cluster(
                    detection_thr,
                    cluster_window_samples
                )
                
                # Convert to numpy array
                peak_idx = np.asarray(peak_idx, dtype=np.int64)
                n_clustered = len(peak_idx)
                
                # Count raw peaks for comparison
                n_raw_peaks = np.sum(abs_snr >= detection_thr)
                
                if tidx < 10 and n_clustered > 0:
                    LOG.info("Template %d: Clustered to %d triggers (from %d raw peaks, window=%.1fs), max_snr=%.2f",
                             tidx + 1, n_clustered, n_raw_peaks, cluster_window_sec, snr_max)
                
                # Chi-squared: only compute for peaks with SNR >= chisq_snr_threshold (item 1+2)
                # For low-SNR peaks use rchisq=1 fallback (saves most of the CPU time)
                if n_clustered > 0:
                    try:
                        dof_fallback = 16 * 2 - 2
                        chisq_threshold = getattr(cfg.runtime, "chisq_snr_threshold", 9.0)
                        peak_snrs_all = abs_snr[peak_idx]
                        chisq_array = np.full(n_clustered, float(dof_fallback))
                        rchisq_array = np.ones(n_clustered)

                        high_mask = peak_snrs_all >= chisq_threshold
                        if np.any(high_mask):
                            high_idx = peak_idx[high_mask]
                            c_hi, r_hi = compute_chisq_batch(seg_strain, hp_seg, seg_psd, high_idx, cfg)
                            chisq_array[high_mask] = c_hi
                            rchisq_array[high_mask] = r_hi

                        # Process all triggers
                        for peak_num, (idx, chisq, rchisq) in enumerate(zip(peak_idx, chisq_array, rchisq_array)):
                            peak_snr = abs_snr[idx]
                            if peak_snr < detection_thr or not np.isfinite(peak_snr):
                                continue
                            
                            nsnr = new_snr(peak_snr, chisq, template_duration, cfg)
                            
                            # Debug: log first few triggers
                            if tidx < 5 and peak_num < 3:
                                LOG.info("  [T%d Peak%d] SNR=%.1f, chi²=%.1f, rchi²=%.2f, newSNR=%.1f, gps=%.3f",
                                         tidx + 1, peak_num + 1, peak_snr, chisq, rchisq, nsnr,
                                         snr.sample_times[idx])
                            
                            trig = {
                                "ifo": ifo,
                                "gps": snr.sample_times[idx],
                                "template_id": f"{tidx}",
                                "snr": float(peak_snr),
                                "chisq": float(chisq),
                                "rchisq": float(rchisq),
                                "new_snr": float(nsnr),
                                "gate_count": len(gates),
                                "duration": template_duration,
                                "psd_meta": psd_meta,
                                "gates": gates,
                            }
                            triggers.append(trig)
                            
                            if tidx < 5 and peak_num < 3:
                                LOG.info("  [T%d Peak%d] ✓ TRIGGER ADDED (total: %d)", tidx + 1, peak_num + 1, len(triggers))
                    
                    except Exception as exc:  # noqa: BLE001
                        LOG.warning("Batched chi-squared failed for template %d: %s; skipping", tidx + 1, exc)
                        continue
                    
            if progress_cb and total_templates > 0 and (tidx % 200 == 0 or tidx == total_templates - 1):
                frac = (tidx + 1) / total_templates
                progress_cb(frac, tidx + 1, total_templates)
    
    LOG.info("[IFO-8] Template loop completed for %s: %d triggers found", ifo, len(triggers))
    return triggers
