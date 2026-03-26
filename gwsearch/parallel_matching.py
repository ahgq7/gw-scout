"""
Parallel template matching for gravitational wave searches.

Uses multiprocessing.Pool with an initializer so that strain/PSD data
is transferred to workers ONCE (not once per template). Each worker process
has its own LAL/PyCBC state, avoiding thread-safety issues.
"""

from __future__ import annotations

import logging
import os
import numpy as np
from multiprocessing import Pool
from typing import List, Dict, Any, Tuple, Optional

from .config import SearchConfig
from .ranking import new_snr

LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worker process global state
#
# With 'fork' start method, child processes inherit the parent's entire memory
# space (copy-on-write). We set these globals BEFORE forking so workers see
# them without any pickling/serialization overhead.
# ---------------------------------------------------------------------------

_W_SEGMENTS = None   # List[Tuple[TimeSeries, FrequencySeries]]
_W_CFG = None        # SearchConfig


def _worker_noop_init() -> None:
    """No-op initializer: fork already copies parent globals."""
    pass


def _worker_spawn_init(segments, cfg) -> None:
    """Initializer for spawn context: set globals from passed args."""
    global _W_SEGMENTS, _W_CFG
    _W_SEGMENTS = segments
    _W_CFG = cfg


def _worker_spawn_init_omp(seg_data, cfg, omp_threads: int) -> None:
    """Spawn initializer: limit OpenMP/BLAS threads per worker, reconstruct segments."""
    import os
    os.environ["OMP_NUM_THREADS"] = str(omp_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(omp_threads)
    os.environ["MKL_NUM_THREADS"] = str(omp_threads)
    global _W_SEGMENTS, _W_CFG
    from pycbc.types import TimeSeries, FrequencySeries
    _W_SEGMENTS = [
        (TimeSeries(s_arr, delta_t=dt, epoch=epoch),
         FrequencySeries(p_arr, delta_f=df))
        for s_arr, dt, epoch, p_arr, df in seg_data
    ]
    _W_CFG = cfg


# ---------------------------------------------------------------------------
# Core per-template processing (runs inside worker process)
# ---------------------------------------------------------------------------

def _process_template_task(task: tuple) -> List[Dict]:
    """
    Process one template against all segments.
    Called inside a Pool worker; uses global _W_SEGMENTS / _W_CFG.
    """
    tidx, tmpl, strain_duration, gates, psd_meta, ifo, total_templates = task

    from pycbc.filter import matched_filter
    from pycbc.waveform import get_fd_waveform
    from pycbc.events.eventmgr import ThresholdCluster

    cfg = _W_CFG
    segments = _W_SEGMENTS
    triggers = []

    try:
        hp, _hc = get_fd_waveform(
            approximant=tmpl["approximant"],
            mass1=tmpl["mass1"],
            mass2=tmpl["mass2"],
            spin1z=tmpl.get("spin1z", 0.0),
            spin2z=tmpl.get("spin2z", 0.0),
            delta_f=1.0 / strain_duration,
            f_lower=tmpl["f_lower"],
        )
        hp.resize(len(segments[0][0]) // 2 + 1)
        template_duration = 1.0 / tmpl["f_lower"] * 100
    except Exception as exc:
        LOG.warning("Template %d waveform failed: %s", tidx, exc)
        return triggers

    if not np.all(np.isfinite(hp.data)):
        return triggers

    detection_thr = cfg.runtime.detection_snr_threshold
    chisq_thr = cfg.runtime.chisq_snr_threshold
    dof = 16 * 2 - 2  # 30

    for seg_strain, seg_psd in segments:
        if not np.all(np.isfinite(seg_strain.data)):
            continue
        try:
            snr = matched_filter(
                hp, seg_strain,
                psd=seg_psd,
                low_frequency_cutoff=cfg.conditioning.low_frequency_cutoff,
            )
        except Exception as exc:
            LOG.warning("Matched filter failed for template %d: %s", tidx, exc)
            continue

        if not np.all(np.isfinite(snr.data)):
            continue

        # Crop 4 seconds from each edge to remove FFT wraparound artifacts.
        # The standard PyCBC GW150914 tutorial uses this approach.
        # The previous 25%-75% crop removed 144s from each side of a 576s block,
        # which pushed GW150914 (at GPS+462s) outside the valid window.
        crop_samples = min(int(4 * snr.sample_rate), len(snr) // 8)
        snr = snr[crop_samples : len(snr) - crop_samples]
        abs_snr = np.abs(snr.numpy())

        if np.max(abs_snr) > 1e6:
            continue

        cluster_samples = int(1.0 * snr.sample_rate)
        thresh_cluster = ThresholdCluster(snr)
        peak_vals, peak_idx = thresh_cluster.threshold_and_cluster(detection_thr, cluster_samples)
        peak_idx = np.asarray(peak_idx, dtype=np.int64)

        if len(peak_idx) == 0:
            continue

        peak_snrs = abs_snr[peak_idx]

        # Chi-squared: only for peaks above chisq_thr (item 1+2)
        chisq_array = np.full(len(peak_idx), float(dof))
        rchisq_array = np.ones(len(peak_idx))

        high_snr_mask = peak_snrs >= chisq_thr
        if np.any(high_snr_mask):
            high_idx = peak_idx[high_snr_mask]
            try:
                from .ranking import compute_chisq_batch
                c_arr, r_arr = compute_chisq_batch(seg_strain, hp, seg_psd, high_idx, cfg)
                chisq_array[high_snr_mask] = c_arr
                rchisq_array[high_snr_mask] = r_arr
            except Exception as exc:
                LOG.warning("Chi-sq failed for template %d: %s", tidx, exc)

        for i, (idx, chisq, rchisq) in enumerate(zip(peak_idx, chisq_array, rchisq_array)):
            peak_snr = peak_snrs[i]
            if not np.isfinite(peak_snr) or peak_snr < detection_thr:
                continue
            nsnr = new_snr(peak_snr, chisq, template_duration, cfg)
            triggers.append({
                "ifo": ifo,
                "gps": float(snr.sample_times[idx]),
                "template_id": str(tidx),
                "snr": float(peak_snr),
                "chisq": float(chisq),
                "rchisq": float(rchisq),
                "new_snr": float(nsnr),
                "gate_count": len(gates),
                "duration": template_duration,
                "psd_meta": psd_meta,
                "gates": gates,
            })

    return triggers


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def process_templates_parallel(
    templates: List[Dict[str, Any]],
    segments: List[Tuple],
    cfg: SearchConfig,
    strain_duration: float,
    gates: List,
    psd_meta: Dict,
    ifo: str,
    use_cuda: bool = False,
    max_workers: Optional[int] = None,
    progress_cb=None,
) -> List[Dict]:
    """
    Process templates in parallel using multiprocessing.Pool.

    Strain/PSD is passed once to each worker via the Pool initializer,
    avoiding repeated pickling overhead. Each worker process has its own
    LAL/PyCBC state, so thread-safety issues are avoided.
    """
    total_templates = len(templates)
    if total_templates == 0:
        return []

    n_workers = max_workers or (os.cpu_count() or 1)
    LOG.info("Multiprocessing: %d templates across %d worker processes", total_templates, n_workers)

    # Set globals BEFORE forking — workers inherit via copy-on-write, zero pickle cost
    global _W_SEGMENTS, _W_CFG
    _W_SEGMENTS = list(segments)
    _W_CFG = cfg

    tasks = [
        (tidx, tmpl, strain_duration, gates, psd_meta, ifo, total_templates)
        for tidx, tmpl in enumerate(templates)
    ]

    all_triggers: List[Dict] = []

    import sys
    if sys.platform == "darwin":
        # macOS: fork copies globals to workers, no pickle overhead
        ctx = __import__("multiprocessing").get_context("fork")
        with ctx.Pool(processes=n_workers, initializer=_worker_noop_init) as pool:
            for completed, result in enumerate(pool.imap_unordered(_process_template_task, tasks, chunksize=4)):
                all_triggers.extend(result)
                if progress_cb and (completed % 32 == 0 or completed == total_templates - 1):
                    progress_cb((completed + 1) / total_templates, completed + 1, total_templates)
    else:
        # Linux: spawn N workers, each limited to cpu_count//N OpenMP threads.
        # This avoids fork+LAL deadlocks while still using all cores.
        # e.g. 20 cores → 5 workers × 4 OMP threads each.
        cpu_count = os.cpu_count() or 1
        spawn_workers = min(max(2, cpu_count // 4), 8)
        omp_per_worker = max(1, cpu_count // spawn_workers)
        LOG.info("Linux: spawn %d workers × %d OMP threads (total %d cores)",
                 spawn_workers, omp_per_worker, spawn_workers * omp_per_worker)
        ctx = __import__("multiprocessing").get_context("spawn")
        seg_data = [(s.numpy().copy(), float(s.delta_t), float(s.start_time),
                     p.numpy().copy(), float(p.delta_f)) for s, p in _W_SEGMENTS]
        with ctx.Pool(
            processes=spawn_workers,
            initializer=_worker_spawn_init_omp,
            initargs=(seg_data, cfg, omp_per_worker),
        ) as pool:
            for completed, result in enumerate(pool.imap_unordered(_process_template_task, tasks, chunksize=4)):
                all_triggers.extend(result)
                if progress_cb and (completed % 32 == 0 or completed == total_templates - 1):
                    progress_cb((completed + 1) / total_templates, completed + 1, total_templates)

    LOG.info("Parallel done: %d triggers from %d templates", len(all_triggers), total_templates)
    return all_triggers


def process_templates_batched_gpu(
    templates, segments, cfg, strain_duration, gates, psd_meta, ifo,
    batch_size=8, progress_cb=None,
):
    """GPU batched stub — falls back to serial (no CUDA on this machine)."""
    from .single_ifo import _process_serial  # avoid circular import
    raise NotImplementedError("GPU batched processing not available without CUDA")
