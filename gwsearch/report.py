from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Any
import numpy as np

try:
    import orjson
except Exception:  # noqa: BLE001
    orjson = None

from .detchar import rerank

LOG = logging.getLogger(__name__)


def _convert_numpy_types(obj: Any) -> Any:
    """
    Recursively convert NumPy types to Python native types for JSON serialization.
    NumPy 2.0 compatibility fix.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_convert_numpy_types(item) for item in obj]
    else:
        return obj


def _json_dumps(obj) -> str:
    """
    JSON serialization with NumPy type support (NumPy 2.0 compatible).
    """
    # Convert all NumPy types to Python native types
    obj_converted = _convert_numpy_types(obj)
    
    if orjson:
        # orjson is faster but stricter - use native Python types
        return orjson.dumps(obj_converted).decode()
    return json.dumps(obj_converted, default=str)


def build_candidates(
    coincs: List[Dict],
    background: Dict,
    cfg,
    template_bank: Dict,
    known_events=None,
) -> List[Dict]:
    """
    Build candidate dictionaries including FAR/IFAR estimation, per-IFO details,
    PSD metadata, detchar rank, bank/config hashes, and known-event matches.
    
    CRITICAL FIX: Robust NaN/inf handling for network_stat
    """
    samples = background.get("samples") or []
    far_base = background.get("far_hz")
    trials_duration = background.get("trials_duration")
    candidates: List[Dict] = []
    for c in coincs:
        stat = c.get("network_stat")
        
        # CRITICAL FIX: Validate network_stat
        if stat is None or not np.isfinite(stat):
            LOG.warning("Skipping candidate with invalid network_stat=%s at GPS %.3f",
                       stat, c.get("gps", 0))
            continue
        if samples and trials_duration and trials_duration > 0:
            # FAR = (number of background coincidences louder than candidate) / trials_duration
            # +1 correction for finite-sample statistics (Farr et al. 2014)
            louder = sum(1 for s in samples if s >= stat)
            far_hz = (louder + 1) / trials_duration
        else:
            far_hz = far_base
        ifar_days = (1.0 / far_hz) / 86400.0 if far_hz else None
        known = known_events.match_event(c["gps"]) if known_events else None
        detchar_rank = rerank(c)
        cand = {
            "gps": float(c["gps"]),
            "ifos": c["ifos"],
            "dt": c["dt"],
            "template_id": c["template_id"],
            "per_ifo": c["per_ifo"],
            "network_stat": stat,
            "far_hz": far_hz,
            "ifar_days": ifar_days,
            "background_trials_duration": trials_duration,
            "known_event": known,
            "bank_hash": template_bank.get("hash"),
            "config_hash": cfg.config_hash,
            "notches": [n.__dict__ for n in cfg.conditioning.notch_list],
            "gating": {
                "snr_threshold": cfg.gating.snr_threshold,
            },
            "conditioning": {
                "sample_rate": cfg.conditioning.sample_rate,
                "highpass_freq": cfg.conditioning.highpass_freq,
                "low_frequency_cutoff": cfg.conditioning.low_frequency_cutoff,
                "high_frequency_cutoff": cfg.conditioning.high_frequency_cutoff,
            },
            "runtime": {
                "detection_snr_threshold": cfg.runtime.detection_snr_threshold,
                "coincidence_same_template": cfg.runtime.coincidence_same_template,
                "cluster_window": cfg.background.cluster_window,
            },
            "detchar_rank": detchar_rank,
        }
        candidates.append(cand)
    candidates.sort(key=lambda x: (x["far_hz"] if x["far_hz"] is not None else 1e9, -x["network_stat"]))

    # Cluster candidates within a 1-second GPS window: keep the best SNR per cluster
    cluster_window = 1.0
    clustered: List[Dict] = []
    for cand in candidates:
        gps = cand["gps"]
        merged = False
        for existing in clustered:
            if abs(existing["gps"] - gps) < cluster_window:
                # Keep the one with better (lower) FAR / higher SNR
                if cand["network_stat"] > existing["network_stat"]:
                    existing.update(cand)
                merged = True
                break
        if not merged:
            clustered.append(dict(cand))

    LOG.debug("Candidates before GPS clustering: %d, after: %d", len(candidates), len(clustered))
    return clustered


def write_outputs(output_dir: Path, candidates: List[Dict], cfg) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl = output_dir / "candidates.jsonl"
    csv_path = output_dir / "candidates.csv"
    with open(jsonl, "w", encoding="utf-8") as jf, open(csv_path, "w", newline="", encoding="utf-8") as cf:
        fieldnames = [
            "run_id",
            "block_id",
            "gps",
            "ifos",
            "dt",
            "template_id",
            "network_stat",
            "far_hz",
            "ifar_days",
            "known_event",
            "detchar_rank",
            "bank_hash",
            "config_hash",
        ]
        writer = csv.DictWriter(cf, fieldnames=fieldnames)
        writer.writeheader()
        for cand in candidates:
            jf.write(_json_dumps(cand) + "\n")
            row = {k: cand.get(k) for k in fieldnames}
            writer.writerow(row)
    (output_dir / "run_config.json").write_text(cfg.to_json())
