from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np

LOG = logging.getLogger(__name__)


def compute_background(
    per_ifo_triggers: Dict[str, List[Dict]],
    slides: int,
    window: float,
    slide_spacing: float,
    data_duration: float = None,
    excise_gps: List[float] = None,
    excise_window: float = 4.0,
    require_same_template: bool = False,
) -> Dict:
    """
    Time-slide background using symmetric offsets spaced by slide_spacing (seconds).

    Args:
        per_ifo_triggers: Per-IFO trigger lists from matched filtering.
        slides: Number of time-slide offsets (both signs applied, so 2*slides total slides).
        window: Coincidence window (seconds).
        slide_spacing: Step between slide offsets (seconds).
        data_duration: Duration of analyzed data (seconds). Used to compute trials_duration
            correctly as 2*slides*data_duration. If None, falls back to 2*slides*window
            (incorrect but preserved for backward compatibility).
        excise_gps: GPS times of loud foreground events to excise from background triggers
            before computing slides. Prevents signal contamination of the background.
        excise_window: Half-width of the excision window around each excise_gps (seconds).
        require_same_template: If True, only count background coincidences between triggers
            with the same template_id. Must match the foreground coincidence criterion.

    Returns:
        Dict with far_hz, ifar_days, samples, trials_duration, slide_spacing, slides.
    """
    if len(per_ifo_triggers) < 2:
        return {"far_hz": None, "ifar_days": None, "samples": [], "trials_duration": 0.0, "slide_spacing": slide_spacing, "slides": slides}

    # Excise triggers near loud foreground events (signal contamination prevention).
    # Real pipelines (PyCBC, GstLAL) always do this before background estimation.
    if excise_gps:
        cleaned: Dict[str, List[Dict]] = {}
        for ifo, trigs in per_ifo_triggers.items():
            cleaned[ifo] = [
                t for t in trigs
                if all(abs(t["gps"] - g) >= excise_window for g in excise_gps)
            ]
        n_excised = sum(len(per_ifo_triggers[ifo]) - len(cleaned[ifo]) for ifo in cleaned)
        if n_excised:
            LOG.info("Signal excision: removed %d triggers within %.1fs of %d loud foreground event(s)",
                     n_excised, excise_window, len(excise_gps))
        per_ifo_triggers = cleaned

    ifos = list(per_ifo_triggers.keys())
    a_trigs = per_ifo_triggers[ifos[0]]
    b_trigs = per_ifo_triggers[ifos[1]]
    coincidences = []
    offsets = [k * slide_spacing for k in range(1, slides + 1)]
    dt_values = [sign * offset for offset in offsets for sign in (-1.0, 1.0)]

    if require_same_template:
        # Group by template_id and only search within matching templates.
        # This avoids O(N_H1 × N_L1) by reducing to sum_t(N_H1_t × N_L1_t).
        a_by_tid: Dict[str, dict] = defaultdict(lambda: {"gps": [], "snr": []})
        b_by_tid: Dict[str, dict] = defaultdict(lambda: {"gps": [], "snr": []})
        for t in a_trigs:
            a_by_tid[t["template_id"]]["gps"].append(t["gps"])
            a_by_tid[t["template_id"]]["snr"].append(t["new_snr"])
        for t in b_trigs:
            b_by_tid[t["template_id"]]["gps"].append(t["gps"])
            b_by_tid[t["template_id"]]["snr"].append(t["new_snr"])
        common_tids = set(a_by_tid.keys()) & set(b_by_tid.keys())
        for tid in common_tids:
            h_gps = np.array(a_by_tid[tid]["gps"])
            h_snr = np.array(a_by_tid[tid]["snr"])
            l_gps = np.array(b_by_tid[tid]["gps"])
            l_snr = np.array(b_by_tid[tid]["snr"])
            h_snr2 = h_snr ** 2
            l_snr2 = l_snr ** 2
            for dt in dt_values:
                # shape (N_h, N_l) — find pairs within window
                diff = np.abs(h_gps[:, None] - (l_gps[None, :] + dt))
                mask = diff < window
                if not np.any(mask):
                    continue
                h_idx, l_idx = np.where(mask)
                stats = np.sqrt(h_snr2[h_idx] + l_snr2[l_idx])
                coincidences.extend(stats.tolist())
    else:
        # All-pairs loop (no template restriction)
        for offset in offsets:
            for sign in (-1.0, 1.0):
                dt = sign * offset
                for h in a_trigs:
                    for l in b_trigs:
                        if abs(h["gps"] - (l["gps"] + dt)) < window:
                            stat = float(np.sqrt(h["new_snr"] ** 2 + l["new_snr"] ** 2))
                            coincidences.append(stat)

    # Correct trials_duration: total background live-time = 2 * slides * data_duration.
    # The coincidence window (cluster_window) is NOT the data duration — using it here
    # overestimates the FAR by a factor of data_duration/window (~1000x for a 512s block).
    if data_duration is not None and data_duration > 0:
        trials_duration = float(2 * slides * data_duration)
    else:
        # Fallback: old (incorrect) formula kept for backward compatibility
        trials_duration = float(2 * slides * window)

    if not coincidences:
        return {"far_hz": None, "ifar_days": None, "samples": [], "trials_duration": trials_duration, "slide_spacing": slide_spacing, "slides": slides}

    rate = len(coincidences) / trials_duration if trials_duration > 0 else None
    if rate is None or rate == 0:
        ifar_days = None
    else:
        ifar_days = (1.0 / rate) / 86400.0
    return {
        "far_hz": rate,
        "ifar_days": ifar_days,
        "samples": coincidences,
        "trials_duration": trials_duration,
        "slide_spacing": slide_spacing,
        "slides": slides,
    }
