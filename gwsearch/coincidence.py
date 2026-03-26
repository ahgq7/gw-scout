from __future__ import annotations

import itertools
from typing import Dict, List

import numpy as np


def find_coincidences(
    per_ifo_triggers: Dict[str, List[Dict]],
    window: float,
    require_same_template: bool = True,
    chirp_tolerance: float = 0.1,
) -> List[Dict]:
    """
    Symmetric pairwise coincidence across all IFO pairs.
    
    CRITICAL FIX: Robust NaN/inf handling for network statistic.
    """
    import logging
    LOG = logging.getLogger(__name__)
    
    if len(per_ifo_triggers) < 2:
        return []
    ifos = list(per_ifo_triggers.keys())
    coincs: List[Dict] = []
    for a_idx in range(len(ifos) - 1):
        for b_idx in range(a_idx + 1, len(ifos)):
            a = ifos[a_idx]
            b = ifos[b_idx]
            a_trigs = per_ifo_triggers.get(a) or []
            b_trigs = per_ifo_triggers.get(b) or []
            if not a_trigs or not b_trigs:
                continue
            for t1 in a_trigs:
                for t2 in b_trigs:
                    if abs(t1["gps"] - t2["gps"]) > window:
                        continue
                    if require_same_template and t1["template_id"] != t2["template_id"]:
                        continue
                    
                    # CRITICAL FIX: Robust network statistic calculation
                    snr1 = t1.get("new_snr", 0.0)
                    snr2 = t2.get("new_snr", 0.0)
                    
                    # Validate inputs
                    if not np.isfinite(snr1):
                        LOG.warning("Invalid new_snr for %s trigger at GPS %.3f: %s",
                                   a, t1.get("gps", 0), snr1)
                        snr1 = 0.0
                    
                    if not np.isfinite(snr2):
                        LOG.warning("Invalid new_snr for %s trigger at GPS %.3f: %s",
                                   b, t2.get("gps", 0), snr2)
                        snr2 = 0.0
                    
                    # Compute network statistic (quadrature sum)
                    try:
                        net_stat = float(np.sqrt(snr1**2 + snr2**2))
                        
                        if not np.isfinite(net_stat):
                            LOG.warning("Network stat computation gave non-finite: snr1=%.2f, snr2=%.2f",
                                       snr1, snr2)
                            continue  # Skip this coincidence
                            
                    except (ValueError, FloatingPointError) as e:
                        LOG.warning("Network stat computation failed: %s", e)
                        continue
                    
                    coincs.append(
                        {
                            "ifos": [a, b],
                            "gps": float((t1["gps"] + t2["gps"]) / 2.0),
                            "dt": float(t1["gps"] - t2["gps"]),
                            "template_id": t1["template_id"],
                            "per_ifo": {a: t1, b: t2},
                            "network_stat": net_stat,
                        }
                    )
    return coincs
