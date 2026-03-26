from __future__ import annotations

from typing import Dict, List


def compute_features(candidate: Dict) -> Dict:
    per_ifo = candidate.get("per_ifo", {})
    rchisq_vals = []
    for info in per_ifo.values():
        rchisq_vals.append(info.get("rchisq", 1.0))
    avg_rchisq = sum(rchisq_vals) / len(rchisq_vals) if rchisq_vals else 1.0
    return {"avg_rchisq": avg_rchisq}


def rerank(candidate: Dict) -> float:
    feats = compute_features(candidate)
    net = candidate.get("network_stat", 0.0)
    return float(net / (1.0 + feats["avg_rchisq"]))
