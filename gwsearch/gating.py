from __future__ import annotations

from typing import List, Tuple

import numpy as np


def find_gates(strain, cfg) -> List[Tuple[int, int]]:
    data = strain.numpy()
    sr = strain.sample_rate
    gates: List[Tuple[int, int]] = []
    thr = cfg.gating.snr_threshold * np.std(data)
    idx = np.where(np.abs(data) > thr)[0]
    if idx.size == 0:
        return gates
    pad = int(cfg.gating.pad * sr)
    max_dur = int(cfg.gating.max_duration * sr)
    start = idx[0]
    end = idx[0]
    for i in idx[1:]:
        if i - end <= pad:
            end = i
        else:
            gates.append((max(start - pad, 0), min(end + pad, len(data) - 1)))
            start = i
            end = i
    gates.append((max(start - pad, 0), min(end + pad, len(data) - 1)))
    gates = [(s, min(s + max_dur, e)) for s, e in gates]
    return gates


def apply_gates(strain, gates: List[Tuple[int, int]]) -> None:
    if not gates:
        return
    data = strain.numpy()
    for s, e in gates:
        w = np.linspace(1, 0, e - s + 1)
        data[s : e + 1] *= w
    strain.data = data
