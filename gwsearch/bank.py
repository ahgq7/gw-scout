from __future__ import annotations

import hashlib
import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from .config import BankConfig

Template = Dict[str, Any]


def _grid(min_v: float, max_v: float, step: float) -> List[float]:
    vals: List[float] = []
    v = min_v
    while v <= max_v + 1e-9:
        vals.append(round(v, 3))
        v += step
    return vals


def _bank_hash(templates: List[Template]) -> str:
    payload = json.dumps(templates, sort_keys=True, default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def build_bank(cfg: BankConfig) -> List[Template]:
    # Very lightweight approximate grid; for production use pycbc_geom_nonspinbank or similar.
    if cfg.quick:
        step = max(1.0, (cfg.max_mass1 - cfg.min_mass1) / max(cfg.max_templates_quick**0.5, 2))
    else:
        # scale step with mismatch; rough heuristic
        step = max(0.3, math.sqrt(cfg.mismatch) * 5.0)

    m1_vals = _grid(cfg.min_mass1, cfg.max_mass1, step)
    m2_vals = _grid(cfg.min_mass2, cfg.max_mass2, step)

    # Spin grid: use configured values or fall back to zero-spin only
    if cfg.include_spins and hasattr(cfg, "spin_grid_values") and cfg.spin_grid_values:
        spin_vals = list(cfg.spin_grid_values)
    else:
        spin_vals = [0.0]

    templates: List[Template] = []
    for m1, m2 in itertools.product(m1_vals, m2_vals):
        if m1 < m2:
            m1, m2 = m2, m1
        if m1 + m2 < 4.0:
            continue
        for s1z, s2z in itertools.product(spin_vals, spin_vals):
            templates.append(
                {
                    "mass1": m1,
                    "mass2": m2,
                    "spin1z": s1z,
                    "spin2z": s2z,
                    "approximant": cfg.approximant,
                    "f_lower": cfg.f_lower,
                }
            )

    # truncate for quick mode
    if cfg.quick and len(templates) > cfg.max_templates_quick:
        templates = templates[: cfg.max_templates_quick]

    return templates


def load_or_build_bank(cfg: BankConfig) -> Dict[str, Any]:
    cfg.cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cfg.cache_dir / f"bank_{cfg.approximant}_m{cfg.mismatch}_q{cfg.quick}.json"

    if cache_file.exists():
        data = json.loads(cache_file.read_text())
        data["hash"] = data.get("hash") or _bank_hash(data["templates"])
        return data

    templates = build_bank(cfg)
    bank_hash = _bank_hash(templates)
    cfg_dict = {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.__dict__.items()}
    data = {"hash": bank_hash, "templates": templates, "meta": {"count": len(templates), "cfg": cfg_dict}}
    cache_file.write_text(json.dumps(data, indent=2, default=str))
    return data
