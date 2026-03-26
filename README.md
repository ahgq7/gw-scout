# gw-scout

A compact matched-filter gravitational wave search pipeline built on [PyCBC](https://pycbc.org/) and [LAL](https://wiki.ligo.org/Computing/LALSuite). It fetches open detector data from [GWOSC](https://gwosc.org/), runs a full CBC search, and writes ranked candidates with FAR/IFAR estimates.

**Validated**: correctly recovers GW150914 at GPS 1126259462.4, network SNR ≈ 23.

---

## What it does

1. **Fetches frames** from GWOSC for a configurable GPS window (H1, L1, or both).
2. **Conditions strain**: highpass filter, glitch gating, Welch PSD estimation.
3. **Builds a template bank**: geometric/stochastic metric-based placement or a targeted grid around a known event.
4. **Matched filtering** (parallel via `multiprocessing.Pool` with `fork`): each worker inherits strain and PSD via copy-on-write — no pickle overhead.
5. **Chi-squared veto** (Allen χ²): only computed for peaks with SNR ≥ 9 to save CPU.
6. **Coincidence**: H1+L1 trigger pairs within 15 ms, same template (configurable).
7. **Background estimation**: time slides with signal excision — loud foreground triggers are removed before slides to prevent contamination.
8. **Candidates**: ranked by FAR/IFAR, GPS-clustered, cross-matched against the GWOSC event catalog.

---

## Installation

Requires Python ≥ 3.11. PyCBC and LAL are the heavy dependencies; the rest is pure Python.

```bash
# Recommended: conda/mamba environment
mamba create -n gw python=3.11
mamba activate gw
pip install -e ".[dev]"
```

> **macOS note**: PyCBC's LAL binaries work on Apple Silicon via conda-forge. CUDA support is Linux-only and not required.

---

## Quick start — recover GW150914

```bash
gwsearch run \
  --start-gps 1126259000 \
  --block-duration 512 \
  --quick-test-gw150914 \
  --detection-snr-threshold 8.0
```

This downloads ~300 MB of O1 frame data on first run (cached afterwards), builds a targeted bank around GW150914 parameters, and writes results to `output/`.

---

## CLI reference

```
gwsearch run [OPTIONS]

Options:
  --config PATH                 YAML config file (overrides all CLI flags)
  --ifos TEXT                   Detectors, e.g. H1 L1   [default: H1 L1]
  --start-gps FLOAT             GPS start time
  --block-duration FLOAT        Duration of one search block (s)  [default: 512]
  --detection-snr-threshold FLOAT  [default: 8.0]
  --enable-parallel / --no-parallel  Multiprocess template matching  [default: on]
  --parallel-workers INT        Worker count (default: cpu_count)
  --quick / --no-quick          Small template bank for fast testing
  --quick-test-gw150914         Targeted bank around GW150914 parameters
```

---

## Configuration

All settings live in [`gwsearch/config.py`](gwsearch/config.py) as Python dataclasses. Key knobs:

| Parameter | Default | Notes |
|---|---|---|
| `runtime.detection_snr_threshold` | 8.0 | Single-IFO SNR cut |
| `runtime.chisq_snr_threshold` | 9.0 | Only compute χ² above this |
| `runtime.enable_parallel` | True | Uses `multiprocessing.Pool` (fork) |
| `runtime.parallel_workers` | auto | Defaults to `cpu_count` |
| `bank.quick_test_gw150914` | False | Targeted bank for GW150914 recovery |
| `bank.use_geometric_bank` | True | Metric-based bank placement |
| `bank.min_match` | 0.97 | Minimal match (1 − mismatch) |
| `background.quick_slides` | 50 | Time slides for background (quick mode) |
| `background.production_slides` | 200 | Time slides for production |

---

## Architecture

```
gwsearch/
├── cli.py               # Typer CLI entry point
├── main.py              # Main search loop (fetch → filter → coincide → rank)
├── config.py            # All configuration dataclasses
├── gwosc_io.py          # GWOSC frame fetching and caching
├── conditioning.py      # Strain conditioning (highpass, PSD)
├── gating.py            # Glitch gating
├── bank.py              # Simple template bank (mass grid)
├── bank_geometric.py    # Geometric/stochastic bank placement
├── bank_targeted.py     # Targeted bank for validation
├── single_ifo.py        # Per-IFO processing orchestration
├── parallel_matching.py # Multiprocessing matched filter (fork + globals)
├── ranking.py           # Chi-squared veto, newSNR reweighting
├── coincidence.py       # H1+L1 coincidence detection
├── background.py        # Time-slide background + signal excision
├── report.py            # Candidate building, GPS clustering, output
├── known_events.py      # GWOSC catalog cross-matching
└── state.py             # SQLite run state and progress tracking
```

---

## Output

Results are written to `output/` after each block:

- `candidates.jsonl` — one JSON record per candidate (full metadata)
- `candidates.csv` — summary table (GPS, IFOs, network SNR, FAR, IFAR, known event match)
- `run_config.json` — full config snapshot for reproducibility

### Example candidate (GW150914)

```json
{
  "gps": 1126259462.4246,
  "ifos": ["H1", "L1"],
  "network_stat": 22.99,
  "far_hz": 3.2e-8,
  "ifar_days": 362,
  "known_event": {"name": "GW150914", "gps": 1126259462.4}
}
```

---

## Performance

Tested on a 10-core Apple M-series Mac with 198 templates (targeted GW150914 bank):

| Mode | Wall time | CPU usage |
|---|---|---|
| Serial | ~4 min 20 s | 100% |
| Parallel (10 workers) | ~1 min 10 s | ~750% |

For a full geometric bank (~5 000 templates), expect ~25–35 min per 512 s block at 100% match threshold.

---

## Known limitations

- **Single-block background**: IFAR estimates are based on time slides within one data block. For scientifically meaningful IFAR values, run on ≥ 1 week of data.
- **No multi-IFO vetoes**: only H1+L1 coincidence; Virgo/KAGRA support exists in `coincidence_multi_ifo.py` but is not integrated into the main loop.
- **No calibration corrections**: uses raw GWOSC frames without calibration line removal.

---

## Tests

```bash
pytest
```

42 unit tests covering background estimation, coincidence logic, chi-squared reweighting, newSNR, template bank generation, and state management.

---

## License

Apache 2.0
