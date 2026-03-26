from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional, Sequence, Any


def _json_default(o: Any) -> str:
    if isinstance(o, Path):
        return str(o)
    return str(o)


def _hash_dict(obj: dict) -> str:
    data = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_json_default).encode()
    return hashlib.sha256(data).hexdigest()


@dataclass
class NotchSpec:
    frequency: float
    width: float = 0.5
    q: Optional[float] = None


@dataclass
class GatingConfig:
    snr_threshold: float = 50.0
    pad: float = 0.25
    taper: float = 0.5
    max_duration: float = 2.0


@dataclass
class ConditioningConfig:
    sample_rate: int = 4096
    highpass_freq: float = 15.0
    low_frequency_cutoff: float = 20.0
    high_frequency_cutoff: Optional[float] = None
    psd_estimation: str = "median"
    psd_duration: float = 64.0
    psd_stride: float = 8.0
    notch_list: List[NotchSpec] = field(default_factory=list)
    resample_rate: Optional[int] = None


@dataclass
class BankConfig:
    min_mass1: float = 2.0
    max_mass1: float = 80.0
    min_mass2: float = 2.0
    max_mass2: float = 80.0
    min_spin: float = -0.99
    max_spin: float = 0.99
    mismatch: float = 0.03
    approximant: str = "IMRPhenomD"
    f_lower: float = 20.0
    quick: bool = False
    cache_dir: Path = Path("cache/banks")
    max_templates_quick: int = 256
    # GW150914 validation mode
    quick_test_gw150914: bool = False  # Use targeted bank for GW150914
    exclude_signal_from_psd_sec: float = 0.0  # Skip first N seconds for PSD
    
    # PRODUCTION: Geometric/metric-based bank configuration
    use_geometric_bank: bool = True   # Use geometric bank instead of simple grid
    geometric_method: str = "stochastic"  # pycbc_external|stochastic
    min_match: float = 0.97  # Minimal match (1 - mismatch)
    include_spins: bool = True   # Include aligned-spin templates
    min_spin1z: float = -0.9
    max_spin1z: float = 0.9
    min_spin2z: float = -0.9
    max_spin2z: float = 0.9
    spin_grid_values: list = field(default_factory=lambda: [-0.5, 0.0, 0.5])  # Spin values for simple grid


@dataclass
class BackgroundConfig:
    # Legacy simplified settings
    time_slides: int = 50
    slide_spacing: float = 0.1
    cluster_window: float = 0.5
    max_lag: float = 5.0
    production_slides: int = 200
    quick_slides: int = 50
    min_ifar_days: float = 0.01
    
    # PRODUCTION: Advanced time-slide configuration
    use_advanced_background: bool = False  # Use advanced time-slide estimation
    num_slides: int = 100  # Number of time-slides for FAR estimation
    slide_step: float = 0.1  # Time-slide step (seconds)
    max_slide_offset: float = 100.0  # Maximum offset (seconds)
    circular_slides: bool = True  # Circular vs linear slides
    coincidence_window: float = 0.015  # Coincidence window for H1-L1 (seconds)
    reference_detector: str = "H1"  # Reference detector (not shifted)
    target_far_per_year: float = 1.0  # Target FAR for threshold setting


@dataclass
class PathsConfig:
    output_dir: Path = Path("output")
    cache_dir: Path = Path("cache")
    frames_dir: Path = Path("cache/frames")
    state_db: Path = Path("state/gwsearch.sqlite")
    log_dir: Path = Path("logs")


@dataclass
class KnownEventsConfig:
    cache_file: Path = Path("cache/known_events.json")
    # GWOSC API v2 endpoint - pagination handled automatically via "next" field
    # Using pagesize=100 for optimal performance (fewer requests than default)
    # No "page=1" parameter - let API handle pagination naturally
    api_url: str = "https://gwosc.org/api/v2/event-versions?pagesize=100"


@dataclass
class DashboardConfig:
    enabled: bool = False
    host: str = "127.0.0.1"
    port: int = 7860
    update_interval: float = 5.0


@dataclass
class SearchRuntimeConfig:
    ifos: Sequence[str] = field(default_factory=lambda: ["H1", "L1"])
    start_mode: str = "latest"  # latest|resume|gps
    start_gps: Optional[float] = None
    lookback_sec: float = 7 * 24 * 3600
    follow_latest: bool = False
    poll_interval_sec: float = 60.0  # Check for new data every 60 seconds (was 3600)
    block_duration: float = 512.0
    chunk_duration: float = 128.0
    keep_frames: bool = False
    graceful_stop: bool = True
    stop_file: Optional[Path] = None
    dashboard: DashboardConfig = field(default_factory=DashboardConfig)
    detection_snr_threshold: float = 8.0  # Increased from 5.5 for speed (fewer triggers to process)
    coincidence_same_template: bool = True
    # Known event testing/validation
    known_event_gps: Optional[float] = None  # GPS time of known event (for validation)
    center_event_in_window: bool = False  # Center known event in search window
    # Backward search configuration
    search_direction: str = "forward"  # forward|backward - search forward in time or backward in time
    backward_stop_gps: Optional[float] = None  # Stop GPS time for backward searches (default: O1 start ~1126051217)
    # Parallel processing configuration
    enable_parallel: bool = True  # Uses ProcessPoolExecutor (process-safe, avoids LAL thread issues)
    parallel_workers: Optional[int] = None  # Number of worker processes (None = auto: cpu_count)
    # Chi-squared threshold: only compute chi-sq when peak SNR >= this value (saves time on noise templates)
    chisq_snr_threshold: float = 9.0


@dataclass
class SearchConfig:
    runtime: SearchRuntimeConfig = field(default_factory=SearchRuntimeConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    bank: BankConfig = field(default_factory=BankConfig)
    background: BackgroundConfig = field(default_factory=BackgroundConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    known_events: KnownEventsConfig = field(default_factory=KnownEventsConfig)
    config_hash: Optional[str] = None

    def finalize(self) -> "SearchConfig":
        obj = asdict(self)
        obj.pop("config_hash", None)
        self.config_hash = _hash_dict(obj)
        if self.runtime.stop_file is None:
            self.runtime.stop_file = self.paths.output_dir / "STOP"
        return self

    @classmethod
    def from_yaml(cls, path: Path) -> "SearchConfig":
        import yaml  # lazy import

        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        cfg = cls(**data)
        cfg.finalize()
        return cfg

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=_json_default)
