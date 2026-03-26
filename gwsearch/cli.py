from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.logging import RichHandler

from .runtime_patches import apply_runtime_patches
from .config import SearchConfig
from .main import run_search

# Apply compatibility patches before any PyCBC imports
apply_runtime_patches()

app = typer.Typer(add_completion=False, no_args_is_help=True)


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True)],
    )


@app.command()
def search(
    config: Optional[Path] = typer.Option(None, help="Path to YAML config"),
    follow_latest: bool = typer.Option(False, "--follow-latest", help="Poll for new data and continue"),
    poll_interval_sec: float = typer.Option(3.0, help="Polling interval when follow-latest"),
    start_mode: str = typer.Option("latest", help="latest|resume|gps"),
    start_gps: Optional[float] = typer.Option(None, help="Start GPS when start-mode=gps"),
    lookback_sec: float = typer.Option(7 * 24 * 3600, help="Lookback when start-mode=latest"),
    block_duration: float = typer.Option(512.0, help="Block duration (s)"),
    chunk_duration: float = typer.Option(128.0, help="Chunk duration (s)"),
    keep_frames: bool = typer.Option(False, help="Keep downloaded frames"),
    graceful_stop: bool = typer.Option(True, help="Finish current block before stop"),
    stop_file: Optional[Path] = typer.Option(None, help="File path to request graceful stop"),
    dashboard: bool = typer.Option(False, "--dashboard", help="Enable local dashboard"),
    dashboard_port: int = typer.Option(7860, help="Dashboard port"),
    detection_snr_threshold: float = typer.Option(5.5, help="Single-IFO detection SNR threshold"),
    coincidence_same_template: bool = typer.Option(True, help="Require same template ID across IFOs in coincidence"),
    bank_quick: bool = typer.Option(False, "--bank-quick", help="Use quick template bank (truncates to max_templates_quick)"),
    bank_max_templates_quick: Optional[int] = typer.Option(None, help="Override quick bank template cap (default config value)"),
    bank_quick_test_gw150914: bool = typer.Option(False, "--bank-quick-test-gw150914", help="Use GW150914 targeted template bank (validation mode)"),
    bank_exclude_signal_from_psd_sec: float = typer.Option(0.0, "--bank-exclude-signal-from-psd-sec", help="Exclude first N seconds from PSD estimation (for known events)"),
    use_geometric_bank: Optional[bool] = typer.Option(None, "--use-geometric-bank/--no-geometric-bank", help="Use metric-based stochastic template bank (default: True)"),
    include_spins: Optional[bool] = typer.Option(None, "--include-spins/--no-spins", help="Include aligned-spin templates in bank (default: True)"),
    geometric_method: str = typer.Option("stochastic", help="Geometric bank method: stochastic|pycbc_external"),
    runtime_known_event_gps: Optional[float] = typer.Option(None, "--runtime-known-event-gps", help="GPS time of known event (for validation)"),
    search_direction: str = typer.Option("forward", help="Search direction: forward|backward (backward searches from latest to past)"),
    backward_stop_gps: Optional[float] = typer.Option(None, help="Stop GPS time for backward searches (default: O1 start ~1126051217)"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
):
    # Enable faulthandler for hang diagnostics (kill -USR2 <pid> to dump stacks)
    import faulthandler, signal
    faulthandler.enable(all_threads=True)
    faulthandler.register(signal.SIGUSR2, all_threads=True, chain=False)
    
    setup_logging(verbose)
    
    # Setup optimal performance backend (CUDA if available, multi-threaded CPU otherwise)
    from .performance import setup_performance_backend, get_device_info
    
    # Show available devices
    device_info = get_device_info()
    typer.echo(f"CPU cores: {device_info['cpu_cores']}")
    if device_info['cuda_devices']:
        for dev in device_info['cuda_devices']:
            typer.echo(f"CUDA device {dev['id']}: {dev['name']} "
                      f"({dev['total_memory_gb']:.1f} GB, "
                      f"compute {dev['compute_capability'][0]}.{dev['compute_capability'][1]})")
    
    # Setup performance backend (auto-detect CUDA, multi-thread CPU)
    backend_info = setup_performance_backend(force_cuda=False)
    typer.echo(f"Performance backend: {backend_info['scheme'].upper()} "
              f"({backend_info['num_threads']} CPU threads)")
    if config:
        cfg = SearchConfig.from_yaml(config)
    else:
        cfg = SearchConfig().finalize()

    cfg.runtime.follow_latest = follow_latest or cfg.runtime.follow_latest
    cfg.runtime.poll_interval_sec = poll_interval_sec or cfg.runtime.poll_interval_sec
    # If start_gps is provided, automatically use gps mode
    if start_gps is not None:
        cfg.runtime.start_mode = "gps"
        cfg.runtime.start_gps = start_gps
    else:
        cfg.runtime.start_mode = start_mode or cfg.runtime.start_mode
        cfg.runtime.start_gps = cfg.runtime.start_gps
    cfg.runtime.lookback_sec = lookback_sec or cfg.runtime.lookback_sec
    cfg.runtime.block_duration = block_duration or cfg.runtime.block_duration
    cfg.runtime.chunk_duration = chunk_duration or cfg.runtime.chunk_duration
    cfg.runtime.keep_frames = keep_frames or cfg.runtime.keep_frames
    cfg.runtime.graceful_stop = graceful_stop
    cfg.runtime.stop_file = stop_file or cfg.runtime.stop_file
    cfg.runtime.dashboard.enabled = dashboard or cfg.runtime.dashboard.enabled
    cfg.runtime.dashboard.port = dashboard_port or cfg.runtime.dashboard.port
    cfg.runtime.detection_snr_threshold = detection_snr_threshold or cfg.runtime.detection_snr_threshold
    cfg.runtime.coincidence_same_template = coincidence_same_template
    cfg.bank.quick = bank_quick or cfg.bank.quick
    if bank_max_templates_quick is not None:
        cfg.bank.max_templates_quick = bank_max_templates_quick
    
    # GW150914 validation mode parameters
    cfg.bank.quick_test_gw150914 = bank_quick_test_gw150914 or cfg.bank.quick_test_gw150914
    cfg.bank.exclude_signal_from_psd_sec = bank_exclude_signal_from_psd_sec or cfg.bank.exclude_signal_from_psd_sec

    # Geometric bank and spin options
    if use_geometric_bank is not None:
        cfg.bank.use_geometric_bank = use_geometric_bank
    if include_spins is not None:
        cfg.bank.include_spins = include_spins
    cfg.bank.geometric_method = geometric_method or cfg.bank.geometric_method
    if runtime_known_event_gps is not None:
        cfg.runtime.known_event_gps = runtime_known_event_gps
    
    # Backward search parameters
    cfg.runtime.search_direction = search_direction or cfg.runtime.search_direction
    if backward_stop_gps is not None:
        cfg.runtime.backward_stop_gps = backward_stop_gps

    typer.echo(f"Config hash: {cfg.config_hash}")
    run_search(cfg)


@app.command()
def show_config():
    cfg = SearchConfig().finalize()
    typer.echo(cfg.to_json())


def main():
    app()


if __name__ == "__main__":
    main()
