from __future__ import annotations

import json
import logging
import signal
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

from .config import SearchConfig
from . import gwosc_io, bank, single_ifo, coincidence, background, report, known_events, state as state_mod, dashboard

LOG = logging.getLogger(__name__)


class GracefulKiller:
    def __init__(self):
        self.stop = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)

    def exit_gracefully(self, *_):
        self.stop = True


def run_search(cfg: SearchConfig) -> None:
        cfg = cfg.finalize()
        cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
        cfg.paths.cache_dir.mkdir(parents=True, exist_ok=True)
        cfg.paths.frames_dir.mkdir(parents=True, exist_ok=True)
        cfg.paths.log_dir.mkdir(parents=True, exist_ok=True)

        # ensure file logging for dashboard /logs endpoint
        log_file = cfg.paths.log_dir / "search.log"
        root_logger = logging.getLogger()
        if not any(isinstance(h, logging.FileHandler) for h in root_logger.handlers):
            fh = logging.FileHandler(log_file)
            fh.setLevel(root_logger.level)
            fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s"))
            root_logger.addHandler(fh)

        # PERFORMANCE: Setup optimal backend (CUDA if available, multi-threaded CPU otherwise)
        # This is already done in CLI, but we get the info here for use_cuda flag
        import os
        use_cuda = os.environ.get('PYCBC_SCHEME', 'cpu').lower() == 'cuda'
        LOG.info("Search running with backend: %s", 'CUDA' if use_cuda else 'CPU (multi-threaded)')

        killer = GracefulKiller()

        state = state_mod.StateManager(cfg.paths.state_db)
        run_id = state.create_run(cfg.config_hash or "unknown", notes="gwsearch run")

        LOG.info("Starting search run_id=%s", run_id)

        # CRITICAL FIX: Use targeted bank for GW150914 validation
        if cfg.bank.quick_test_gw150914:
            LOG.info("GW150914 TEST MODE: Generating targeted template bank")
            from .bank_targeted import generate_gw150914_test_bank, save_bank_to_dict
            templates = generate_gw150914_test_bank(grid_points_per_dim=15)
            template_bank = save_bank_to_dict(templates)
            LOG.info("Using %d targeted templates around GW150914 (m1=36.2, m2=29.1 M☉)",
                     len(templates))
        elif cfg.bank.use_geometric_bank:
            LOG.info("GEOMETRIC BANK MODE: Building metric-based template bank (method=%s)", cfg.bank.geometric_method)
            from .bank_geometric import load_or_build_geometric_bank, GeometricBankConfig
            geo_cfg = GeometricBankConfig(
                min_mass1=cfg.bank.min_mass1,
                max_mass1=cfg.bank.max_mass1,
                min_mass2=cfg.bank.min_mass2,
                max_mass2=cfg.bank.max_mass2,
                min_spin1z=cfg.bank.min_spin1z,
                max_spin1z=cfg.bank.max_spin1z,
                min_spin2z=cfg.bank.min_spin2z,
                max_spin2z=cfg.bank.max_spin2z,
                include_spins=cfg.bank.include_spins,
                min_match=cfg.bank.min_match,
                approximant=cfg.bank.approximant,
                f_lower=cfg.bank.f_lower,
                method=cfg.bank.geometric_method,
                cache_dir=cfg.bank.cache_dir,
            )
            template_bank = load_or_build_geometric_bank(geo_cfg)
            LOG.info("Geometric bank ready: %d templates (min_match=%.3f, spins=%s)",
                     len(template_bank["templates"]), cfg.bank.min_match, cfg.bank.include_spins)
        else:
            template_bank = bank.load_or_build_bank(cfg.bank)
    
        # Initialize current position - must happen regardless of template bank type
        start_time = _resolve_start_time(cfg, state, run_id)
        current = start_time
        
        # Set default backward stop time to O1 start if not specified
        if cfg.runtime.search_direction == "backward" and cfg.runtime.backward_stop_gps is None:
            cfg.runtime.backward_stop_gps = 1126051217.0  # O1 start: Sep 12, 2015
            LOG.info("Backward search: no stop time specified, using O1 start (GPS %.1f)", cfg.runtime.backward_stop_gps)

        known_events_cache = known_events.KnownEvents(cache_path=cfg.known_events.cache_file, url=cfg.known_events.api_url)
        known_events_cache.refresh_async()

        if cfg.runtime.dashboard.enabled:
            dashboard.start_dashboard(
                db_path=cfg.paths.state_db,
                stop_file=cfg.runtime.stop_file,
                log_file=cfg.paths.log_dir / "search.log",
                host=cfg.runtime.dashboard.host,
                port=cfg.runtime.dashboard.port,
            )

        while True:
            if killer.stop:
                LOG.info("Received stop signal, exiting after current block")
            
            # Determine block boundaries based on search direction
            if cfg.runtime.search_direction == "backward":
                # For backward search: block_start < block_end, but we move backward in time
                block_end = current  # current is the END of this block
                block_start = current - cfg.runtime.block_duration
                
                # Check if we've reached the stop point
                if cfg.runtime.backward_stop_gps and block_start < cfg.runtime.backward_stop_gps:
                    LOG.info("Reached backward search stop time (GPS %.1f), exiting", cfg.runtime.backward_stop_gps)
                    break
            else:
                # Forward search: normal behavior
                block_start = current
                block_end = current + cfg.runtime.block_duration
            state.set_value(
                "progress",
                json.dumps({"run_id": run_id, "block_start": block_start, "block_end": block_end, "stage": "init", "pct": 0.0, "ts": time.time()}),
            )

            try:
                frame_map = gwosc_io.fetch_frames(
                    ifos=list(cfg.runtime.ifos),
                    start=block_start,
                    end=block_end,
                    cache_dir=cfg.paths.frames_dir,
                    keep_frames=cfg.runtime.keep_frames,
                )
                state.set_value(
                    "progress",
                    json.dumps({"run_id": run_id, "block_start": block_start, "block_end": block_end, "stage": "frames", "pct": 0.2, "ts": time.time(), "msg": "frames downloaded"}),
                )
            except Exception as exc:  # noqa: BLE001
                LOG.exception("Frame fetch failed for block starting %s", block_start)
                state.set_value(
                    "progress",
                    json.dumps({"run_id": run_id, "block_start": block_start, "block_end": block_end, "stage": "error", "pct": 1.0, "ts": time.time(), "msg": f"frame fetch error: {exc}"}),
                )
                state.record_block(run_id, list(cfg.runtime.ifos), block_start, block_end, status="error", error=str(exc))
                if cfg.runtime.follow_latest:
                    time.sleep(cfg.runtime.poll_interval_sec)
                    continue
                raise

            block_id = state.record_block(run_id, list(cfg.runtime.ifos), block_start, block_end, status="processing")

            per_ifo_triggers: Dict[str, List[Dict]] = {}
            all_data_gaps = True
            for ifo, frame_path in frame_map.items():
                try:
                    per_ifo_triggers[ifo] = single_ifo.process_ifo_block(
                        ifo=ifo,
                        frame_path=frame_path,
                        cfg=cfg,
                        bank=template_bank,
                        block_start=block_start,
                        block_end=block_end,
                        use_cuda=use_cuda,  # Pass CUDA flag for optimization decisions
                        progress_cb=lambda frac, done, total, ifo=ifo: state.set_value(
                            "progress",
                            json.dumps(
                                {
                                    "run_id": run_id,
                                    "block_start": block_start,
                                    "block_end": block_end,
                                    "stage": "processing",
                                    "pct": 0.2 + 0.5 * max(0.0, min(1.0, frac)),
                                    "ts": time.time(),
                                    "msg": f"{ifo}: templates {done}/{total}",
                                }
                            ),
                        ),
                    )
                    all_data_gaps = False  # At least one IFO processed successfully
                except RuntimeError as exc:
                    # Check if this is a data gap error
                    if "data gap" in str(exc).lower() or "detector was off" in str(exc).lower() or "rms" in str(exc).lower():
                        LOG.warning("Data gap detected for %s in block %.1f-%.1f: %s", ifo, block_start, block_end, exc)
                        per_ifo_triggers[ifo] = []  # Empty triggers for this IFO
                        continue
                    else:
                        # Real error, re-raise
                        raise
                except Exception as exc:  # noqa: BLE001
                    LOG.exception("Processing failed for %s", ifo)
                    state.set_value(
                        "progress",
                        json.dumps({"run_id": run_id, "block_start": block_start, "block_end": block_end, "stage": "error", "pct": 1.0, "ts": time.time(), "msg": f"processing failed {ifo}: {exc}"}),
                    )
                    state.record_block(run_id, [ifo], block_start, block_end, status="error", error=str(exc))
                    raise
            
            # If all IFOs have data gaps, skip this block
            if all_data_gaps:
                LOG.warning("All IFOs have data gaps for block %.1f-%.1f; skipping", block_start, block_end)
                state.record_block(run_id, list(cfg.runtime.ifos), block_start, block_end, status="skipped", error="data gap")
                # Move to next block based on search direction
                if cfg.runtime.search_direction == "backward":
                    current = block_start  # Move backward
                else:
                    current = block_end  # Move forward
                if not cfg.runtime.follow_latest:
                    break
                continue

            state.set_value(
                "progress",
                json.dumps({"run_id": run_id, "block_start": block_start, "block_end": block_end, "stage": "ifo_done", "pct": 0.6, "ts": time.time()}),
            )

            coincs = coincidence.find_coincidences(
                per_ifo_triggers,
                window=0.015,
                require_same_template=cfg.runtime.coincidence_same_template,
                chirp_tolerance=0.1,
            )

            # Excise loud foreground events from background to prevent signal contamination.
            # Any coincidence with network_stat > 12 (well above noise floor) is treated as
            # a real signal and removed from both IFO trigger lists before time slides.
            loud_coinc_gps = [c["gps"] for c in coincs if c.get("network_stat", 0) > 12.0]

            bkg = background.compute_background(
                per_ifo_triggers,
                slides=cfg.background.quick_slides if cfg.bank.quick else cfg.background.production_slides,
                window=cfg.background.cluster_window,
                slide_spacing=cfg.background.slide_spacing,
                data_duration=cfg.runtime.block_duration,
                excise_gps=loud_coinc_gps if loud_coinc_gps else None,
            )

            candidates = report.build_candidates(
                coincs=coincs,
                background=bkg,
                cfg=cfg,
                template_bank=template_bank,
                known_events=known_events_cache,
            )
            
            # LOG CANDIDATE SUMMARY WITH KNOWN EVENT STATUS
            if len(candidates) > 0:
                known_cands = [c for c in candidates if c.get("known_event")]
                new_cands = [c for c in candidates if not c.get("known_event")]
                
                LOG.info("=" * 80)
                LOG.info("FOUND %d CANDIDATES in block GPS %.1f-%.1f:", len(candidates), block_start, block_end)
                LOG.info("=" * 80)
                
                if known_cands:
                    LOG.info("✓ KNOWN EVENTS RECOVERED (%d):", len(known_cands))
                    for c in known_cands:
                        known_name = c["known_event"].get("name", "Unknown") if isinstance(c.get("known_event"), dict) else str(c.get("known_event"))
                        LOG.info("  - GPS %.3f | %s | SNR %.1f | IFAR %.1f days | Event: %s",
                                 c["gps"], "+".join(c["ifos"]), c["network_stat"],
                                 c.get("ifar_days", 0), known_name)
                
                if new_cands:
                    LOG.info("★ NEW/UNKNOWN CANDIDATES (%d):", len(new_cands))
                    for c in new_cands:
                        LOG.info("  - GPS %.3f | %s | SNR %.1f | IFAR %.1f days | *** NEW EVENT ***",
                                 c["gps"], "+".join(c["ifos"]), c["network_stat"],
                                 c.get("ifar_days", 0))
                
                LOG.info("=" * 80)
            else:
                LOG.info("No candidates found in block GPS %.1f-%.1f", block_start, block_end)
            
            state.set_value(
                "progress",
                json.dumps({"run_id": run_id, "block_start": block_start, "block_end": block_end, "stage": "background", "pct": 0.8, "ts": time.time(), "msg": f"coincs={len(coincs)} cand={len(candidates)}"}),
            )

            # persist background summary
            state.record_background(
                run_id=run_id,
                slide=bkg.get("slide_spacing", 0.0),
                far=bkg.get("far_hz"),
                payload=bkg,
            )
        
            for cand in candidates:
                cand["run_id"] = run_id
                cand["block_id"] = block_id
                state.record_candidate(
                    run_id=run_id,
                    gps=cand["gps"],
                    ifos=cand["ifos"],
                    template_id=cand["template_id"],
                    network_stat=cand["network_stat"],
                    far=cand["far_hz"],
                    ifar_days=cand["ifar_days"],
                    payload=cand,
                )

            state.record_block(run_id, list(cfg.runtime.ifos), block_start, block_end, status="done")

            report.write_outputs(cfg.paths.output_dir, candidates, cfg)
            state.set_value(
                "progress",
                json.dumps({"run_id": run_id, "block_start": block_start, "block_end": block_end, "stage": "done", "pct": 1.0, "ts": time.time(), "msg": f"cand_written={len(candidates)}"}),
            )

            if killer.stop or (cfg.runtime.stop_file and Path(cfg.runtime.stop_file).exists()):
                LOG.info("Graceful stop requested; finishing after block %s", block_id)
                break

            # Move to next block based on search direction
            if cfg.runtime.search_direction == "backward":
                current = block_start  # Move backward in time
                LOG.info("Backward search: moving to block ending at GPS %.1f", current)
            else:
                current = block_end  # Move forward in time

            if cfg.runtime.follow_latest:
                if not gwosc_io.has_new_data(list(cfg.runtime.ifos), current, cfg):
                    LOG.info("At current edge; sleeping for %s sec", cfg.runtime.poll_interval_sec)
                    time.sleep(cfg.runtime.poll_interval_sec)
                continue
            else:
                # terminate after single scan if not follow-latest
                pass

            if not cfg.runtime.follow_latest:
                break


def _resolve_start_time(cfg: SearchConfig, state: state_mod.StateManager, run_id: int) -> float:
    if cfg.runtime.start_mode == "resume":
        last = state.get_last_block_end(run_id)
        if last is not None:
            return last
    if cfg.runtime.start_mode == "gps" and cfg.runtime.start_gps:
        return cfg.runtime.start_gps
    if cfg.runtime.start_mode == "latest":
        latest_end = gwosc_io.discover_latest_end(cfg.runtime.ifos, cfg.runtime.lookback_sec, cfg.paths.cache_dir)
        if cfg.runtime.search_direction == "backward":
            # For backward search, start at the latest available data
            LOG.info("Backward search: starting from latest available data (GPS %.1f)", latest_end)
            return latest_end
        else:
            # For forward search, start lookback seconds before latest
            return latest_end - cfg.runtime.lookback_sec
    # default fallback
    now = time.time()
    LOG.warning("Falling back to start at now-lookback")
    if cfg.runtime.search_direction == "backward":
        return now  # Start from now for backward search
    return now - cfg.runtime.lookback_sec
