from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from cachetools import TTLCache
from filelock import FileLock
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

LOG = logging.getLogger(__name__)

# Cache for latest-end discovery to avoid hammering GWOSC
_latest_cache: TTLCache = TTLCache(maxsize=8, ttl=1800)


def _parse_end_from_urls(urls: Iterable[str]) -> Optional[float]:
    max_end = None
    pat = re.compile(r"-(\d+)-(\d+)\.(?:hdf5|gwf)")
    for url in urls:
        m = pat.search(url)
        if not m:
            continue
        start = float(m.group(1))
        dur = float(m.group(2))
        end = start + dur
        if max_end is None or end > max_end:
            max_end = end
    return max_end


def _gwosc_urls_pycbc(ifo: str, start: float, end: float) -> List[str]:
    try:
        from pycbc.frame.gwosc import gwosc_frame_urls

        return gwosc_frame_urls(ifo, start, end) or []
    except Exception as exc:  # noqa: BLE001
        LOG.debug("pycbc gwosc_frame_urls failed: %s", exc)
        return []


def _gwosc_urls_locate(ifo: str, start: float, end: float) -> List[str]:
    try:
        from gwosc.locate import get_urls

        urls = get_urls(ifo, start, end, format="hdf5", host="https://gwosc.org") or []
        if not urls:
            urls = get_urls(ifo, start, end, format="gwf", host="https://gwosc.org") or []
        return urls
    except Exception as exc:  # noqa: BLE001
        LOG.debug("gwosc.locate get_urls failed: %s", exc)
        return []


def _gwosc_urls_direct(ifo: str, start: float, end: float) -> List[str]:
    """
    Query the GWOSC strain-files REST API directly for a frame containing [start, end).
    No event-versions call, no PyCBC dependency. Responses are cached by the gwosc library.
    """
    try:
        from gwosc.api.v2 import fetch_json
        from gwosc.api import DEFAULT_URL
    except Exception as exc:  # noqa: BLE001
        LOG.debug("gwosc.api.v2 not available: %s", exc)
        return []

    # Find which runs overlap with [start, end)
    runs_to_try = [
        run for (run, rs, re) in _RUN_SEGMENTS
        if rs <= start < re or rs < end <= re
    ]
    if not runs_to_try:
        runs_to_try = [run for (run, rs, _) in _RUN_SEGMENTS]

    # Query slightly before start to find frames that begin before the block
    q_start = int(start) - 4096
    q_end = int(end)

    for run in runs_to_try:
        url = (
            f"{DEFAULT_URL}/api/v2/runs/{run}/strain-files"
            f"?detector={ifo}&start={q_start}&stop={q_end}&sample-rate=4"
        )
        try:
            data = fetch_json(url)
            for item in data.get("results", []):
                gps_s = float(item.get("gps_start", 0))
                hdf5_url = item.get("hdf5_url", "")
                # 4096s frame: must fully contain [start, end)
                if gps_s <= start and gps_s + 4096 >= end and hdf5_url:
                    return [hdf5_url]
        except Exception as exc:  # noqa: BLE001
            LOG.debug("_gwosc_urls_direct run=%s ifo=%s failed: %s", run, ifo, exc)

    return []


def _latest_from_runs(ifos: Iterable[str]) -> Optional[float]:
    try:
        from gwosc import datasets
    except Exception as exc:  # noqa: BLE001
        LOG.debug("gwosc.datasets not available: %s", exc)
        return None

    ends = []
    for run in ["O4b3Disc", "O4b2Disc", "O4b1Disc", "O4a", "O3b", "O3a", "O2", "O1"]:
        try:
            seg = datasets.run_segment(run)
            if seg:
                ends.append(float(seg[1]))
        except Exception as exc:  # noqa: BLE001
            LOG.debug("run_segment(%s) failed: %s", run, exc)
            continue
    if ends:
        return max(ends)
    return None


def discover_latest_end(ifos: Iterable[str], lookback_sec: float, cache_dir: Path) -> float:
    """
    Discover the latest available end time. Fallback chain:
    - in-memory cache
    - disk cache (valid 30 min)
    - URL resolvers around now ± lookback_sec
    - if empty, fallback to GW150914 epoch (~1126259462)
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "latest_end.json"
    now = time.time()
    fallback_known = 1126259462.4  # GW150914 peak-ish

    cache_key = f"{','.join(ifos)}:{int(lookback_sec)}"
    if cache_key in _latest_cache:
        return _latest_cache[cache_key]

    if cache_file.exists():
        try:
            data = json.loads(cache_file.read_text())
            if cache_key in data and (now - data[cache_key]["ts"]) < 1800:
                cached_end = float(data[cache_key]["end"])
                if cached_end <= now + lookback_sec * 1.5:
                    return cached_end
                LOG.warning("Ignoring stale latest_end cache end=%s (> now+lookback); recomputing", cached_end)
        except Exception as exc:  # noqa: BLE001
            LOG.debug("Ignoring corrupt latest_end cache: %s", exc)

    ref_ifo = list(ifos)[0]
    start_probe = now - lookback_sec
    end_probe = now + lookback_sec

    urls = _gwosc_urls_pycbc(ref_ifo, start_probe, end_probe)
    if not urls:
        urls = _gwosc_urls_locate(ref_ifo, start_probe, end_probe)

    latest = _parse_end_from_urls(urls)
    if latest is None:
        run_latest = _latest_from_runs(ifos)
        if run_latest is not None:
            latest = run_latest
        else:
            LOG.warning("No GWOSC URLs found during latest discovery; falling back to known epoch %.1f", fallback_known)
            latest = fallback_known

    latest = min(latest, now + lookback_sec)  # clamp to reasonable horizon

    try:
        cache_data = {}
        if cache_file.exists():
            cache_data = json.loads(cache_file.read_text())
        cache_data[cache_key] = {"end": latest, "ts": now}
        cache_file.write_text(json.dumps(cache_data))
    except Exception as exc:  # noqa: BLE001
        LOG.debug("Failed to write latest_end cache: %s", exc)

    _latest_cache[cache_key] = latest
    return latest


def has_new_data(ifos: Iterable[str], current_end: float, cfg) -> bool:
    latest = discover_latest_end(ifos, cfg.runtime.lookback_sec, cfg.paths.cache_dir)
    return latest > current_end + 1.0


def _gwosc_frame_starts_in_run(ifo: str, run: str, start: float, end: float) -> List[float]:
    """
    Query the GWOSC paginated API for frame file gps_start values in [start, end).
    This works even when there are gaps (unlike gwosc.locate.get_urls).
    Returns sorted list of GPS start times.
    """
    try:
        from gwosc.api.v2 import fetch_json
    except Exception as exc:
        LOG.debug("gwosc.api.v2 not available: %s", exc)
        return []

    base = f"https://gwosc.org/api/v2/runs/{run}/strain-files"
    params = f"detector={ifo}&start={int(start)}&stop={int(end)}&sample-rate=4"
    url = f"{base}?{params}"
    starts = []
    page = 1
    while True:
        try:
            data = fetch_json(f"{url}&page={page}" if page > 1 else url)
        except Exception as exc:
            LOG.debug("_gwosc_frame_starts_in_run %s %s page=%d failed: %s", ifo, run, page, exc)
            break
        for item in data.get("results", []):
            gps = item.get("gps_start")
            if gps is not None:
                starts.append(float(gps))
        if page >= data.get("num_pages", 1):
            break
        page += 1
    return sorted(starts)


# Known observing run boundaries (GPS). Used as fallback when gwosc.datasets unavailable.
# O4b is not released as a full run — only discovery snippets are public on GWOSC.
_RUN_SEGMENTS = [
    ("O1",       1126051217.0, 1137254417.0),
    ("O2",       1164556817.0, 1187733618.0),
    ("O3a",      1238166018.0, 1253977218.0),
    ("O3b",      1256655618.0, 1269363618.0),
    ("O4a",      1368195220.0, 1389456018.0),
    ("O4b1Disc", 1412722688.0, 1412726784.0),  # ~2024-10-11 UTC
    ("O4b2Disc", 1415274496.0, 1415278592.0),  # ~2024-11-10 UTC
    ("O4b3Disc", 1420877824.0, 1420881920.0),  # ~2025-01-14 UTC
]


def _get_run_segments() -> List[tuple]:
    """Return [(run_name, start, end), ...] sorted by start, using GWOSC or hardcoded fallback."""
    try:
        from gwosc import datasets
        segs = []
        for run, fallback_start, fallback_end in _RUN_SEGMENTS:
            try:
                s, e = datasets.run_segment(run)
                segs.append((run, float(s), float(e)))
            except Exception:
                segs.append((run, fallback_start, fallback_end))
        return segs
    except Exception:
        return list(_RUN_SEGMENTS)


def find_previous_available_end(ifos: Iterable[str], before_gps: float) -> Optional[float]:
    """
    Find the latest GPS end time before `before_gps` where all IFOs have data.
    Uses the GWOSC paginated API to handle gaps correctly.
    Searches backward in 30-day windows to avoid fetching entire runs.
    """
    ifos = list(ifos)
    run_segments = _get_run_segments()

    for run_name, run_start, run_end in reversed(run_segments):
        if run_end < before_gps - 86400 * 365 * 5:
            break
        # Try progressively wider windows within this run, starting close to before_gps
        for window in [86400, 86400 * 7, 86400 * 30, 86400 * 90]:
            search_end = min(run_end, before_gps)
            search_start = max(run_start, before_gps - window)
            if search_start >= search_end:
                continue

            LOG.debug("find_previous_available_end: run %s window=%.0fd GPS %.1f-%.1f",
                      run_name, window / 86400, search_start, search_end)

            per_ifo_starts: Dict[str, List[float]] = {}
            for ifo in ifos:
                per_ifo_starts[ifo] = _gwosc_frame_starts_in_run(ifo, run_name, search_start, search_end)

            if not all(per_ifo_starts.get(ifo) for ifo in ifos):
                LOG.debug("find_previous_available_end: run %s window=%.0fd missing IFO data; widening",
                          run_name, window / 86400)
                continue

            ref_starts = set(per_ifo_starts[ifos[0]])
            for ifo in ifos[1:]:
                other = set(per_ifo_starts[ifo])
                # Use strict < 4096 so adjacent (non-overlapping) frames don't match.
                # Two 4096s frames overlap only if their starts differ by less than 4096s.
                ref_starts = {s for s in ref_starts if any(abs(s - o) < 4096 for o in other)}

            if not ref_starts:
                LOG.debug("find_previous_available_end: run %s window=%.0fd IFOs don't overlap; widening",
                          run_name, window / 86400)
                continue

            latest_start = max(ref_starts)
            latest_end = latest_start + 4096.0
            LOG.info("find_previous_available_end: last overlapping frame in %s ends GPS %.1f (%.1f hours before gap)",
                     run_name, latest_end, (before_gps - latest_end) / 3600)
            return latest_end

    LOG.warning("find_previous_available_end: no overlapping data found for %s before GPS %.1f", ifos, before_gps)
    return None


def get_previous_run_end(before_gps: float) -> Optional[float]:
    """Return the end GPS of the latest known run segment strictly before before_gps."""
    run_segs = sorted(_get_run_segments(), key=lambda x: x[1])
    result = None
    for _, rs, re in run_segs:
        if re < before_gps:
            result = re
    return result


@retry(
    retry=retry_if_exception_type((requests.RequestException, IOError)),
    wait=wait_exponential(multiplier=1, min=2, max=60),
    stop=stop_after_attempt(5),
)
def _download_atomic(url: str, dest: Path, timeout=(10, 300)) -> Path:
    """
    Atomic download with file locking to prevent corruption from concurrent access.
    Downloads to .partial file, validates size, then atomically renames.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)
    lock = FileLock(str(dest) + ".lock", timeout=600)
    
    with lock:
        # If file already exists and appears valid, keep it
        if dest.exists() and dest.stat().st_size > 1024:  # At least 1KB
            LOG.debug("Frame file %s already exists (size=%d), skipping download", dest, dest.stat().st_size)
            return dest
        
        # Download to temporary .partial file
        tmp = dest.with_suffix(dest.suffix + ".partial")
        LOG.info("Downloading %s -> %s", url, dest)
        
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                r.raise_for_status()
                expected_size = int(r.headers.get("Content-Length", "0")) or None
                
                with open(tmp, "wb") as f:
                    downloaded = 0
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                    f.flush()
                    os.fsync(f.fileno())
                
                # Validate size if Content-Length was provided
                actual_size = tmp.stat().st_size
                if expected_size and actual_size != expected_size:
                    tmp.unlink(missing_ok=True)
                    raise IOError(f"Size mismatch: got {actual_size}, expected {expected_size}")
                
                LOG.debug("Download complete: %d bytes", actual_size)
            
            # Atomic rename (POSIX systems)
            tmp.replace(dest)
            LOG.info("Successfully downloaded and validated %s", dest)
            
        except Exception:
            # Clean up partial file on any error
            tmp.unlink(missing_ok=True)
            raise
    
    return dest


def _find_cached_frame(ifo: str, start: float, end: float, cache_dir: Path) -> Optional[Path]:
    """
    Check if a valid frame file already exists in cache_dir that covers [start, end).
    Parses filenames like H-H1_GWOSC_O4a_4KHZ_R1-1389453312-4096.hdf5.
    Returns the path if found, None otherwise. Zero HTTP calls.
    """
    pat = re.compile(r"-(\d+)-(\d+)\.hdf5$")
    ifo_prefix = ifo[0]  # 'H' for H1, 'L' for L1, 'V' for V1
    for f in cache_dir.glob(f"{ifo_prefix}-{ifo}*.hdf5"):
        m = pat.search(f.name)
        if not m:
            continue
        f_start = float(m.group(1))
        f_dur = float(m.group(2))
        f_end = f_start + f_dur
        if f_start <= start and f_end >= end and f.stat().st_size > 1024:
            LOG.debug("Cache hit: %s covers GPS [%.1f, %.1f)", f.name, start, end)
            return f
    return None


def fetch_frames(
    ifos: List[str],
    start: float,
    end: float,
    cache_dir: Path,
    keep_frames: bool = False,
) -> Dict[str, Path]:
    """
    Fetch GWOSC frame files for specified IFOs and time range.
    Checks local cache first — no HTTP call if the file is already on disk.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}

    for ifo in ifos:
        # --- Cache-first: skip HTTP entirely if we already have the file ---
        cached = _find_cached_frame(ifo, start, end, cache_dir)
        if cached is not None:
            results[ifo] = cached
            continue

        # Cache miss: resolve URL via direct strain-files API (no event-versions call)
        urls = _gwosc_urls_direct(ifo, start, end)
        if not urls:
            raise RuntimeError(f"No GWOSC URLs found for {ifo} {start}-{end}")

        url = urls[0]
        fname = Path(url.split("/")[-1])
        dest = cache_dir / fname
        _download_atomic(url, dest)
        results[ifo] = dest

    return results