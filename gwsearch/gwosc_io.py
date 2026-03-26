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


def _latest_from_runs(ifos: Iterable[str]) -> Optional[float]:
    try:
        from gwosc import datasets
    except Exception as exc:  # noqa: BLE001
        LOG.debug("gwosc.datasets not available: %s", exc)
        return None

    ends = []
    for run in ["O4b", "O4a", "O4", "O3b", "O3a", "O2", "O1"]:
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


def fetch_frames(
    ifos: List[str],
    start: float,
    end: float,
    cache_dir: Path,
    keep_frames: bool = False,
) -> Dict[str, Path]:
    """
    Fetch GWOSC frame files for specified IFOs and time range.
    Uses atomic downloads with file locking to prevent corruption.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    results: Dict[str, Path] = {}

    for ifo in ifos:
        urls = _gwosc_urls_pycbc(ifo, start, end)
        if not urls:
            urls = _gwosc_urls_locate(ifo, start, end)
        if not urls:
            raise RuntimeError(f"No GWOSC URLs found for {ifo} {start}-{end}")

        # Choose first url (typically covers the requested range)
        url = urls[0]
        fname = Path(url.split("/")[-1])
        dest = cache_dir / fname
        
        # Atomic download handles validation and locking
        _download_atomic(url, dest)
        results[ifo] = dest

    if not keep_frames:
        # Caller responsible for cleaning after use; we only signal paths
        pass

    return results