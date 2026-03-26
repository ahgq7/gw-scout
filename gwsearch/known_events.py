from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Dict, Optional

import requests

LOG = logging.getLogger(__name__)


class KnownEvents:
    def __init__(self, cache_path: Path, url: str):
        self.cache_path = Path(cache_path)
        self.url = url
        self._events: Dict[str, float] = {}
        self._lock = threading.Lock()
        self._last_fetch = 0.0
        self._fetch_disabled = False
        self._session = requests.Session()  # Connection pooling
        self._load_cache()

    def _fetch_json_paginated(self, url: str, timeout: int = 60) -> Optional[Dict]:
        """
        Fetch paginated GWOSC API v2 data with proper throttling.
        
        Args:
            url: Base API URL (without page parameters)
            timeout: Request timeout in seconds (default 60s for large datasets)
            
        Returns:
            Dict with "events" key mapping event names to GPS times, or None on failure
        """
        results = {}
        merged: Dict[str, float] = {}
        next_url = url
        page_count = 0
        
        while next_url:
            try:
                # Use session for connection pooling, add proper headers
                r = self._session.get(
                    next_url,
                    timeout=timeout,
                    headers={
                        "Accept": "application/json",
                        "Accept-Encoding": "gzip, deflate",
                        "User-Agent": "gwsearch/1.0 (gravitational wave search pipeline)",
                    },
                )
                r.raise_for_status()
                
                ctype = r.headers.get("Content-Type", "")
                if "json" not in ctype.lower():
                    LOG.warning("Known-events fetch failed: non-JSON response (Content-Type=%s); disabling further fetches. Body preview: %.200s", ctype, r.text)
                    self._fetch_disabled = True
                    return None
                    
                try:
                    data = r.json()
                except Exception as exc:  # noqa: BLE001
                    LOG.warning("Known-events fetch failed (parse): %s; body preview: %.200s", exc, r.text)
                    self._fetch_disabled = True
                    return None
                
                # Extract event data using correct GWOSC API v2 field names
                for item in data.get("results", []):
                    # API v2 uses "name" field (not "event" or "display_name")
                    name = item.get("name") or item.get("shortName")
                    # API v2 uses "gps" field (not "t_0")
                    gps = item.get("gps")
                    
                    if name and gps is not None:
                        merged[name] = float(gps)  # Convert scientific notation to float
                
                page_count += 1
                LOG.debug("Fetched page %d: %d events on this page, %d total", page_count, len(data.get("results", [])), len(merged))
                
                # Follow pagination using "next" field from response
                next_url = data.get("next")
                
                # GWOSC recommendation: 300-500ms delay between requests to avoid 429 errors
                if next_url:
                    time.sleep(0.35)  # 350ms throttle
                    
            except requests.exceptions.Timeout as exc:
                LOG.warning("Known-events fetch timeout on page %d: %s", page_count + 1, exc)
                # Don't disable on timeout - may be temporary
                return None if page_count == 0 else {"events": merged}  # Return partial results if we got some
            except requests.exceptions.RequestException as exc:
                LOG.warning("Known-events fetch failed on page %d: %s", page_count + 1, exc)
                self._fetch_disabled = True
                return None
                
        results["events"] = merged
        return results

    def _load_cache(self) -> None:
        if self.cache_path.exists():
            try:
                data = json.loads(self.cache_path.read_text())
                self._events = {k: float(v) for k, v in data.get("events", {}).items()}
                LOG.debug("Loaded known-events cache %s entries", len(self._events))
            except Exception as exc:  # noqa: BLE001
                LOG.debug("Failed to load known-events cache: %s", exc)

    def refresh_async(self) -> None:
        t = threading.Thread(target=self.refresh, daemon=True)
        t.start()

    def refresh(self, force: bool = False) -> None:
        if self._fetch_disabled:
            return
        if not force and self._events and (time.time() - self._last_fetch) < 3600:
            return
        try:
            data = self._fetch_json_paginated(self.url)
            if data is None:
                return
            events = data.get("events", {})
            if events:
                with self._lock:
                    self._events = events
                    self._last_fetch = time.time()
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                self.cache_path.write_text(json.dumps({"events": events, "fetched": self._last_fetch}))
                LOG.info("Fetched %s known events", len(events))
        except Exception as exc:  # noqa: BLE001
            LOG.warning("Known-events fetch failed: %s", exc)

    def match_event(self, gps: float, tol: float = 1.0) -> Optional[str]:
        best = None
        best_dt = None
        with self._lock:
            for name, egps in self._events.items():
                dt = abs(gps - egps)
                if dt <= tol and (best_dt is None or dt < best_dt):
                    best = name
                    best_dt = dt
        if best:
            return best
        try:
            from gwosc import datasets

            ev = datasets.event_at_gps(gps)
            return ev
        except Exception:  # noqa: BLE001
            return None
