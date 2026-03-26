from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


class StateManager:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._init_schema()

    def _init_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at REAL,
                config_hash TEXT,
                notes TEXT
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS blocks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                ifos TEXT,
                gps_start REAL,
                gps_end REAL,
                status TEXT,
                error TEXT,
                created_at REAL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS candidates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                gps REAL,
                ifos TEXT,
                template_id TEXT,
                network_stat REAL,
                far REAL,
                ifar_days REAL,
                payload TEXT,
                created_at REAL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS background (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                slide REAL,
                far REAL,
                payload TEXT,
                created_at REAL
            );
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS kv (
                k TEXT PRIMARY KEY,
                v TEXT
            );
            """
        )
        self._conn.commit()

    def create_run(self, config_hash: str, notes: str = "") -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO runs (started_at, config_hash, notes) VALUES (?, ?, ?)",
                (time.time(), config_hash, notes),
            )
            self._conn.commit()
            return cur.lastrowid

    def record_block(
        self,
        run_id: int,
        ifos: List[str],
        gps_start: float,
        gps_end: float,
        status: str,
        error: Optional[str] = None,
    ) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO blocks (run_id, ifos, gps_start, gps_end, status, error, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, ",".join(ifos), gps_start, gps_end, status, error, time.time()),
            )
            self._conn.commit()
            return cur.lastrowid

    def record_candidate(
        self,
        run_id: int,
        gps: float,
        ifos: List[str],
        template_id: str,
        network_stat: float,
        far: float,
        ifar_days: float,
        payload: Dict[str, Any],
    ) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO candidates (run_id, gps, ifos, template_id, network_stat, far, ifar_days, payload, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    gps,
                    ",".join(ifos),
                    template_id,
                    network_stat,
                    far,
                    ifar_days,
                    json.dumps(payload),
                    time.time(),
                ),
            )
            self._conn.commit()
            return cur.lastrowid

    def record_background(
        self,
        run_id: int,
        slide: float,
        far: float,
        payload: Dict[str, Any],
    ) -> int:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO background (run_id, slide, far, payload, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (run_id, slide, far, json.dumps(payload), time.time()),
            )
            self._conn.commit()
            return cur.lastrowid

    def get_last_block_end(self, run_id: Optional[int] = None) -> Optional[float]:
        cur = self._conn.cursor()
        if run_id is None:
            cur.execute("SELECT MAX(gps_end) FROM blocks")
        else:
            cur.execute("SELECT MAX(gps_end) FROM blocks WHERE run_id=?", (run_id,))
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None

    def get_min_block_start(self, run_id: Optional[int] = None) -> Optional[float]:
        """For backward search resume: return the lowest gps_start processed so far."""
        cur = self._conn.cursor()
        if run_id is None:
            cur.execute("SELECT MIN(gps_start) FROM blocks WHERE status IN ('done','gap')")
        else:
            cur.execute("SELECT MIN(gps_start) FROM blocks WHERE run_id=? AND status IN ('done','gap')", (run_id,))
        row = cur.fetchone()
        return row[0] if row and row[0] is not None else None

    def get_value(self, key: str) -> Optional[str]:
        cur = self._conn.cursor()
        cur.execute("SELECT v FROM kv WHERE k=?", (key,))
        row = cur.fetchone()
        return row[0] if row else None

    def set_value(self, key: str, value: str) -> None:
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "INSERT INTO kv (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v=excluded.v",
                (key, value),
            )
            self._conn.commit()
