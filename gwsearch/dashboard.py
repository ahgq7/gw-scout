from __future__ import annotations

import logging
import threading
import subprocess
import socket
from pathlib import Path
from typing import Any, Dict, List

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, HTMLResponse
import uvicorn
import sqlite3

LOG = logging.getLogger(__name__)


def _read_recent_candidates(db_path: Path, limit: int = 20):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute(
        "SELECT run_id, gps, ifos, network_stat, far, ifar_days, template_id, created_at FROM candidates ORDER BY created_at DESC LIMIT ?",
        (limit,),
    )
    rows = cur.fetchall()
    con.close()
    return [
        {
            "run_id": r[0],
            "gps": r[1],
            "ifos": (r[2] or "").split(","),
            "network_stat": r[3],
            "far": r[4],
            "ifar_days": r[5],
            "template_id": r[6],
            "created_at": r[7],
        }
        for r in rows
    ]


def _read_recent_blocks(db_path: Path, limit: int = 20):
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT gps_start, gps_end, status, error, created_at FROM blocks ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    con.close()
    return [
        {
            "gps_start": r[0],
            "gps_end": r[1],
            "status": r[2],
            "error": r[3],
            "created_at": r[4],
        }
        for r in rows
    ]


def _read_background(db_path: Path, limit: int = 50) -> List[Dict[str, Any]]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT slide, far, payload, created_at FROM background ORDER BY created_at DESC LIMIT ?", (limit,))
    rows = cur.fetchall()
    con.close()
    out = []
    for r in rows:
        payload = r[2]
        try:
            import json
            payload = json.loads(payload) if isinstance(payload, str) else payload
        except Exception:
            pass
        out.append({"slide": r[0], "far": r[1], "payload": payload, "created_at": r[3]})
    return out


def _read_status(db_path: Path) -> Dict[str, Any]:
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("SELECT id, started_at, config_hash FROM runs ORDER BY started_at DESC LIMIT 1")
    run_row = cur.fetchone()
    run_id = run_row[0] if run_row else None

    def _one(query: str, params=()):
        cur.execute(query, params)
        row = cur.fetchone()
        return row[0] if row else None

    block_total = _one("SELECT COUNT(*) FROM blocks WHERE run_id=?", (run_id,)) if run_id else 0
    block_done = _one("SELECT COUNT(*) FROM blocks WHERE run_id=? AND status='done'", (run_id,)) if run_id else 0
    block_err = _one("SELECT COUNT(*) FROM blocks WHERE run_id=? AND status='error'", (run_id,)) if run_id else 0
    block_proc = _one("SELECT COUNT(*) FROM blocks WHERE run_id=? AND status='processing'", (run_id,)) if run_id else 0
    cand_count = _one("SELECT COUNT(*) FROM candidates WHERE run_id=?", (run_id,)) if run_id else 0
    bkg_count = _one("SELECT COUNT(*) FROM background WHERE run_id=?", (run_id,)) if run_id else 0
    last_block = _one("SELECT MAX(created_at) FROM blocks WHERE run_id=?", (run_id,)) if run_id else None
    last_cand = _one("SELECT MAX(created_at) FROM candidates WHERE run_id=?", (run_id,)) if run_id else None

    con.close()
    return {
        "latest_run_id": run_id,
        "latest_run_started_at": run_row[1] if run_row else None,
        "latest_run_config_hash": run_row[2] if run_row else None,
        "block_count": block_total,
        "block_done": block_done,
        "block_error": block_err,
        "block_processing": block_proc,
        "candidate_count": cand_count,
        "background_count": bkg_count,
        "last_block_created_at": last_block,
        "last_candidate_created_at": last_cand,
    }


def create_app(db_path: Path, stop_file: Path, log_file: Path):
    app = FastAPI()

    @app.get("/")
    def index():
        return {
            "status": _read_status(db_path),
            "candidates": _read_recent_candidates(db_path),
            "blocks": _read_recent_blocks(db_path),
            "background": _read_background(db_path, limit=10),
        }

    @app.get("/status")
    def status():
        status = _read_status(db_path)
        try:
            con = sqlite3.connect(db_path)
            cur = con.cursor()
            cur.execute("SELECT v FROM kv WHERE k='progress'")
            row = cur.fetchone()
            if row and row[0]:
                import json
                status["progress"] = json.loads(row[0])
        except Exception as exc:  # noqa: BLE001
            LOG.debug("progress read failed: %s", exc)
        finally:
            try:
                con.close()
            except Exception:
                pass
        return status

    @app.get("/candidates")
    def candidates(limit: int = 50):
        return _read_recent_candidates(db_path, limit=limit)

    @app.get("/blocks")
    def blocks(limit: int = 50):
        return _read_recent_blocks(db_path, limit=limit)

    @app.get("/background")
    def background(limit: int = 50):
        return _read_background(db_path, limit=limit)

    @app.get("/logs", response_class=PlainTextResponse)
    def logs():
        if log_file.exists():
            return log_file.read_text()[-4000:]
        return "no logs yet"

    @app.get("/control/stop")
    def control_stop():
        stop_file.parent.mkdir(parents=True, exist_ok=True)
        stop_file.write_text("stop requested")
        return {"status": "ok"}

    @app.get("/ui", response_class=HTMLResponse)
    def ui():
        return """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <title>GWSearch Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
  <style>
    :root {
      --bg: #0b1021;
      --panel: #0f172a;
      --panel-2: #111827;
      --text: #e5e7eb;
      --muted: #9ca3af;
      --accent: #38bdf8;
      --accent-2: #a855f7;
      --border: #1f2937;
      --ok: #34d399;
      --warn: #fbbf24;
      --err: #f87171;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0; padding: 0;
      font-family: 'Inter', system-ui, -apple-system, sans-serif;
      background: radial-gradient(circle at 10% 20%, rgba(56,189,248,0.12), transparent 25%),
                  radial-gradient(circle at 80% 0%, rgba(168,85,247,0.08), transparent 25%),
                  var(--bg);
      color: var(--text);
      min-height: 100vh;
    }
    header {
      padding: 18px 24px;
      border-bottom: 1px solid var(--border);
      display: flex;
      align-items: center;
      justify-content: space-between;
      background: rgba(15,23,42,0.75);
      backdrop-filter: blur(6px);
      position: sticky;
      top: 0;
      z-index: 10;
    }
    h1 { margin: 0; font-weight: 700; letter-spacing: -0.01em; }
    .container { padding: 20px 24px 40px; max-width: 1280px; margin: 0 auto; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-bottom: 18px;
    }
    .card {
      background: linear-gradient(145deg, rgba(15,23,42,0.9), rgba(17,24,39,0.92));
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px 16px;
      box-shadow: 0 10px 30px rgba(0,0,0,0.35);
    }
    .card h2 {
      margin: 0 0 10px 0;
      font-size: 15px;
      letter-spacing: -0.01em;
      color: var(--accent);
    }
    .meta { display: flex; gap: 12px; flex-wrap: wrap; color: var(--muted); font-size: 13px; }
    table { border-collapse: collapse; width: 100%; margin-top: 8px; }
    th, td { padding: 8px 10px; border-bottom: 1px solid var(--border); font-size: 13px; }
    th { text-align: left; color: var(--muted); background: rgba(17,24,39,0.4); position: sticky; top: 0; }
    tbody tr:hover { background: rgba(56,189,248,0.06); }
    pre {
      background: var(--panel-2);
      border: 1px solid var(--border);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 12px;
      max-height: 260px;
      overflow: auto;
    }
    .pill { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 12px; }
    .pill.ok { background: rgba(52,211,153,0.15); color: var(--ok); }
    .pill.err { background: rgba(248,113,113,0.15); color: var(--err); }
    .pill.warn { background: rgba(251,191,36,0.15); color: var(--warn); }
    a { color: var(--accent); text-decoration: none; }
    footer { margin-top: 12px; text-align: center; color: var(--muted); font-size: 12px; }
    .progress-wrap { margin-top: 8px; }
    .progress-label { display: flex; justify-content: space-between; color: var(--muted); font-size: 12px; }
    .progress-msg { color: var(--muted); font-size: 12px; margin-top: 4px; }
    .progress-bar {
      width: 100%;
      height: 10px;
      border-radius: 999px;
      background: rgba(255,255,255,0.08);
      overflow: hidden;
      border: 1px solid var(--border);
    }
    .progress-bar > div {
      height: 100%;
      width: 0%;
      background: linear-gradient(90deg, var(--accent), var(--accent-2));
      transition: width 0.3s ease;
    }
  </style>
</head>
<body>
  <header>
    <div>
      <h1>GWSearch Dashboard</h1>
      <div class="meta">
        <span>Live status & candidates</span>
        <span>Auto-refresh 5s</span>
      </div>
    </div>
    <div class="meta">
      <span>UI: /ui</span>
      <span>API: /status /candidates /blocks /background /logs</span>
    </div>
  </header>

  <div class="container">
    <div class="grid">
      <div class="card" id="status-card">
        <h2>Status</h2>
        <pre id="status">Loading...</pre>
        <div class="progress-wrap">
          <div class="progress-label">
            <span id="progress-stage">init</span>
            <span id="progress-pct">0%</span>
          </div>
          <div class="progress-bar"><div id="progress-bar"></div></div>
          <div class="progress-label" id="progress-counts"></div>
          <div class="progress-msg" id="progress-msg"></div>
        </div>
      </div>
      <div class="card">
        <h2>Logs (tail)</h2>
        <pre id="logs">Loading...</pre>
      </div>
    </div>

    <div class="card">
      <h2>Candidates (latest)</h2>
      <table id="candidates"><thead><tr><th>run</th><th>gps</th><th>ifos</th><th>net stat</th><th>FAR</th><th>IFAR days</th><th>template</th><th>created</th></tr></thead><tbody></tbody></table>
    </div>

    <div class="grid">
      <div class="card">
        <h2>Blocks (latest)</h2>
        <table id="blocks"><thead><tr><th>gps_start</th><th>gps_end</th><th>status</th><th>error</th><th>created</th></tr></thead><tbody></tbody></table>
      </div>
      <div class="card">
        <h2>Background (latest)</h2>
        <table id="background"><thead><tr><th>slide</th><th>far</th><th>created</th></tr></thead><tbody></tbody></table>
      </div>
    </div>

    <footer>GWSearch local dashboard · refreshed every 5 seconds</footer>
  </div>

<script>
async function fetchJson(url) { const r = await fetch(url); return await r.json(); }
function pill(text, cls="ok") { return `<span class="pill ${cls}">${text}</span>`; }

async function refresh() {
  try {
    const status = await fetchJson("/status");
    document.getElementById("status").textContent = JSON.stringify(status, null, 2);
    const p = (status && status.progress) ? status.progress : null;
    const pct = p && typeof p.pct === "number" ? Math.max(0, Math.min(1, p.pct)) : 0;
    document.getElementById("progress-pct").textContent = `${(pct * 100).toFixed(2)}%`;
    document.getElementById("progress-bar").style.width = `${pct * 100}%`;
    document.getElementById("progress-stage").textContent = p && p.stage ? p.stage : "idle";
    const done = status.block_done ?? 0;
    const total = status.block_count ?? 0;
    const err = status.block_error ?? 0;
    const proc = status.block_processing ?? 0;
    document.getElementById("progress-counts").textContent = `blocks: ${done}/${total} done, processing=${proc}, error=${err}`;
    document.getElementById("progress-msg").textContent = p && p.msg ? p.msg : "";

    const cand = await fetchJson("/candidates?limit=20");
    const cbody = document.querySelector("#candidates tbody");
    cbody.innerHTML = "";
    cand.forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${row.run_id ?? ""}</td><td>${row.gps?.toFixed ? row.gps.toFixed(3) : row.gps}</td><td>${(row.ifos||[]).join(",")}</td><td>${row.network_stat?.toFixed ? row.network_stat.toFixed(3) : row.network_stat}</td><td>${row.far}</td><td>${row.ifar_days}</td><td>${row.template_id}</td><td>${row.created_at}</td>`;
      cbody.appendChild(tr);
    });

    const blocks = await fetchJson("/blocks?limit=20");
    const bbody = document.querySelector("#blocks tbody");
    bbody.innerHTML = "";
    blocks.forEach(row => {
      const statusPill = row.status === "done" ? pill("done","ok") :
                         row.status === "error" ? pill("error","err") : pill(row.status,"warn");
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${row.gps_start}</td><td>${row.gps_end}</td><td>${statusPill}</td><td>${row.error ?? ""}</td><td>${row.created_at}</td>`;
      bbody.appendChild(tr);
    });

    const bkg = await fetchJson("/background?limit=20");
    const bgbody = document.querySelector("#background tbody");
    bgbody.innerHTML = "";
    bkg.forEach(row => {
      const tr = document.createElement("tr");
      tr.innerHTML = `<td>${row.slide}</td><td>${row.far}</td><td>${row.created_at}</td>`;
      bgbody.appendChild(tr);
    });

    const logs = await fetch("/logs");
    document.getElementById("logs").textContent = await logs.text();
  } catch (e) {
    console.error(e);
  }
}
refresh();
setInterval(refresh, 5000);
</script>
</body>
</html>
"""
    return app


def _ensure_port_free(host: str, port: int) -> None:
    """If port is in use, attempt to kill the owning process (Linux)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        in_use = s.connect_ex((host, port)) == 0
    if not in_use:
        return
    try:
        subprocess.run(["fuser", "-k", f"{port}/tcp"], check=False, capture_output=True)
        LOG.warning("Port %s:%s was in use; attempted to kill owner via fuser -k", host, port)
    except FileNotFoundError:
        LOG.error("Port %s:%s in use and 'fuser' not available; please free it manually", host, port)


def start_dashboard(db_path: Path, stop_file: Path, log_file: Path, host: str, port: int):
    _ensure_port_free(host, port)
    app = create_app(db_path, stop_file, log_file)
    thread = threading.Thread(
        target=lambda: uvicorn.run(app, host=host, port=port, log_level="warning"),
        daemon=True,
    )
    thread.start()
    LOG.info("Dashboard running at http://%s:%s", host, port)
