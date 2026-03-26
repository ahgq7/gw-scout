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
        return r"""
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>GW Scout</title>
<style>
:root{
  --bg:#06090f;
  --surface:#0d1117;
  --surface2:#111820;
  --border:#1e2733;
  --text:#c9d1d9;
  --muted:#6e7681;
  --blue:#58a6ff;
  --purple:#bc8cff;
  --green:#3fb950;
  --yellow:#d29922;
  --red:#f85149;
  --cyan:#39c5cf;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  background:var(--bg);color:var(--text);font-size:13px;min-height:100vh}

/* HEADER */
header{
  display:flex;align-items:center;justify-content:space-between;
  padding:12px 20px;
  background:var(--surface);
  border-bottom:1px solid var(--border);
  position:sticky;top:0;z-index:100;
}
.logo{display:flex;align-items:center;gap:10px}
.logo svg{flex-shrink:0}
.logo-text h1{font-size:15px;font-weight:600;color:#fff;letter-spacing:.01em}
.logo-text p{font-size:11px;color:var(--muted);margin-top:1px}
.header-right{display:flex;align-items:center;gap:8px}
.dot{width:8px;height:8px;border-radius:50%;background:var(--green);
  animation:pulse 2s infinite;flex-shrink:0}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
.refresh-label{font-size:11px;color:var(--muted)}

/* MAIN LAYOUT */
main{padding:16px 20px;max-width:1400px;margin:0 auto;display:flex;flex-direction:column;gap:14px}

/* STAT CHIPS */
.stats-row{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:10px}
.stat-chip{
  background:var(--surface);border:1px solid var(--border);border-radius:10px;
  padding:12px 14px;display:flex;flex-direction:column;gap:4px;
}
.stat-label{font-size:11px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.stat-value{font-size:20px;font-weight:700;color:#fff;line-height:1}
.stat-sub{font-size:11px;color:var(--muted)}
.stat-chip.highlight{border-color:rgba(88,166,255,.3);background:rgba(88,166,255,.05)}
.stat-chip.alert{border-color:rgba(63,185,80,.4);background:rgba(63,185,80,.07)}

/* CURRENT BLOCK CARD */
.block-card{
  background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:14px 16px;
}
.block-header{display:flex;align-items:center;justify-content:space-between;margin-bottom:10px}
.block-header-left{display:flex;align-items:center;gap:8px}
.block-title{font-size:13px;font-weight:600;color:#fff}
.stage-pill{
  font-size:11px;padding:2px 8px;border-radius:999px;
  background:rgba(88,166,255,.15);color:var(--blue);font-weight:500;
}
.stage-pill.done{background:rgba(63,185,80,.15);color:var(--green)}
.stage-pill.error{background:rgba(248,81,73,.15);color:var(--red)}
.pct-label{font-size:12px;color:var(--muted);margin-left:auto}
.pbar-track{height:6px;border-radius:999px;background:var(--border);overflow:hidden;margin-top:6px}
.pbar-fill{height:100%;border-radius:999px;background:linear-gradient(90deg,var(--blue),var(--purple));
  transition:width .4s ease;width:0%}
.block-meta{display:flex;gap:16px;flex-wrap:wrap;margin-top:8px}
.block-meta-item{display:flex;flex-direction:column;gap:2px}
.bm-label{font-size:10px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}
.bm-value{font-size:12px;color:var(--text)}
.bm-value.mono{font-family:'SF Mono',Monaco,monospace;font-size:11px}

/* TWO COL */
.two-col{display:grid;grid-template-columns:1fr 1fr;gap:14px}
@media(max-width:900px){.two-col{grid-template-columns:1fr}}

/* CARDS */
.card{background:var(--surface);border:1px solid var(--border);border-radius:10px;overflow:hidden}
.card-header{
  padding:10px 14px;border-bottom:1px solid var(--border);
  display:flex;align-items:center;justify-content:space-between;
}
.card-title{font-size:12px;font-weight:600;color:#fff;text-transform:uppercase;letter-spacing:.07em}
.card-badge{font-size:11px;color:var(--muted)}

/* TABLE */
.tbl-wrap{overflow-x:auto;max-height:260px;overflow-y:auto}
table{width:100%;border-collapse:collapse}
th{
  padding:7px 10px;text-align:left;font-size:11px;font-weight:500;
  color:var(--muted);text-transform:uppercase;letter-spacing:.05em;
  background:var(--surface2);position:sticky;top:0;white-space:nowrap;
}
td{padding:7px 10px;border-top:1px solid var(--border);font-size:12px;white-space:nowrap}
tr:hover td{background:rgba(255,255,255,.025)}
.mono{font-family:'SF Mono',Monaco,monospace;font-size:11px}

/* PILLS */
.pill{display:inline-flex;align-items:center;padding:2px 7px;border-radius:999px;font-size:11px;font-weight:500}
.pill.ok{background:rgba(63,185,80,.15);color:var(--green)}
.pill.err{background:rgba(248,81,73,.15);color:var(--red)}
.pill.warn{background:rgba(210,153,34,.15);color:var(--yellow)}
.pill.blue{background:rgba(88,166,255,.15);color:var(--blue)}

/* IFAR coloring */
.sig-high{color:var(--green);font-weight:600}
.sig-med{color:var(--yellow)}
.sig-low{color:var(--muted)}

/* LOG */
.log-box{
  font-family:'SF Mono',Monaco,monospace;font-size:11px;line-height:1.6;
  padding:10px 12px;max-height:260px;overflow-y:auto;
  background:var(--surface2);color:var(--muted);white-space:pre-wrap;word-break:break-all;
}
.log-INFO{color:#8b949e}
.log-WARNING{color:var(--yellow)}
.log-ERROR{color:var(--red)}
.log-DEBUG{color:#3d4450}

/* TIMELINE */
.timeline{display:flex;height:18px;border-radius:6px;overflow:hidden;background:var(--border);gap:1px}
.tl-seg{flex:1;display:flex;align-items:center;justify-content:center;font-size:9px;
  font-weight:600;letter-spacing:.04em;transition:opacity .2s;cursor:default}
.tl-seg:hover{opacity:.8}
.tl-future{background:var(--border);color:var(--muted)}
.tl-O1{background:rgba(88,166,255,.5);color:#fff}
.tl-O2{background:rgba(88,166,255,.65);color:#fff}
.tl-O3a{background:rgba(188,140,255,.5);color:#fff}
.tl-O3b{background:rgba(188,140,255,.65);color:#fff}
.tl-O4a{background:rgba(57,197,207,.6);color:#fff}
.tl-O4b{background:rgba(255,160,50,.7);color:#fff}
.tl-current{box-shadow:inset 0 0 0 2px #fff;opacity:1!important}
.tl-done{opacity:.45}
.timeline-labels{display:flex;margin-top:4px}
.tl-label{flex:1;text-align:center;font-size:10px;color:var(--muted)}

footer{text-align:center;padding:16px;color:var(--muted);font-size:11px;border-top:1px solid var(--border);margin-top:8px}
</style>
</head>
<body>

<header>
  <div class="logo">
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none">
      <circle cx="14" cy="14" r="13" stroke="#58a6ff" stroke-width="1.5" opacity=".4"/>
      <circle cx="14" cy="14" r="8" stroke="#bc8cff" stroke-width="1.5" opacity=".5"/>
      <circle cx="14" cy="14" r="3" fill="#58a6ff"/>
      <line x1="14" y1="1" x2="14" y2="6" stroke="#58a6ff" stroke-width="1.5" opacity=".6"/>
      <line x1="14" y1="22" x2="14" y2="27" stroke="#58a6ff" stroke-width="1.5" opacity=".6"/>
      <line x1="1" y1="14" x2="6" y2="14" stroke="#58a6ff" stroke-width="1.5" opacity=".6"/>
      <line x1="22" y1="14" x2="27" y2="14" stroke="#58a6ff" stroke-width="1.5" opacity=".6"/>
    </svg>
    <div class="logo-text">
      <h1>GW Scout</h1>
      <p>Matched-filter gravitational wave search</p>
    </div>
  </div>
  <div class="header-right">
    <div class="dot" id="live-dot"></div>
    <span class="refresh-label" id="last-refresh">–</span>
  </div>
</header>

<main>

  <!-- STAT CHIPS -->
  <div class="stats-row" id="stats-row">
    <div class="stat-chip highlight">
      <span class="stat-label">Current GPS</span>
      <span class="stat-value mono" id="chip-gps">–</span>
      <span class="stat-sub" id="chip-utc">–</span>
    </div>
    <div class="stat-chip">
      <span class="stat-label">Observing Run</span>
      <span class="stat-value" id="chip-run">–</span>
      <span class="stat-sub" id="chip-run-dates">–</span>
    </div>
    <div class="stat-chip">
      <span class="stat-label">Blocks Processed</span>
      <span class="stat-value" id="chip-blocks">–</span>
      <span class="stat-sub" id="chip-blocks-sub">–</span>
    </div>
    <div class="stat-chip alert" id="chip-cand-card">
      <span class="stat-label">Candidates Found</span>
      <span class="stat-value" id="chip-cand">–</span>
      <span class="stat-sub" id="chip-cand-sub">this run</span>
    </div>
  </div>

  <!-- CURRENT BLOCK -->
  <div class="block-card" id="block-card">
    <div class="block-header">
      <div class="block-header-left">
        <span class="block-title">Current Block</span>
        <span class="stage-pill" id="stage-pill">idle</span>
      </div>
      <span class="pct-label" id="pct-label">0%</span>
    </div>
    <div class="pbar-track"><div class="pbar-fill" id="pbar"></div></div>
    <div class="block-meta" id="block-meta">
      <div class="block-meta-item"><span class="bm-label">Start</span><span class="bm-value mono" id="bm-start">–</span></div>
      <div class="block-meta-item"><span class="bm-label">End</span><span class="bm-value mono" id="bm-end">–</span></div>
      <div class="block-meta-item"><span class="bm-label">UTC Start</span><span class="bm-value" id="bm-utc">–</span></div>
      <div class="block-meta-item"><span class="bm-label">Stage msg</span><span class="bm-value" id="bm-msg">–</span></div>
    </div>
  </div>

  <!-- SEARCH TIMELINE -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">Search Timeline</span>
      <span class="card-badge">O1 → O4b (backward)</span>
    </div>
    <div style="padding:12px 14px">
      <div class="timeline" id="timeline">
        <div class="tl-seg tl-O1" title="O1: 2015–2016">O1</div>
        <div class="tl-seg tl-O2" title="O2: 2017">O2</div>
        <div class="tl-seg tl-O3a" title="O3a: 2019">O3a</div>
        <div class="tl-seg tl-O3b" title="O3b: 2019–2020">O3b</div>
        <div class="tl-seg tl-O4a" title="O4a: 2023–2024">O4a</div>
        <div class="tl-seg tl-O4b" title="O4b1Disc: Oct 2024" style="flex:.15">4b①</div>
        <div class="tl-seg tl-O4b" title="O4b2Disc: Nov 2024" style="flex:.15">4b②</div>
        <div class="tl-seg tl-O4b" title="O4b3Disc: Jan 2025" style="flex:.15">4b③</div>
      </div>
      <div class="timeline-labels">
        <span class="tl-label">2015</span>
        <span class="tl-label">2017</span>
        <span class="tl-label">2019</span>
        <span class="tl-label">2020</span>
        <span class="tl-label">2024</span>
        <span class="tl-label" style="flex:.45">O4b</span>
      </div>
    </div>
  </div>

  <!-- CANDIDATES + LOG -->
  <div class="two-col">
    <div class="card">
      <div class="card-header">
        <span class="card-title">Candidates</span>
        <span class="card-badge" id="cand-count">0 total</span>
      </div>
      <div class="tbl-wrap">
        <table>
          <thead><tr>
            <th>UTC Time</th><th>IFOs</th><th>SNR</th><th>IFAR</th><th>Known</th>
          </tr></thead>
          <tbody id="cand-body"></tbody>
        </table>
      </div>
    </div>
    <div class="card">
      <div class="card-header">
        <span class="card-title">Live Log</span>
        <span class="card-badge">last 60 lines</span>
      </div>
      <div class="log-box" id="log-box">Loading...</div>
    </div>
  </div>

  <!-- BLOCKS TABLE -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">Recent Blocks</span>
      <span class="card-badge" id="block-count">–</span>
    </div>
    <div class="tbl-wrap">
      <table>
        <thead><tr>
          <th>UTC Start</th><th>Duration</th><th>Status</th><th>Note</th>
        </tr></thead>
        <tbody id="block-body"></tbody>
      </table>
    </div>
  </div>

</main>
<footer>GW Scout &mdash; auto-refresh every 5s</footer>

<script>
// GPS epoch offset: GPS=0 is Jan 6 1980 00:00:00 UTC = Unix 315964800
// Also subtract current leap seconds (18 since 2017)
const GPS_UNIX_OFFSET = 315964800 - 18;

function gpsToDate(gps) {
  if (!gps) return null;
  return new Date((gps + GPS_UNIX_OFFSET) * 1000);
}
function gpsToUTC(gps, short=false) {
  const d = gpsToDate(gps);
  if (!d) return '–';
  if (short) return d.toISOString().replace('T',' ').slice(0,19) + ' UTC';
  return d.toISOString().replace('T',' ').slice(0,23) + ' UTC';
}
function gpsToRun(gps) {
  if (!gps) return {run:'–',dates:'–'};
  if (gps >= 1420877824 && gps < 1420881920) return {run:'O4b3Disc',dates:'Jan 2025'};
  if (gps >= 1415274496 && gps < 1415278592) return {run:'O4b2Disc',dates:'Nov 2024'};
  if (gps >= 1412722688 && gps < 1412726784) return {run:'O4b1Disc',dates:'Oct 2024'};
  if (gps >= 1368195220 && gps < 1389456018) return {run:'O4a',dates:'May 2023 – Jan 2024'};
  if (gps >= 1256655618 && gps < 1269363618) return {run:'O3b',dates:'Nov 2019 – Mar 2020'};
  if (gps >= 1238166018 && gps < 1253977218) return {run:'O3a',dates:'Apr – Oct 2019'};
  if (gps >= 1164556817 && gps < 1187733618) return {run:'O2',dates:'Nov 2016 – Aug 2017'};
  if (gps >= 1126051217 && gps < 1137254417) return {run:'O1',dates:'Sep 2015 – Jan 2016'};
  return {run:'?',dates:'unknown'};
}
function fmtIfar(days) {
  if (days == null) return '–';
  if (days > 365) return `<span class="sig-high">${(days/365).toFixed(1)} yr</span>`;
  if (days > 30)  return `<span class="sig-high">${days.toFixed(1)} days</span>`;
  if (days > 1)   return `<span class="sig-med">${days.toFixed(2)} days</span>`;
  return `<span class="sig-low">${(days*24).toFixed(2)} hr</span>`;
}
function pill(t,cls){return`<span class="pill ${cls}">${t}</span>`}

function colorLog(line) {
  if (line.includes(' ERROR '))   return `<span class="log-ERROR">${escHtml(line)}</span>`;
  if (line.includes(' WARNING ')) return `<span class="log-WARNING">${escHtml(line)}</span>`;
  if (line.includes(' DEBUG '))   return `<span class="log-DEBUG">${escHtml(line)}</span>`;
  return `<span class="log-INFO">${escHtml(line)}</span>`;
}
function escHtml(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;')}

let prevCandCount = 0;

async function refresh() {
  try {
    const [status, cands, blocks, logsResp] = await Promise.all([
      fetch('/status').then(r=>r.json()),
      fetch('/candidates?limit=50').then(r=>r.json()),
      fetch('/blocks?limit=30').then(r=>r.json()),
      fetch('/logs').then(r=>r.text()),
    ]);

    // --- Live dot ---
    document.getElementById('last-refresh').textContent =
      new Date().toLocaleTimeString();

    // --- Progress ---
    const p = status.progress || {};
    const pct = typeof p.pct === 'number' ? Math.max(0,Math.min(1,p.pct)) : 0;
    const stage = p.stage || 'idle';
    document.getElementById('pbar').style.width = (pct*100)+'%';
    document.getElementById('pct-label').textContent = (pct*100).toFixed(0)+'%';
    const stagePill = document.getElementById('stage-pill');
    stagePill.textContent = stage;
    stagePill.className = 'stage-pill' + (stage==='done'?' done':stage==='error'?' error':'');
    document.getElementById('bm-msg').textContent = p.msg || '–';

    const gpsStart = p.block_start;
    const gpsEnd = p.block_end;
    document.getElementById('bm-start').textContent = gpsStart ? gpsStart.toFixed(1) : '–';
    document.getElementById('bm-end').textContent   = gpsEnd   ? gpsEnd.toFixed(1)   : '–';
    document.getElementById('bm-utc').textContent   = gpsToUTC(gpsStart, true);

    // --- Stat chips ---
    document.getElementById('chip-gps').textContent = gpsStart ? gpsStart.toFixed(0) : '–';
    document.getElementById('chip-utc').textContent = gpsToUTC(gpsStart, true);
    const runInfo = gpsToRun(gpsStart);
    document.getElementById('chip-run').textContent = runInfo.run;
    document.getElementById('chip-run-dates').textContent = runInfo.dates;
    const done  = status.block_done  ?? 0;
    const total = status.block_count ?? 0;
    const err   = status.block_error ?? 0;
    document.getElementById('chip-blocks').textContent = done;
    document.getElementById('chip-blocks-sub').textContent = `${err} error${err!==1?'s':''} · ${total} total`;
    const candCount = status.candidate_count ?? 0;
    document.getElementById('chip-cand').textContent = candCount;
    document.getElementById('cand-count').textContent = candCount + ' total';
    if (candCount > prevCandCount) {
      const chip = document.getElementById('chip-cand-card');
      chip.style.borderColor = 'rgba(63,185,80,.8)';
      chip.style.background  = 'rgba(63,185,80,.15)';
      setTimeout(()=>{chip.style.borderColor='';chip.style.background=''},3000);
    }
    prevCandCount = candCount;

    // --- Timeline highlight ---
    const runClass = {'O1':'tl-O1','O2':'tl-O2','O3a':'tl-O3a','O3b':'tl-O3b','O4a':'tl-O4a','O4b1Disc':'tl-O4b','O4b2Disc':'tl-O4b','O4b3Disc':'tl-O4b'};
    document.querySelectorAll('.tl-seg').forEach(el=>{
      const title = el.getAttribute('title')||'';
      const segRun = Object.keys(runClass).find(k=>title.startsWith(k));
      el.classList.remove('tl-current','tl-done');
      if (segRun === runInfo.run) el.classList.add('tl-current');
    });

    // --- Candidates ---
    const cbody = document.getElementById('cand-body');
    cbody.innerHTML = '';
    cands.forEach(r => {
      const ifar = typeof r.ifar_days === 'number' ? r.ifar_days : null;
      const snr  = typeof r.network_stat === 'number' ? r.network_stat.toFixed(2) : r.network_stat;
      const known = r.known_event ? pill(r.known_event,'blue') : '';
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="mono">${gpsToUTC(r.gps,true)}</td>
        <td>${(r.ifos||[]).join(',')}</td>
        <td><strong>${snr}</strong></td>
        <td>${fmtIfar(ifar)}</td>
        <td>${known}</td>`;
      cbody.appendChild(tr);
    });

    // --- Blocks ---
    const bbody = document.getElementById('block-body');
    bbody.innerHTML = '';
    document.getElementById('block-count').textContent =
      `${done} done · ${err} error`;
    blocks.forEach(r => {
      const dur = r.gps_end && r.gps_start ? (r.gps_end-r.gps_start).toFixed(0)+'s' : '–';
      const sp  = r.status==='done'  ? pill('done','ok')
                : r.status==='error' ? pill('error','err')
                :                      pill(r.status,'warn');
      const note = r.error ? `<span style="color:var(--red);max-width:300px;overflow:hidden;text-overflow:ellipsis;display:inline-block">${escHtml(r.error.slice(0,80))}</span>` : '';
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td class="mono">${gpsToUTC(r.gps_start,true)}</td>
        <td>${dur}</td>
        <td>${sp}</td>
        <td>${note}</td>`;
      bbody.appendChild(tr);
    });

    // --- Logs ---
    const lines = logsResp.split('\n').slice(-60);
    document.getElementById('log-box').innerHTML =
      lines.map(colorLog).join('\n');
    const lb = document.getElementById('log-box');
    lb.scrollTop = lb.scrollHeight;

  } catch(e) {
    console.error(e);
    document.getElementById('live-dot').style.background = 'var(--red)';
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
