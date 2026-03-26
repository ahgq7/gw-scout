from pathlib import Path

from gwsearch import state


def test_state_persistence(tmp_path: Path):
    db = tmp_path / "state.sqlite"
    mgr = state.StateManager(db)
    run_id = mgr.create_run("hash123")
    mgr.record_block(run_id, ["H1", "L1"], 0.0, 10.0, status="done")
    mgr.record_candidate(run_id, gps=5.0, ifos=["H1", "L1"], template_id="t0", network_stat=9.0, far=1e-4, ifar_days=1.0, payload={"gps": 5.0})
    mgr.record_background(run_id, slide=0.5, far=1e-4, payload={"samples": [0.1]})
    last = mgr.get_last_block_end(run_id)
    assert last == 10.0
