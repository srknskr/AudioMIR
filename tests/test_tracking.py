import tempfile
from pathlib import Path

from automir.experiments.sqlite_store import ExperimentStore


def test_sqlite_run_and_candidate_persistence():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_experiments.sqlite"
        store = ExperimentStore(str(db_path))

        run_id = "test_run_01"
        store.record_run(
            run_id=run_id,
            timestamp="2026-08-17T10:00:00",
            git_hash="abcdef1",
            strategy="evolutionary",
            dataset="synthetic",
            config={"seed": 42},
            device_info={"device_name": "CPU"},
            total_evaluations=2,
            wall_clock_s=12.5,
        )

        store.record_candidate(
            run_id=run_id,
            candidate_id="cand_01",
            generation=0,
            config_dict={"representation": "logmel", "conv_blocks": 3},
            metrics_dict={"tempo_acc_4": 91.2, "latency_ms": 6.5},
            pareto_rank=0,
            crowding_distance=1.5,
            failed=False,
        )

        # Retrieve run
        run_data = store.get_run(run_id)
        assert run_data is not None
        assert run_data["strategy"] == "evolutionary"
        assert run_data["config"]["seed"] == 42

        # Retrieve candidates
        cands = store.get_candidates(run_id)
        assert len(cands) == 1
        assert cands[0]["candidate_id"] == "cand_01"
        assert cands[0]["metrics"]["tempo_acc_4"] == 91.2
        assert not cands[0]["failed"]
