import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List, Optional


class ExperimentStore:
    """Persistent SQLite database tracking all AutoML experiment runs and evaluated candidates."""

    def __init__(self, db_path: str = "results/experiments.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    timestamp TEXT,
                    git_hash TEXT,
                    strategy TEXT,
                    dataset TEXT,
                    config_json TEXT,
                    device_info_json TEXT,
                    total_evaluations INTEGER,
                    wall_clock_s REAL
                )
            """)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT,
                    run_id TEXT,
                    generation INTEGER,
                    config_json TEXT,
                    metrics_json TEXT,
                    pareto_rank INTEGER,
                    crowding_distance REAL,
                    failed INTEGER,
                    failure_reason TEXT,
                    PRIMARY KEY (run_id, candidate_id),
                    FOREIGN KEY (run_id) REFERENCES runs (run_id)
                )
            """)
            conn.commit()

    def record_run(
        self,
        run_id: str,
        timestamp: str,
        git_hash: str,
        strategy: str,
        dataset: str,
        config: Dict[str, Any],
        device_info: Dict[str, Any],
        total_evaluations: int = 0,
        wall_clock_s: float = 0.0,
    ) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO runs (
                    run_id, timestamp, git_hash, strategy, dataset,
                    config_json, device_info_json, total_evaluations, wall_clock_s
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run_id,
                timestamp,
                git_hash,
                strategy,
                dataset,
                json.dumps(config),
                json.dumps(device_info),
                total_evaluations,
                wall_clock_s,
            ))
            conn.commit()

    def record_candidate(
        self,
        run_id: str,
        candidate_id: str,
        generation: int,
        config_dict: Dict[str, Any],
        metrics_dict: Dict[str, Any],
        pareto_rank: int = 0,
        crowding_distance: float = 0.0,
        failed: bool = False,
        failure_reason: Optional[str] = None,
    ) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO candidates (
                    candidate_id, run_id, generation, config_json, metrics_json,
                    pareto_rank, crowding_distance, failed, failure_reason
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                candidate_id,
                run_id,
                generation,
                json.dumps(config_dict),
                json.dumps(metrics_dict),
                pareto_rank,
                crowding_distance,
                1 if failed else 0,
                failure_reason,
            ))
            conn.commit()

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,))
            row = cursor.fetchone()
            if not row:
                return None
            data = dict(row)
            data["config"] = json.loads(data["config_json"])
            data["device_info"] = json.loads(data["device_info_json"])
            return data

    def get_candidates(self, run_id: str) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM candidates WHERE run_id = ?", (run_id,))
            rows = cursor.fetchall()
            results = []
            for r in rows:
                d = dict(r)
                d["config"] = json.loads(d["config_json"])
                d["metrics"] = json.loads(d["metrics_json"])
                d["failed"] = bool(d["failed"])
                results.append(d)
            return results

    def get_all_runs(self) -> List[Dict[str, Any]]:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM runs ORDER BY timestamp DESC")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]
