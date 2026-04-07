"""
Experiment runner with SQLite tracking.
Runs training experiments and records results in a database.

Usage:
    uv run -m src.experiment --config configs/baseline.toml
    uv run -m src.experiment --config configs/baseline.toml --description "SwiGLU test"
"""

import argparse
import json
import sqlite3
import subprocess
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from src.model.config import ExperimentConfig

DB_PATH = Path("experiments.db")


def init_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Initialize SQLite database with experiment tables."""
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            commit_hash TEXT,
            config_json TEXT NOT NULL,
            val_bpb REAL,
            peak_memory_mb REAL,
            training_seconds REAL,
            mfu_percent REAL,
            total_tokens_m REAL,
            num_params_m REAL,
            num_steps INTEGER,
            status TEXT NOT NULL DEFAULT 'running',
            description TEXT,
            error_message TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS experiment_queue (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            config_json TEXT NOT NULL,
            description TEXT,
            priority INTEGER DEFAULT 0,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def get_git_hash() -> str:
    """Get current short git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short=7", "HEAD"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() if result.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class ExperimentRunner:
    """Manages experiment lifecycle with SQLite tracking."""

    def __init__(self, db_path: Path = DB_PATH):
        self.conn = init_db(db_path)

    def run_experiment(self, config: ExperimentConfig, description: str = "") -> dict:
        """Run a single experiment and record results."""
        desc = description or config.description
        config_json = json.dumps(config.to_dict(), indent=2)

        # Insert running experiment
        cursor = self.conn.execute(
            "INSERT INTO experiments (started_at, commit_hash, config_json, status, description) "
            "VALUES (?, ?, ?, 'running', ?)",
            (now_iso(), get_git_hash(), config_json, desc),
        )
        exp_id = cursor.lastrowid
        self.conn.commit()
        print(f"Experiment #{exp_id}: {desc}")

        try:
            from src.train import train
            results = train(config)

            status = results.get("status", "ok")
            if status == "crash":
                self.conn.execute(
                    "UPDATE experiments SET finished_at=?, status='crash', "
                    "error_message=?, num_steps=?, num_params_m=? WHERE id=?",
                    (now_iso(), results.get("error", "unknown"),
                     results.get("num_steps"), results.get("num_params_M"), exp_id),
                )
            else:
                self.conn.execute(
                    "UPDATE experiments SET finished_at=?, val_bpb=?, peak_memory_mb=?, "
                    "training_seconds=?, mfu_percent=?, total_tokens_m=?, num_params_m=?, "
                    "num_steps=?, status='done' WHERE id=?",
                    (now_iso(), results.get("val_bpb"), results.get("peak_vram_mb"),
                     results.get("training_seconds"), results.get("mfu_percent"),
                     results.get("total_tokens_M"), results.get("num_params_M"),
                     results.get("num_steps"), exp_id),
                )
            self.conn.commit()
            return results

        except Exception as e:
            tb = traceback.format_exc()
            self.conn.execute(
                "UPDATE experiments SET finished_at=?, status='crash', error_message=? WHERE id=?",
                (now_iso(), f"{e}\n{tb}", exp_id),
            )
            self.conn.commit()
            print(f"Experiment #{exp_id} crashed: {e}")
            return {"status": "crash", "error": str(e)}

    def get_best_bpb(self) -> float | None:
        """Get the best val_bpb from completed experiments."""
        row = self.conn.execute(
            "SELECT MIN(val_bpb) FROM experiments WHERE status='done' AND val_bpb > 0"
        ).fetchone()
        return row[0] if row and row[0] else None

    def enqueue(self, config: ExperimentConfig, description: str = "", priority: int = 0):
        """Add an experiment to the queue."""
        config_json = json.dumps(config.to_dict(), indent=2)
        self.conn.execute(
            "INSERT INTO experiment_queue (config_json, description, priority, created_at) "
            "VALUES (?, ?, ?, ?)",
            (config_json, description, priority, now_iso()),
        )
        self.conn.commit()

    def pop_queue(self) -> tuple | None:
        """Pop the highest-priority experiment from the queue."""
        row = self.conn.execute(
            "SELECT id, config_json, description FROM experiment_queue "
            "ORDER BY priority DESC, id ASC LIMIT 1"
        ).fetchone()
        if row:
            self.conn.execute("DELETE FROM experiment_queue WHERE id=?", (row[0],))
            self.conn.commit()
            return row
        return None

    def close(self):
        self.conn.close()


def main():
    parser = argparse.ArgumentParser(description="Run a training experiment with tracking")
    parser.add_argument("--config", type=str, required=True, help="TOML config path")
    parser.add_argument("--description", type=str, default="", help="Experiment description")
    parser.add_argument("--db", type=str, default="experiments.db", help="SQLite DB path")
    args = parser.parse_args()

    config = ExperimentConfig.from_toml(args.config)
    if args.description:
        config.description = args.description

    runner = ExperimentRunner(Path(args.db))
    try:
        results = runner.run_experiment(config, args.description)
        best = runner.get_best_bpb()
        if best is not None:
            print(f"\nBest val_bpb so far: {best:.6f}")
    finally:
        runner.close()


if __name__ == "__main__":
    main()
