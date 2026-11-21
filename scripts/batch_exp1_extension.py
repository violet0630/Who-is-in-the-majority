"""Run the frozen supplemental tasks after the technical trial passes its audit."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

from scripts.audit_exp1_extension import audit
from scripts.run_exp1_extension import read


def run_remaining(repo, prepared, runs):
    pilot = audit(repo, prepared, runs, trial=True)
    returned = {model for row in pilot["trajectories"] for model in row["response_models"]}
    if returned != {"deepseek-flash"}:
        raise ValueError(f"Technical trial needs model review: {sorted(returned)}")

    tasks = [task["task_id"] for task in read(prepared / "tasks.json")
             if task["phase"] == "remaining"]
    # The frozen per-run worker count is four. Alongside the main experiment two
    # runner, two trajectories here leave the combined gateway load at 16.
    def execute(task):
        destination = runs / task
        if (destination / "trajectory.jsonl.gz").is_file() and (destination / "result.json").is_file():
            return {"task": task, "status": "existing"}
        if destination.exists():
            return {"task": task, "status": "incomplete_existing"}
        command = [sys.executable, "-B", "-u", "-m", "scripts.run_exp1_extension", "--repo", str(repo),
                   "--prepared", str(prepared), "--task", task, "--runs", str(runs), "--execute"]
        code = subprocess.run(command, cwd=repo).returncode
        return {"task": task, "status": "failed" if code else "finished"}

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = list(pool.map(execute, tasks))
    print(json.dumps({"statuses": statuses}, ensure_ascii=False), flush=True)
    if any(row["status"] in {"failed", "incomplete_existing"} for row in statuses):
        raise RuntimeError("Batch has incomplete trajectories; check the task logs and preserve failed attempts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--runs", type=Path, required=True)
    parser.add_argument("--allow-api", action="store_true")
    args = parser.parse_args()
    if not args.allow_api:
        parser.error("Provide --allow-api after reviewing the technical trial")
    run_remaining(args.repo.resolve(), args.prepared.resolve(), args.runs.resolve())


if __name__ == "__main__":
    main()
