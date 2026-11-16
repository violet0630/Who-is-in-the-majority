"""Analyze the common completed eight-round prefix after the gateway outage."""

from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil

from majority_study import ROOT, infer_positions, save_json


def main():
    destination = ROOT / "03_positions"
    amendment = {
        "recorded_at": datetime.now(timezone(timedelta(hours=8))).isoformat(),
        "planned_rounds": 10, "analysis_horizon": 8, "user_permitted_rounds": [5, 10],
        "reason": "Configured gateway returned HTTP 502: Request failed on all available accounts; short health checks also failed after concurrency stopped.",
        "rule": "Use the longest common completed prefix across all 27 runs, before calculating inference.",
        "primary_window": [6, 7, 8], "early_window": [3, 4, 5],
        "original_attempts_retained": True,
    }
    save_json(destination / "analysis_amendment.json", amendment)
    results = []
    for graph in range(3):
        for repeat in range(3):
            prepared = json.loads((destination / f"g{graph}-r{repeat}_positions.json").read_text(encoding="utf-8"))
            for branch in ("TH", "TL", "U0"):
                folder = destination / f"g{graph}-r{repeat}-{branch}"
                source = next(folder.glob("trajectory*.jsonl.gz"))
                metrics, diagnostics = [], []
                with gzip.open(source, "rt", encoding="utf-8") as stream:
                    manifest = json.loads(next(stream))["data"]
                    for line in stream:
                        if "metrics" not in line[:30] and "diagnostics" not in line[:30]:
                            continue
                        record = json.loads(line)
                        if record["data"]["round"] <= 8:
                            (metrics if record["kind"] == "metrics" else diagnostics).append(record["data"])
                assert [row["round"] for row in metrics] == list(range(1, 9))
                assert len(diagnostics) == 8
                assert len(manifest["network"]["communities"]) == 1000
                assert len(manifest["network"]["observers"]) == 50
                assert len(manifest["advocates"]) == 200
                assert all(abs(row["true_advocate_support"] - 0.2) < 1e-12 for row in metrics)
                summary = {name: sum(row[name] for row in metrics[-3:]) / 3 for name in (
                    "perceived_majority", "feed_majority", "advocate_perception_bias", "public_conformity", "opponent_silence")}
                results.append({"graph": graph, "repeat": repeat, "branch": branch,
                    "seed": manifest["group_seed"], "size": 1000, "k": 200,
                    "settings": {**manifest["settings"], "rounds": 8, "tail_rounds": 3},
                    "source_planned_rounds": 10, "source_trajectory": str(source),
                    "analysis_horizon": 8, "tail_rounds": 3, "summary": summary,
                    "metrics": metrics, "diagnostics": diagnostics,
                    "seed_diagnostics": prepared["diagnostics"][branch]})
    assert len(results) == 27
    protocol = json.loads((destination / "protocol.json").read_text(encoding="utf-8"))
    protocol.update(settings={**protocol["settings"], "rounds": 8},
                    primary_outcome="mean observer perceived_majority in rounds 6-8",
                    time_analysis="paired change: rounds 6-8 minus rounds 3-5",
                    analysis_horizon=8, amendment="analysis_amendment.json")
    save_json(destination / "analysis_protocol.json", protocol)
    save_json(destination / "results.json", results)
    infer_positions(results, destination, horizon=8)
    executed = ROOT / "majority_study.py"
    shutil.copy2(executed, ROOT / "majority_study.executed10.py")
    current = Path(__file__).parent / "majority_study.py"
    shutil.copy2(current, executed)
    save_json(ROOT / "analysis_execution_manifest.json", {
        "analysis_horizon": 8, "executed_simulation_source": "majority_study.executed10.py",
        "inference_source_sha256": hashlib.sha256(current.read_bytes()).hexdigest(),
        "prefix_extractor_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "complete_prefixes": 27, "round_metrics": 216,
    })
    print((destination / "inference.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
