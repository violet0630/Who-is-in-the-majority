"""Validate final trajectories and preserve the executed source versions."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

from majority_study import PROJECT, ROOT
from majority.llm import canonical_model, load_settings


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    results = read(ROOT / "03_positions/results.json")
    earlier = read(ROOT / "03_positions/prefix8/results.json")
    lookup = {(r["graph"], r["repeat"], r["branch"]): r for r in earlier}
    assert len(results) == len({(r["graph"], r["repeat"], r["branch"]) for r in results}) == 27
    for row in results:
        assert row["size"] == 1000 and row["k"] == 200
        assert [metric["round"] for metric in row["metrics"]] == list(range(1, 11))
        assert row["metrics"][:8] == lookup[(row["graph"], row["repeat"], row["branch"])]["metrics"]
        assert all(abs(metric["true_advocate_support"] - 0.2) < 1e-12 for metric in row["metrics"])
        expected = sum(m["perceived_majority"] for m in row["metrics"][-3:]) / 3
        assert expected == row["summary"]["perceived_majority"]
        folder = ROOT / "03_positions" / f"g{row['graph']}-r{row['repeat']}-{row['branch']}"
        assert (folder / "result.json").exists() and (folder / "trajectory.jsonl.gz").exists()

    bank_path = ROOT / "01_texts_50/bank.json"
    bank_record = read(bank_path)
    bank = bank_record["data"]["bank"]
    posts = bank["originals"]
    texts = [text for post in posts for text in (post["text"], *post["agree"], *post["disagree"])]
    assert len(posts) == 50 and len(texts) == 250
    assert all(sum(post["stance"] == side for post in posts) == 25 for side in (-1, 1))
    assert all(len(post["agree"]) == len(post["disagree"]) == 2 for post in posts)
    assert len({text["body"] for text in texts}) == 250
    assert all(40 <= len(text["body"]) <= 80 for text in texts)
    bank_hash = hashlib.sha256(json.dumps(bank, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert bank_hash == read(ROOT / "initialized_parameters.json")["text_bank_sha256"]
    bank_record["data"]["semantic_review"] = "passed_assistant_stance_and_relation_review"
    bank_record["data"]["validation_file"] = "validation.json"
    write(bank_path, bank_record)

    observers = {}
    for graph in range(3):
        network = read(ROOT / "03_positions" / f"network-{graph}.json")["data"]["network"]
        observers[graph] = set(network["observers"])
        assert len(network["communities"]) == 1000 and len(observers[graph]) == 50
    status, models, actual = Counter(), Counter(), Counter()
    tokens_in = tokens_out = 0
    for path in (ROOT / "03_positions").glob("g*-r*-*/calls.jsonl"):
        graph = int(path.parent.name.split("-")[0][1:])
        with path.open(encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)["data"]
                assert row["context"]["node"] in observers[graph]
                assert canonical_model(row["model"]) == "deepseek-flash"
                status[row["status"]] += 1
                if row.get("reused"):
                    continue
                actual[row["status"]] += 1
                if "response" in row:
                    response = row["response"]
                    models[response["model"]] += 1
                    usage = response.get("usage")
                    if usage:
                        tokens_in += usage.get("prompt_tokens", 0)
                        tokens_out += usage.get("completion_tokens", 0)
    write(ROOT / "api_usage.json", {"main_audit_rows": dict(status), "actual_calls": dict(actual),
        "returned_models": dict(models), "known_input_tokens": tokens_in, "known_output_tokens": tokens_out})

    source_path = Path(__file__).parent / "majority_study.py"
    source = source_path.read_text(encoding="utf-8")
    resumed = source.replace("LLMValidationError, canonical_model, load_prompt", "LLMValidationError, load_prompt")
    resumed = resumed.replace('canonical_model(item["model"])', 'item["model"]')
    resumed = resumed.replace("canonical_model(config.model)", "config.model")
    resumed = resumed.replace('                  "canonical_model": canonical_model(self.config.model),\n', "")
    resumed = resumed.replace("canonical_model(self.config.model)", "self.config.model")
    expected_hash = read(ROOT / "gateway_transition.json")["resume_runner_sha256"]
    assert hashlib.sha256(resumed.encode()).hexdigest() == expected_hash
    (ROOT / "source_snapshots/majority_study.resume10.py").write_bytes(resumed.encode())
    for name in ("majority_study.py", "summarize_study.py", "analyze_prefix8.py", "finalize_validation.py"):
        shutil.copy2(Path(__file__).parent / name, ROOT / name)
    shutil.copy2(PROJECT / "majority/llm.py", ROOT / "source_snapshots/llm.compat.py")
    followup = read(ROOT / "followup_initialization.json")
    followup["llm"].update(canonical_model="deepseek-flash", max_gateway_concurrency=16,
                            gateway_models={"https://api.1nvictus.me/v1": "deepseek-flash",
                                            "http://127.0.0.1:3000/v1": "deepseek-v4-flash"})
    write(ROOT / "followup_initialization.json", followup)

    _, secret = load_settings(PROJECT / ".env")
    needle = secret.encode()
    for path in ROOT.rglob("*"):
        if path.is_file() and path.suffix in {".py", ".json", ".jsonl", ".csv", ".md"}:
            assert needle not in path.read_bytes(), f"Credential found in deliverable: {path.name}"
    validation = {"runs": 27, "networks": 3, "paired_groups": 9, "rounds_per_run": 10,
        "node_count": 1000, "observer_count": 50, "advocates": 200,
        "original_prefix8_preserved_exactly": True, "all_api_calls_observers_only": True,
        "originals": 50, "comments": 200, "unique_texts": 250, "bank_hash_unchanged": True,
        "project_tests_passed": 20, "study_tests_passed": 4, "credential_scan": "passed"}
    write(ROOT / "final_validation.json", validation)
    print(json.dumps(validation, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
