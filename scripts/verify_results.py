"""Verify the published archive, study design, and observer-only API calls."""

import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "experiments/2026-11-16"


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    manifest = read(ROOT / "archive_manifest.json")
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        assert path.stat().st_size == entry["bytes"], entry["path"]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == entry["sha256"], entry["path"]

    results = read(ROOT / "03_positions/results.json")
    earlier = read(ROOT / "03_positions/prefix8/results.json")
    prefixes = {(row["graph"], row["repeat"], row["branch"]): row["metrics"] for row in earlier}
    assert len(results) == len({(row["graph"], row["repeat"], row["branch"]) for row in results}) == 27
    for row in results:
        assert row["size"] == 1000 and row["k"] == 200
        assert [metric["round"] for metric in row["metrics"]] == list(range(1, 11))
        assert row["metrics"][:8] == prefixes[(row["graph"], row["repeat"], row["branch"])]
        assert all(abs(metric["true_advocate_support"] - 0.2) < 1e-12 for metric in row["metrics"])
        tail = sum(metric["perceived_majority"] for metric in row["metrics"][-3:]) / 3
        assert tail == row["summary"]["perceived_majority"]

    bank = read(ROOT / "01_texts_50/bank.json")["data"]["bank"]
    posts = bank["originals"]
    assert len(posts) == 50
    assert all(sum(post["stance"] == side for post in posts) == 25 for side in (-1, 1))
    assert all(len(post["agree"]) == len(post["disagree"]) == 2 for post in posts)
    texts = [text["body"] for post in posts for text in (post["text"], *post["agree"], *post["disagree"])]
    assert len(texts) == len(set(texts)) == 250
    assert all(40 <= len(text) <= 80 for text in texts)
    digest = hashlib.sha256(json.dumps(bank, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    assert digest == read(ROOT / "initialized_parameters.json")["text_bank_sha256"]

    observers = {graph: set(read(ROOT / f"03_positions/network-{graph}.json")["data"]["network"]["observers"])
                 for graph in range(3)}
    assert all(len(nodes) == 50 for nodes in observers.values())
    calls = sorted((ROOT / "03_positions").glob("g*-r*-*/calls.jsonl.gz"))
    assert len(calls) == 27
    for path in calls:
        graph = int(path.parent.name.split("-")[0][1:])
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            for line in stream:
                assert json.loads(line)["data"]["context"]["node"] in observers[graph], path.parent.name
    print(json.dumps({"archive_files_verified": len(manifest["files"]), "runs": 27, "rounds": 10,
                      "originals": 50, "comments": 200, "observer_only_api": True}))


if __name__ == "__main__":
    main()
