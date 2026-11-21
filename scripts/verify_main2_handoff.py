"""Verify the published main experiment 2 handoff using its frozen code."""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from zipfile import ZipFile


ROOT = Path(__file__).resolve().parents[1]
BUNDLE_DIR = ROOT / "experiments/2026-11-18-main2-preparation"


def verify(output: Path) -> None:
    info = json.loads((BUNDLE_DIR / "bundle_info.json").read_text(encoding="utf-8"))
    archive = BUNDLE_DIR / info["archive"]
    if hashlib.sha256(archive.read_bytes()).hexdigest() != info["sha256"]:
        raise ValueError("Handoff ZIP hash mismatch")

    with ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [entry.filename for entry in entries]
        if len(entries) != info["file_count"] or len(set(names)) != len(names):
            raise ValueError("Unexpected ZIP inventory")
        for name in names:
            path = Path(name)
            if (not path.parts or path.parts[0] != info["archive_root"]
                    or ".." in path.parts or "\\" in name or ":" in name):
                raise ValueError(f"Invalid ZIP entry: {name}")
        output = output.resolve()
        output.mkdir(parents=True, exist_ok=False)
        bundle.extractall(output)

    prepared = output / info["archive_root"]
    # Import the exact bundled source from its own directory so the checkout
    # stays clean and Git cannot normalize the verified source bytes.
    subprocess.run(
        [sys.executable, "-B", "-c", (
            "import json, sys; from scripts.majority_factorial import load_prepared; "
            "m = load_prepared(sys.argv[1]); "
            "print(json.dumps(dict(bundle_id=m['bundle_id'], files=len(m['files']), "
            "tasks=len(m['tasks']), pilot=sum(t['pilot'] for t in m['tasks']))))"
        ), str(prepared)],
        cwd=prepared / "source",
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New extraction directory")
    verify(parser.parse_args().output)


if __name__ == "__main__":
    main()
