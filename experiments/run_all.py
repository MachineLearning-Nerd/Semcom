"""Run all hypothesis experiments and render a markdown verdict table."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from secom.results import render_table

ARTIFACTS = ROOT / "artifacts"
DOCS = ROOT / "docs"
RESULTS = DOCS / "RESULTS.md"


def main() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    (ARTIFACTS / "results.jsonl").unlink(missing_ok=True)

    scripts = sorted((ROOT / "experiments").glob("exp_h*.py"))
    for script in scripts:
        print(f"\n===== {script.name} =====")
        subprocess.run([sys.executable, str(script)], check=True)

    table = render_table()
    DOCS.mkdir(exist_ok=True)
    RESULTS.write_text(
        "\n".join(
            [
                f"# SECOM Hypothesis Results",
                f"Generated: {datetime.now().isoformat()}",
                "",
                table,
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\nWrote {RESULTS}")
    print(table)


if __name__ == "__main__":
    main()
