"""Remove participant identity fields from a leaderboard CSV.

Expected columns: position, user, team, algorithm, created, mean_position.
The output intentionally contains no user/team strings or hyperlinks.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

OUTPUT_FIELDS = ("position", "anonymous_entry", "algorithm", "created", "mean_position")


def anonymize(source: Path, destination: Path) -> None:
    with source.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    required = {"position", "user", "team", "algorithm", "created", "mean_position"}
    if not rows:
        raise ValueError("leaderboard CSV is empty")
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"leaderboard CSV lacks fields: {sorted(missing)}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS, lineterminator="\n")
        writer.writeheader()
        for index, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "position": row["position"].strip(),
                    "anonymous_entry": f"Entry-{index:02d}",
                    "algorithm": row["algorithm"].strip(),
                    "created": row["created"].strip(),
                    "mean_position": row["mean_position"].strip(),
                }
            )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    anonymize(args.source, args.destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
