from __future__ import annotations

import json
from pathlib import Path

from cbct_reasoner.data import normalize_report


def validate_report(report: str, *, maximum_characters: int = 20_000) -> str:
    """Validate output shape without modifying clinical meaning."""
    if not isinstance(report, str):
        raise TypeError("report must be a string")
    cleaned = normalize_report(report)
    if not cleaned:
        raise ValueError("report cannot be empty")
    if len(cleaned) > maximum_characters:
        raise ValueError(f"report exceeds {maximum_characters} characters")
    if "\x00" in report:
        raise ValueError("report contains a NUL character")
    return cleaned


def write_challenge_output(report: str, destination: str | Path) -> Path:
    """Write the official {report: string} output contract atomically."""
    output = Path(destination)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    payload = {"report": validate_report(report)}
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)
    return output


def read_challenge_output(source: str | Path) -> str:
    path = Path(source)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"report"}:
        raise ValueError(f"{path} must contain exactly one 'report' field")
    return validate_report(payload["report"])
