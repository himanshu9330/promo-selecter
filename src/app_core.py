"""Reusable pipeline orchestration and editor feedback helpers for the app."""

import csv
import json
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from src.parse import parse_episode
from src.candidates import generate_candidates, candidate_funnel
from src.prefilter import prefilter_candidates


ROOT = Path(__file__).resolve().parent.parent


def process_uploaded_bytes(filename: str, content: bytes, top_n: int = 12) -> dict:
    """Decode an uploaded script, cache-free parse, and build candidate records."""
    text = content.decode("utf-8", errors="replace")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", encoding="utf-8", delete=False) as handle:
        handle.write(text)
        temp_path = handle.name

    try:
        episode = parse_episode(temp_path)
        candidates = generate_candidates(episode)
        top = prefilter_candidates(candidates, top_n=top_n)
        funnel = candidate_funnel(episode)
        match = re.search(r"\d+", str(episode["episode_number"]))
        episode_id = f"ep{int(match.group()) :03d}" if match else "ep000"
        funnel["top_n"] = len(top)

        return {
            "episode_id": episode_id,
            "episode": episode,
            "filename": filename,
            "candidates": candidates,
            "top": top,
            "funnel": funnel,
        }
    finally:
        Path(temp_path).unlink(missing_ok=True)


def save_run(episode_id: str, filename: str, payload: dict) -> Path:
    """Persist a run under outputs/history/ by episode ID and filename."""
    history = ROOT / "outputs" / "history"
    history.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "-", filename)
    path = history / f"{episode_id}-{safe_name}.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def snap_to_line_boundary(value: float, clip: dict, side: str) -> float:
    """Snap a trim value to the nearest dialogue segment boundary."""
    boundaries = [float(event["start_s"] if side == "start" else event["end_s"])
                  for event in clip["dialogue"]]
    if not boundaries:
        return value
    return min(boundaries, key=lambda boundary: abs(boundary - value))


def save_editor_feedback(clip_id: str, decision: str, start_trim: float,
                         end_trim: float, model_name: str = "") -> dict:
    """Append one human decision to outputs/editor_feedback.csv."""
    destination = ROOT / "outputs" / "editor_feedback.csv"
    destination.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "clip_id": clip_id,
        "decision": decision,
        "start_trim_seconds": start_trim,
        "end_trim_seconds": end_trim,
        "model": model_name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    needs_header = not destination.exists() or destination.stat().st_size == 0
    with destination.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if needs_header:
            writer.writeheader()
        writer.writerow(row)
    return row


def read_editor_feedback() -> list[dict]:
    destination = ROOT / "outputs" / "editor_feedback.csv"
    if not destination.exists() or destination.stat().st_size == 0:
        return []
    with destination.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames and "clip_id" in reader.fieldnames:
            return [r for r in reader if r.get("clip_id")]
        # Fallback for headerless csv
        handle.seek(0)
        raw_reader = csv.reader(handle)
        fieldnames = ["clip_id", "decision", "start_trim_seconds", "end_trim_seconds", "model", "timestamp"]
        rows = []
        for line in raw_reader:
            if line and len(line) >= 2:
                if line[0] == "clip_id":
                    continue
                rows.append(dict(zip(fieldnames, line)))
        return rows
