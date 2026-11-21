"""Tests for report generation and export serialization."""

from src.parse import format_timestamp
from src.report import build_export_zip, write_all_outputs


def test_format_timestamp():
    assert format_timestamp(0) == "00:00.0"
    assert format_timestamp(11.3) == "00:11.3"
    assert format_timestamp(65.5) == "01:05.5"
    assert format_timestamp(3665.2) == "01:01:05.2"


def test_write_all_outputs_and_zip(tmp_path):
    promos = [
        {
            "rank": 1,
            "episode_number": "1",
            "clip_id": "ep01_w001",
            "start_s": 10.0,
            "end_s": 35.0,
            "duration_s": 25.0,
            "weighted_score": 0.88,
            "hook_line": "Look behind you.",
            "cliffhanger_line": "Who are you?",
            "why_it_works": "Urgent tone.",
            "scores": {"hook": 9, "cliffhanger": 9, "emotion": 8},
            "standalone": 9,
        }
    ]
    paths = write_all_outputs(promos, base_dir=str(tmp_path))
    assert "promos.json" in paths["json"]
    assert "promos.csv" in paths["csv"]
    assert "report.md" in paths["markdown"]

    zip_bytes = build_export_zip(raw_dir=str(tmp_path / "raw"))
    assert len(zip_bytes) > 0
