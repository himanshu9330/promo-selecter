"""Tests for validation rules and rejection reporting."""

from src.validate import validate_scores, validate_scores_detailed


def test_validate_scores_detailed_success():
    clip = {
        "clip_id": "ep01_w001",
        "duration_s": 30.0,
        "dialogue": [
            {"speaker": "A", "text": "First line", "start_s": 0.0},
            {"speaker": "B", "text": "Last line", "start_s": 10.0},
        ],
    }
    score = {
        "clip_id": "ep01_w001",
        "hook_line": "First line",
        "cliffhanger_line": "Last line",
        "scores": {"hook": 8, "cliffhanger": 9, "emotion": 7},
    }
    accepted, rejected = validate_scores_detailed([score], [clip])
    assert len(accepted) == 1
    assert len(rejected) == 0


def test_validate_scores_detailed_rejections():
    clip = {
        "clip_id": "ep01_w001",
        "duration_s": 15.0,  # invalid duration (< 20s)
        "dialogue": [
            {"speaker": "A", "text": "Real line", "start_s": 0.0},
        ],
    }
    score = {
        "clip_id": "ep01_w001",
        "hook_line": "Hallucinated line",  # not in clip
        "cliffhanger_line": "Real line",
        "scores": {"hook": 12, "cliffhanger": 9, "emotion": 7},  # 12 > 10
    }
    accepted, rejected = validate_scores_detailed([score], [clip])
    assert len(accepted) == 0
    assert len(rejected) == 1
    reasons = " ".join(rejected[0]["reasons"])
    assert "outside 20-60s" in reasons
    assert "not present in clip" in reasons
    assert "outside 1-10" in reasons
