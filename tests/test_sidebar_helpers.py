"""Tests for sidebar helpers and pipeline logic."""

from src.parse import inspect_screenplay, parse_screenplay_text
from src.rank import normalize_weights, add_weighted_scores, rerank_existing_run


def test_rerank_existing_run(monkeypatch):
    monkeypatch.setattr("src.rank.get_embedding_model", lambda: None)
    run = {

        "episode_id": "ep01",
        "scored_candidates": [
            {
                "clip_id": "ep01_w001",
                "episode_number": "1",
                "start_s": 10.0,
                "end_s": 40.0,
                "duration_s": 30.0,
                "dialogue": [{"speaker": "A", "text": "hello"}],
                "scores": {"hook": 10, "cliffhanger": 2, "emotion": 5},
                "standalone": 8,
            },
            {
                "clip_id": "ep01_w002",
                "episode_number": "1",
                "start_s": 50.0,
                "end_s": 80.0,
                "duration_s": 30.0,
                "dialogue": [{"speaker": "B", "text": "bye"}],
                "scores": {"hook": 2, "cliffhanger": 10, "emotion": 5},
                "standalone": 8,
            },
        ],
    }

    # High hook weight prefers w001
    run_hook = rerank_existing_run(run, {"hook": 1.0, "cliffhanger": 0.0, "standalone": 0.0, "emotion": 0.0})
    assert run_hook["promos"][0]["clip_id"] == "ep01_w001"

    # High cliffhanger weight prefers w002
    run_cliff = rerank_existing_run(run, {"hook": 0.0, "cliffhanger": 1.0, "standalone": 0.0, "emotion": 0.0})
    assert run_cliff["promos"][0]["clip_id"] == "ep01_w002"

from src.llm import check_llm_connection


def test_check_llm_connection():
    # Calling with empty key or invalid model should return (False, error) without throwing
    success, msg = check_llm_connection(provider="invalid_provider", model="invalid_model")
    assert success is False
    assert "Unsupported LLM provider" in msg



def test_normalize_weights():
    # Standard weights sum to 1.0
    w = normalize_weights(0.3, 0.3, 0.25, 0.15)
    assert round(sum(w.values()), 2) == 1.0
    assert w["hook"] == 0.3
    assert w["cliffhanger"] == 0.3

    # Scaled weights normalize proportionally
    w2 = normalize_weights(2.0, 2.0, 0.0, 0.0)
    assert w2["hook"] == 0.5
    assert w2["cliffhanger"] == 0.5
    assert w2["standalone"] == 0.0
    assert w2["emotion"] == 0.0

    # Zero weights fallback to equal
    w0 = normalize_weights(0, 0, 0, 0)
    assert w0["hook"] == 0.25


def test_inspect_screenplay():
    script = """EPISODE NUMBER: 42
EPISODE LOGLINE: Secrets unravel.
--- SCRIPT START ---
SCENE 1
Scene start: 00:00.0
Scene end: 00:30.0
**ALICE:**
Start: (00:01.0) | End: (00:05.0)
"Wait, what did you do?"
"""
    res = inspect_screenplay(script, "test.txt")
    assert res["is_valid"] is True
    assert res["episode_number"] == "42"
    assert res["event_count"] == 1
    assert res["dialogue_count"] == 1
    assert len(res["problems"]) == 0


def test_re_ranking_with_custom_weights():
    clip = {
        "clip_id": "ep01_w001",
        "scores": {"hook": 10, "cliffhanger": 2, "emotion": 5},
    }
    standalone = {"ep01_w001": {"understandability": 8}}

    # Heavy hook weight
    res_hook = add_weighted_scores([clip], standalone, weights={"hook": 1.0, "cliffhanger": 0.0, "standalone": 0.0, "emotion": 0.0})
    assert res_hook[0]["weighted_score"] == 1.0

    # Heavy cliffhanger weight
    res_cliff = add_weighted_scores([clip], standalone, weights={"hook": 0.0, "cliffhanger": 1.0, "standalone": 0.0, "emotion": 0.0})
    assert res_cliff[0]["weighted_score"] == 0.2
