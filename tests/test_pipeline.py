"""Tests for pipeline execution and error diagnostics."""

import pytest
from src.pipeline import run_episode_pipeline, PipelineError


SAMPLE_SCRIPT = """EPISODE NUMBER: 1
EPISODE LOGLINE: A family betrayal.
--- SCRIPT START ---
SCENE 1
Scene start: 00:00.0
Scene end: 00:50.0
**SARAH:**
Start: (00:05.0) | End: (00:10.0)
"Why did you conceal this from me?"
**MARK (Angry):**
Start: (00:15.0) | End: (00:20.0)
"Because you would never understand!"
**SARAH (Desperate):**
Start: (00:30.0) | End: (00:35.0)
"Is the contract already signed?"
"""


def test_run_episode_pipeline_rules_only(monkeypatch):
    monkeypatch.setattr("src.rank.get_embedding_model", lambda: None)
    res = run_episode_pipeline(
        text=SAMPLE_SCRIPT,
        filename="sample.txt",
        mode="Rules only",
        weights={"hook": 0.3, "cliffhanger": 0.3, "standalone": 0.25, "emotion": 0.15},
        min_seconds=20,
        max_seconds=60,
        top_n=5,
        max_promos=2,
    )
    assert res["stopped"] is False
    assert res["episode_id"] == "ep01"
    assert len(res["promos"]) > 0
    assert res["promos"][0]["rank"] == 1
    assert "start_s" in res["promos"][0]
    assert "end_s" in res["promos"][0]


def test_run_episode_pipeline_empty_error():
    with pytest.raises(PipelineError) as exc_info:
        run_episode_pipeline(
            text="",
            filename="empty.txt",
            mode="Rules only",
            weights={},
        )
    assert "is empty" in str(exc_info.value.message)
    assert "Upload a valid" in exc_info.value.suggested_fix


def test_run_episode_pipeline_missing_key_error():
    with pytest.raises(PipelineError) as exc_info:
        run_episode_pipeline(
            text=SAMPLE_SCRIPT,
            filename="sample.txt",
            mode="Rules + LLM",
            weights={},
            provider="openrouter",
            api_key="",
        )
    assert "API key missing" in str(exc_info.value.message)
    assert "Enter your API key" in exc_info.value.suggested_fix


def test_run_episode_pipeline_stop_flag():
    res = run_episode_pipeline(
        text=SAMPLE_SCRIPT,
        filename="sample.txt",
        mode="Rules only",
        weights={},
        check_stop=lambda: True,  # simulate immediate stop
    )
    assert res["stopped"] is True
