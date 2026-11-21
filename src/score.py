"""LLM-backed clip scoring in clip_id form."""

import logging

from pydantic import BaseModel, Field

from src.llm import llm


log = logging.getLogger(__name__)


class ScoreDimensions(BaseModel):
    hook: int = Field(ge=1, le=10)
    cliffhanger: int = Field(ge=1, le=10)
    emotion: int = Field(ge=1, le=10)


class ClipScore(BaseModel):
    clip_id: str
    hook_line: str
    hook_type: str
    cliffhanger_line: str
    why_it_works: str
    scores: ScoreDimensions
    problems: list[str]


class StandaloneTest(BaseModel):
    clip_id: str
    characters: str
    conflict: str
    what_happens_next: str
    understandability: int = Field(ge=1, le=10)


def _prompt(episode: dict, clip: dict) -> str:
    lines = []
    for event in clip["dialogue"]:
        speaker = event["speaker"] or "NARRATOR"
        tone = f" ({event['tone']})" if event.get("tone") else ""
        lines.append(f"{speaker}{tone}: {event['text']}")

    return """
You are judging a microdrama promo clip. JSON only.
Never output timestamps. Refer to the clip only by its clip_id.
Quote hook_line and cliffhanger_line only as exact lines from the clip.
Do not rewrite, paraphrase, or invent lines.

SERIES SUMMARY:
{logline}
CONTINUITY:
{continuity}

CLIP ID:
{clip_id}

CLIP LINES:
{lines}

RUBRIC:
- hook: Is the opening immediately understandable and attention-grabbing within 3 seconds? 1-10.
- cliffhanger: Does the ending leave an unresolved, urgent question or threat? 1-10.
- emotion: Is the emotional turn immediate, clear, and strong? 1-10.

Return exactly this nested JSON shape:
{{
  "clip_id": "{clip_id}",
  "hook_line": "exact line from CLIP LINES",
  "hook_type": "one short label",
  "cliffhanger_line": "exact line from CLIP LINES",
  "why_it_works": "short explanation",
  "scores": {{"hook": 8, "cliffhanger": 8, "emotion": 9}},
  "problems": ["one short issue"]
}}

hook_line and cliffhanger_line must be copied exactly. Scores must be integers from 1 through 10.
""".format(
        logline=episode.get("logline", ""),
        continuity=episode.get("continuity_notes", ""),
        clip_id=clip["clip_id"],
        lines="\n".join(lines),
    )


def _validate(raw: dict, clip: dict) -> ClipScore:
    score = ClipScore.model_validate(raw)
    if score.clip_id != clip["clip_id"]:
        raise ValueError("LLM returned a different clip_id")

    exact_lines = {event["text"] for event in clip["dialogue"]}
    if score.hook_line not in exact_lines:
        raise ValueError("hook_line is not an exact clip line")
    if score.cliffhanger_line not in exact_lines:
        raise ValueError("cliffhanger_line is not an exact clip line")
    return score


def standalone_test(clips: list[dict], provider: str | None = None,
                   model: str | None = None, api_key: str | None = None,
                   base_url: str | None = None) -> list[dict]:
    """Blind-test clips without series context; retry invalid JSON once."""
    results: list[dict] = []

    for clip in clips:
        lines = [
            f"{event['speaker'] or 'NARRATOR'}: {event['text']}"
            for event in clip["dialogue"]
        ]
        prompt = """You see a clip from a drama series. You receive no series summary.
JSON only. Never output timestamps. Answer only about the visible text.

CLIP ID: {clip_id}
CLIP TEXT:
{lines}

Return exactly this JSON shape:
{{"clip_id":"{clip_id}","characters":"...","conflict":"...","what_happens_next":"...","understandability":8}}
""".format(clip_id=clip["clip_id"], lines="\n".join(lines))

        for attempt in range(2):
            try:
                raw = llm(prompt, StandaloneTest.model_json_schema(), provider=provider,
                          model=model, api_key=api_key, base_url=base_url)
                test = StandaloneTest.model_validate(raw)
                if test.clip_id != clip["clip_id"]:
                    raise ValueError("standalone test returned a different clip_id")
                results.append(test.model_dump())
                break
            except Exception as exc:
                if attempt == 0:
                    log.warning("Retrying standalone test for %s: %s", clip["clip_id"], exc)
                else:
                    log.error("Skipping standalone test for %s: %s", clip["clip_id"], exc)

    return results


def score_clips(episode: dict, clips: list[dict], provider: str | None = None,
               model: str | None = None, api_key: str | None = None,
               base_url: str | None = None) -> list[dict]:
    """Score clip IDs once, retrying bad JSON or validation once."""
    results: list[dict] = []

    for clip in clips:
        prompt = _prompt(episode, clip)
        schema = ClipScore.model_json_schema()

        for attempt in range(2):
            try:
                raw = llm(prompt, schema, provider=provider,
                          model=model, api_key=api_key, base_url=base_url)
                score = _validate(raw, clip)
                results.append(score.model_dump())
                break
            except Exception as exc:
                if attempt == 0:
                    log.warning("Retrying %s after invalid response: %s", clip["clip_id"], exc)
                else:
                    log.error("Skipping %s after invalid response: %s", clip["clip_id"], exc)

    return results
