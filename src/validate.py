"""Validate LLM output and resolve clip IDs against code-owned data."""

import logging

log = logging.getLogger(__name__)


def validate_scores_detailed(scores: list[dict], clips: list[dict]) -> tuple[list[dict], list[dict]]:
    """Validate score records and return tuple of (accepted, rejected_with_reasons)."""
    clips_by_id = {clip["clip_id"]: clip for clip in clips}
    accepted: list[dict] = []
    rejected: list[dict] = []

    for score in scores:
        clip_id = score.get("clip_id")
        reasons: list[str] = []
        clip = clips_by_id.get(clip_id)

        if clip is None:
            reasons.append("unknown clip_id")
        else:
            duration = float(clip["duration_s"])
            if not 20 <= duration <= 60:
                reasons.append(f"duration {duration:.3f}s outside 20-60s")

            segments = clip.get("dialogue", [])
            if any(
                float(segments[i]["start_s"]) > float(segments[i + 1]["start_s"])
                for i in range(len(segments) - 1)
            ):
                reasons.append("dialogue segments are not in order")

            exact_lines = {event["text"] for event in segments}
            for field in ("hook_line", "cliffhanger_line"):
                quote = score.get(field)
                if quote not in exact_lines:
                    reasons.append(f"{field} quote not present in clip")

        nested = score.get("scores", {})
        for field in ("hook", "cliffhanger", "emotion"):
            value = nested.get(field)
            if not isinstance(value, int) or isinstance(value, bool) or not 1 <= value <= 10:
                reasons.append(f"{field} score outside 1-10")

        if reasons:
            for reason in reasons:
                log.warning("Rejected %s: %s", clip_id or "<missing clip_id>", reason)
            rejected.append({"clip_id": clip_id, "reasons": reasons, "raw_score": score})
        else:
            accepted.append(score)

    return accepted, rejected


def validate_scores(scores: list[dict], clips: list[dict]) -> list[dict]:
    """Validate score records and code-owned clip constraints."""
    accepted, _ = validate_scores_detailed(scores, clips)
    return accepted
