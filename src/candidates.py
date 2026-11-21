"""Deterministic candidate clip generation."""

import re


def _episode_number(episode: dict) -> int:
    match = re.search(r"\d+", str(episode.get("episode_number", "")))
    return int(match.group()) if match else 1


def _hook_ok(episode: dict, start: float) -> tuple[bool, str]:
    """Soft hook flag: dialogue or action may begin or overlap the first 3 seconds."""
    hook_events = [
        event for event in episode["events"]
        if event["type"] in {"dialogue", "action"}
        and event["text"].strip()
        and float(event["start_s"]) < start + 3.0
        and float(event["end_s"]) > start
    ]
    if hook_events:
        return True, "dialogue/action activity starts or overlaps first 3 seconds"
    return False, "no dialogue/action overlaps the first 3 seconds"


def _cliffhanger_ok(end_event: dict) -> tuple[bool, str]:
    text = end_event["text"].strip()
    tone = end_event.get("tone", "").lower()
    strong = any(word in tone for word in ("desperate", "shock", "betrayed", "angered", "cruel", "gloating"))
    punctuation = text.endswith("?") or "..." in text or text.endswith("!")
    if strong or punctuation:
        return True, "strong ending tone or unresolved punctuation"
    return False, "ending has neutral tone and resolved punctuation"


def generate_candidates(episode: dict) -> list[dict]:
    """Generate dialogue-start/end windows and soft hook/cliffhanger metadata."""
    dialogue = [
        event for event in episode["events"]
        if event["type"] == "dialogue" and event["text"].strip()
    ]
    episode_number = _episode_number(episode)
    candidates: list[dict] = []

    for start_index, start_event in enumerate(dialogue):
        start = float(start_event["start_s"])
        for end_index in range(start_index + 1, len(dialogue)):
            end = float(dialogue[end_index]["end_s"])
            duration = end - start
            if duration < 20:
                continue
            if duration > 60:
                break

            hook_ok, hook_reason = _hook_ok(episode, start)
            cliff_ok, cliff_reason = _cliffhanger_ok(dialogue[end_index])
            candidates.append({
                "clip_id": f"ep{episode_number:02d}_w{len(candidates) + 1:03d}",
                "start_s": round(start, 3),
                "end_s": round(end, 3),
                "duration_s": round(duration, 3),
                "dialogue": dialogue[start_index:end_index + 1],
                "hook_ok": hook_ok,
                "hook_reason": hook_reason,
                "cliffhanger_ok": cliff_ok,
                "cliffhanger_reason": cliff_reason,
            })

    return candidates


def candidate_funnel(episode: dict) -> dict:
    """Return counts and rejection examples for debug output."""
    events = episode["events"]
    dialogue = [event for event in events if event["type"] == "dialogue" and event["text"].strip()]

    raw_windows: list[dict] = []
    length_rejected: list[dict] = []
    hook_rejected: list[dict] = []
    cliff_rejected: list[dict] = []

    for start_index, start_event in enumerate(dialogue):
        for end_index in range(start_index + 1, len(dialogue)):
            end_event = dialogue[end_index]
            start = float(start_event["start_s"])
            end = float(end_event["end_s"])
            duration = end - start

            if duration < 20:
                length_rejected.append({"start_s": start, "end_s": end, "reason": "duration under 20 seconds"})
                continue
            if duration > 60:
                length_rejected.append({"start_s": start, "end_s": end, "reason": "duration over 60 seconds"})
                break

            hook_ok, hook_reason = _hook_ok(episode, start)
            window = {
                "start_s": start,
                "end_s": end,
                "duration_s": duration,
                "hook_ok": hook_ok,
                "hook_reason": hook_reason,
            }
            if not hook_ok:
                hook_rejected.append(window)

            cliff_ok, cliff_reason = _cliffhanger_ok(end_event)
            window["cliffhanger_ok"] = cliff_ok
            window["cliffhanger_reason"] = cliff_reason
            if not cliff_ok:
                cliff_rejected.append(window)

            raw_windows.append(window)

    length_ok = len(raw_windows) + len(hook_rejected) + len(cliff_rejected)
    return {
        "events": len(events),
        "dialogue_events": len(dialogue),
        "raw_windows": len(raw_windows) + len(hook_rejected) + len(cliff_rejected) + len(length_rejected),
        "after_length": length_ok,
        "after_hook": length_ok - len(hook_rejected),
        "after_cliffhanger": length_ok - len(hook_rejected) - len(cliff_rejected),
        "length_rejected": length_rejected[:3],
        "hook_rejected": hook_rejected[:3],
        "cliff_rejected": cliff_rejected[:3],
    }
