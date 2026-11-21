"""Screenplay parsing and timestamp normalization."""

import logging
import re
from pathlib import Path


log = logging.getLogger(__name__)

_TIMESTAMP = re.compile(r"(?<!\d)(\d{1,2}:)?([0-5]?\d):(\d{1,2}(?:\.\d+)?)")
_TIME_LINE = re.compile(
    r"^\s*Start:\s*\(?((?:\d{1,2}:)?\d{1,2}:\d{1,2}(?:\.\d+)?)\)?\s*"
    r"\|\s*End:\s*\(?((?:\d{1,2}:)?\d{1,2}:\d{1,2}(?:\.\d+)?)\)?",
    re.IGNORECASE,
)
_HEADER = re.compile(r"^\s*EPISODE (?:NUMBER|NO\.?)\s*:\s*(.*)$", re.IGNORECASE)
_LOGLINE = re.compile(r"^\s*EPISODE LOGLINE\s*:\s*(.*)$", re.IGNORECASE)
_CONTINUITY = re.compile(r"^\s*CONTINUITY NOTES\s*:\s*(.*)$", re.IGNORECASE)
_SCENE = re.compile(r"^\s*SCENE\s+(.+?)\s*$", re.IGNORECASE)
_SCENE_FIELD = re.compile(r"^\s*Scene (start|end|heading)\s*:\s*(.*)$", re.IGNORECASE)
_BLOCK = re.compile(r"^\s*(?:\*\*|\u002a\u002a)?(CAMERA|ACTION|[^:*]+?)\b(?:\*\*|\u002a\u002a)?\s*$", re.IGNORECASE)


def parse_timestamp(value: str) -> float:
    """Convert m:ss.s or h:mm:ss.s to seconds."""
    value = value.strip().strip("()")
    parts = value.split(":")
    try:
        if len(parts) == 2:
            minutes, seconds = parts
            total = int(minutes) * 60 + float(seconds)
        elif len(parts) == 3:
            hours, minutes, seconds = parts
            total = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
        else:
            raise ValueError
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid timestamp: {value!r}") from exc
    return round(total, 3)


def format_timestamp(seconds: float) -> str:
    """Convert seconds to mm:ss.s or hh:mm:ss.s."""
    total_sec = max(0.0, float(seconds))
    hours = int(total_sec // 3600)
    minutes = int((total_sec % 3600) // 60)
    rem_sec = total_sec % 60
    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{rem_sec:04.1f}"
    return f"{minutes:02d}:{rem_sec:04.1f}"



def _clean_inline_label(line: str) -> str:
    return line.strip().strip("*").strip()


def _parse_block_label(line: str) -> tuple[str, str, str] | None:
    """Return event type, speaker, tone for an explicit block label."""
    stripped = line.strip()
    if not (stripped.startswith("**") and stripped.endswith("**")):
        return None

    label = _clean_inline_label(stripped).rstrip(":").strip()
    if not label:
        return None

    upper = label.upper()
    if upper == "CAMERA":
        return "camera", "", ""
    if upper == "ACTION":
        return "action", "", ""

    os_match = re.match(r"^(.+?)\s*\(O\.S\.\)\s*(?:\(([^()]*)\))?$", label, re.IGNORECASE)
    if os_match:
        return "dialogue", f"{os_match.group(1).strip()} (O.S.)", (os_match.group(2) or "").strip()

    tone_match = re.match(r"^(.+?)\s*\(([^()]*)\)$", label)
    if tone_match:
        return "dialogue", tone_match.group(1).strip(), tone_match.group(2).strip()

    return "dialogue", label, ""


def parse_screenplay_text(text: str, source_name: str = "") -> dict:
    """Parse episode metadata and scene events directly from text content."""
    lines = text.splitlines()
    episode_number = ""
    logline = ""
    continuity = ""
    script_started = False
    scene_start: float | None = None
    scene_end: float | None = None
    events: list[dict] = []
    problems: list[str] = []

    current_label: tuple[str, str, str] | None = None
    current_time: tuple[float, float] | None = None
    text_lines: list[str] = []

    def finish_block() -> None:
        nonlocal current_label, current_time, text_lines
        if current_label is None or current_time is None:
            text_lines = []
            return

        raw_text = "\n".join(text_lines).strip()
        event_type, speaker, tone = current_label
        local_start, local_end = current_time

        if scene_start is not None:
            scene_duration = scene_end - scene_start if scene_end is not None else None
            block_fits_local = (
                local_start >= 0
                and (scene_duration is None or local_end <= scene_duration)
            )
            block_fits_episode = (
                local_start >= scene_start
                and (scene_end is None or local_end <= scene_end)
            )

            if block_fits_episode:
                start_s = local_start
                end_s = local_end
            elif block_fits_local:
                start_s = scene_start + local_start
                end_s = scene_start + local_end
            else:
                start_s = scene_start + local_start
                end_s = scene_start + local_end
        else:
            start_s = local_start
            end_s = local_end

        if event_type == "dialogue":
            if len(raw_text) >= 2 and raw_text[0] == raw_text[-1] and raw_text[0] in {'"', "“", "‘"}:
                raw_text = raw_text[1:-1].strip()

        if not raw_text:
            problems.append(f"Empty {event_type} text at {local_start:.3f}-{local_end:.3f}s")
            log.warning("Empty %s text at %.3fs-%.3fs", event_type, local_start, local_end)
        if end_s <= start_s:
            problems.append(f"Invalid timestamp range {start_s:.3f}-{end_s:.3f}s")
            log.warning("Invalid timestamp range %.3fs-%.3fs", start_s, end_s)

        events.append({
            "type": event_type,
            "speaker": speaker,
            "tone": tone,
            "text": raw_text,
            "start_s": round(start_s, 3),
            "end_s": round(end_s, 3),
        })
        current_label = None
        current_time = None
        text_lines = []

    for line in lines:
        header_match = _HEADER.match(line)
        if header_match:
            episode_number = header_match.group(1).strip()
            continue
        logline_match = _LOGLINE.match(line)
        if logline_match:
            logline = logline_match.group(1).strip()
            continue
        continuity_match = _CONTINUITY.match(line)
        if continuity_match:
            continuity = continuity_match.group(1).strip()
            continue

        if re.match(r"^\s*---\s*SCRIPT START\s*---\s*$", line, re.IGNORECASE):
            script_started = True
            continue
        if not script_started:
            continue

        scene_match = _SCENE.match(line)
        if scene_match:
            finish_block()
            scene_start = None
            scene_end = None
            continue

        field_match = _SCENE_FIELD.match(line)
        if field_match:
            field = field_match.group(1).lower()
            value = field_match.group(2).strip()
            try:
                if field == "start":
                    scene_start = parse_timestamp(value)
                elif field == "end":
                    scene_end = parse_timestamp(value)
            except ValueError as exc:
                problems.append(str(exc))
                log.warning("%s", exc)
            continue

        time_match = _TIME_LINE.match(line)
        if time_match:
            finish_block()
            try:
                current_time = (
                    parse_timestamp(time_match.group(1)),
                    parse_timestamp(time_match.group(2)),
                )
            except ValueError as exc:
                problems.append(str(exc))
                log.warning("%s", exc)
                current_time = None
            continue

        block_label = _parse_block_label(line)
        if block_label is not None:
            finish_block()
            current_label = block_label
            continue

        if current_label is not None:
            text_lines.append(line.strip())

    finish_block()

    if not episode_number:
        problems.append("Missing episode number")
        log.warning("Missing episode number")
    if not script_started:
        problems.append("Missing --- SCRIPT START --- marker")
        log.warning("Missing --- SCRIPT START --- marker")

    if problems:
        log.warning("Parsed %d problem(s) from %s", len(problems), source_name or "<in-memory>")

    return {
        "episode_number": episode_number,
        "logline": logline,
        "continuity_notes": continuity,
        "events": events,
        "problems": problems,
    }


def parse_episode(path: str) -> dict:
    """Parse episode metadata and scene events from a screenplay file."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)

    text = source.read_text(encoding="utf-8-sig")
    return parse_screenplay_text(text, source_name=str(source))


def inspect_screenplay(text: str, filename: str = "") -> dict:
    """Inspect and summarize screenplay structure and warnings."""
    parsed = parse_screenplay_text(text, source_name=filename)
    dialogue_events = [e for e in parsed.get("events", []) if e.get("type") == "dialogue"]
    return {
        "filename": filename,
        "episode_number": parsed.get("episode_number", ""),
        "logline": parsed.get("logline", ""),
        "event_count": len(parsed.get("events", [])),
        "dialogue_count": len(dialogue_events),
        "problems": parsed.get("problems", []),
        "is_valid": bool(parsed.get("episode_number") and len(parsed.get("events", [])) > 0),
        "parsed": parsed,
    }

