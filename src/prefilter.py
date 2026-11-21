"""Fast deterministic prefiltering for clips."""

import re


STRONG_TONES = (
    "shocked", "shock", "angry", "outraged", "betrayed", "desperate",
    "gloating", "smug", "mocking", "cruel", "heartbroken", "devastated",
    "horrified", "suspicious", "furious",
)
SHOCK_WORDS = (
    "betray", "betrayed", "marry", "marrying", "lied", "lie", "wait",
    "never", "impossible", "pregnant", "father", "sister", "swore",
    "changed", "mind", "love", "jewel", "daughter", "mistake",
)


def _overlap_ratio(a: dict, b: dict) -> float:
    start = max(float(a["start_s"]), float(b["start_s"]))
    end = min(float(a["end_s"]), float(b["end_s"]))
    overlap = max(0.0, end - start)
    union = float(a["duration_s"]) + float(b["duration_s"]) - overlap
    return overlap / union if union > 0 else 0.0


def prefilter_candidates(
    candidates: list[dict],
    top_n: int = 15,
    max_overlap: float = 0.45,
) -> list[dict]:
    """Score windows with exact text and tone signals, retaining top N diverse candidate scenes."""
    scored: list[dict] = []

    for candidate in candidates:
        dialogue = candidate["dialogue"]
        all_text = " ".join(event["text"] for event in dialogue)
        lower_text = all_text.lower()

        tone_hits = []
        for event in dialogue:
            tone = event.get("tone", "").lower()
            for signal in STRONG_TONES:
                if signal in tone:
                    tone_hits.append(signal)

        end_text = dialogue[-1]["text"].strip()
        punctuation_score = 0
        reason_parts = []
        if end_text.endswith("?"):
            punctuation_score += 3
            reason_parts.append("ends in a question")
        if "..." in end_text or end_text.endswith("..."):
            punctuation_score += 3
            reason_parts.append("open-ended ellipsis")

        shock_hits = sorted({word for word in SHOCK_WORDS if re.search(rf"\b{re.escape(word)}\b", lower_text)})
        shock_score = min(len(shock_hits), 6)

        if tone_hits:
            tone_unique = sorted(set(tone_hits))
            reason_parts.append("strong tones: " + ", ".join(tone_unique[:4]))
        if shock_hits:
            reason_parts.append("shock words: " + ", ".join(shock_hits[:5]))

        word_count = len(re.findall(r"\b\w+\b", all_text))
        pace = word_count / candidate["duration_s"]
        pace_score = 2 if pace >= 2.2 else 1 if pace >= 1.5 else 0
        if pace_score:
            reason_parts.append(f"fast line pace ({pace:.2f} words/s)")

        tone_score = min(len(set(tone_hits)), 6)
        score = (tone_score * 2) + (punctuation_score * 2) + (shock_score * 2) + (pace_score * 1)

        result = dict(candidate)
        result["prefilter_score"] = round(score, 3)
        result["hook_ok"] = bool(candidate.get("hook_ok", True))
        result["cliffhanger_ok"] = bool(candidate.get("cliffhanger_ok", True))
        result["low_confidence"] = not (result["hook_ok"] and result["cliffhanger_ok"])
        result["reason"] = "; ".join(reason_parts) if reason_parts else "dialogue-heavy window"
        scored.append(result)

    scored.sort(key=lambda item: (
        -item["prefilter_score"],
        item["duration_s"],
        item["start_s"],
    ))
    strong = [item for item in scored if item["hook_ok"] and item["cliffhanger_ok"]]
    pool = strong if len(strong) >= 5 else scored

    # Diverse window selection: avoid packing the top N with identical 1-second shifts
    selected: list[dict] = []
    for item in pool:
        overlap = any(
            _overlap_ratio(item, other) > max_overlap
            for other in selected
        )
        if not overlap:
            selected.append(item)
        if len(selected) >= top_n:
            break

    # If diverse selection hasn't filled top_n, backfill with remaining highest scores
    if len(selected) < top_n:
        for item in pool:
            if item not in selected:
                selected.append(item)
            if len(selected) >= top_n:
                break

    return selected
