"""Pipeline orchestration with step-level callbacks, stop checks, and friendly diagnostics."""

import time
from typing import Callable

from src.parse import parse_screenplay_text
from src.candidates import generate_candidates, candidate_funnel
from src.prefilter import prefilter_candidates
from src.score import score_clips, standalone_test
from src.validate import validate_scores_detailed
from src.rank import add_weighted_scores, rank_clips
from src.report import archive_raw_llm_response


class PipelineError(Exception):
    """Exception with user-friendly diagnosis and actionable fix."""

    def __init__(self, message: str, suggested_fix: str):
        super().__init__(message)
        self.message = message
        self.suggested_fix = suggested_fix


def run_episode_pipeline(
    text: str,
    filename: str,
    mode: str,
    weights: dict,
    min_seconds: int = 20,
    max_seconds: int = 60,
    top_n: int = 12,
    max_promos: int = 5,
    provider: str | None = None,
    model: str | None = None,
    api_key: str | None = None,
    on_step: Callable[[str], None] | None = None,
    check_stop: Callable[[], bool] | None = None,
) -> dict:
    """Execute the end-to-end promo discovery pipeline for a single episode script."""

    def notify(step_name: str) -> None:
        if on_step is not None:
            on_step(step_name)

    def is_stopped() -> bool:
        return check_stop() if check_stop is not None else False

    start_time = time.time()
    stats = {"llm_calls": 0, "cache_hits": 0, "rejections": 0}

    # 1. Parsing
    notify("Parsing screenplay")
    if not text.strip():
        raise PipelineError(
            f"Screenplay file '{filename}' is empty.",
            "Upload a valid .txt screenplay containing episode headers and timestamped dialogue lines.",
        )

    episode = parse_screenplay_text(text, source_name=filename)
    if not episode.get("events"):
        raise PipelineError(
            f"No screenplay events could be parsed from '{filename}'.",
            "Ensure the script contains the header '--- SCRIPT START ---' followed by 'Start: (mm:ss.s) | End: (mm:ss.s)'.",
        )

    if is_stopped():
        return {"stopped": True, "filename": filename, "stats": stats}

    # 2. Cutting windows
    notify("Cutting candidate windows")
    all_candidates = generate_candidates(episode)
    candidates = [
        c for c in all_candidates
        if min_seconds <= float(c["duration_s"]) <= max_seconds
    ]
    funnel = candidate_funnel(episode)

    if not candidates:
        raise PipelineError(
            f"No candidate windows between {min_seconds}s and {max_seconds}s found in Episode {episode.get('episode_number', '')}.",
            "Check the Quality tab debug funnel to see why windows were rejected, or try widening the min/max duration sliders.",
        )

    if is_stopped():
        return {"stopped": True, "filename": filename, "stats": stats}

    # 3. Pre-filtering
    prefilter_pool_size = max(top_n, max_promos * 3)
    notify(f"Pre-filtering top candidates")
    top_candidates = prefilter_candidates(candidates, top_n=prefilter_pool_size)

    if is_stopped():
        return {"stopped": True, "filename": filename, "stats": stats}

    # 4. Scoring & Validation
    scored_pool: list[dict] = []
    rejected_list: list[dict] = []
    blind_map: dict[str, dict] = {}

    if mode == "Rules + LLM":
        if not api_key and (provider or "").lower() != "ollama":
            raise PipelineError(
                "API key missing for Rules + LLM mode.",
                "Enter your API key in the Model sidebar section, or switch the mode to 'Rules only'.",
            )

        clips_to_score = top_candidates[:top_n]
        for idx, clip in enumerate(clips_to_score, 1):
            if is_stopped():
                break

            clip_id = clip["clip_id"]
            notify(f"Scoring clip {idx} of {len(clips_to_score)} ({clip_id})")

            try:
                # LLM Score Call
                score_res = score_clips(
                    episode,
                    [clip],
                    provider=provider,
                    model=model,
                    api_key=api_key,
                )
                if score_res:
                    s_item = score_res[0]
                    if s_item.get("__cache_hit__"):
                        stats["cache_hits"] += 1
                    else:
                        stats["llm_calls"] += 1
                    archive_raw_llm_response("outputs/raw", "score", clip_id, s_item)

                    # LLM Standalone Blind Test
                    notify(f"Blind standalone test for {clip_id}")
                    blind_res = standalone_test(
                        [clip],
                        provider=provider,
                        model=model,
                        api_key=api_key,
                    )
                    if blind_res:
                        b_item = blind_res[0]
                        blind_map[clip_id] = b_item
                        if b_item.get("__cache_hit__"):
                            stats["cache_hits"] += 1
                        else:
                            stats["llm_calls"] += 1
                        archive_raw_llm_response("outputs/raw", "standalone", clip_id, b_item)

                    # Validation
                    notify(f"Validating {clip_id}")
                    accepted, rejected = validate_scores_detailed([s_item], [clip])
                    if accepted:
                        item = dict(clip)
                        item.update(accepted[0])
                        item["episode_number"] = episode.get("episode_number", "1")
                        item["standalone"] = blind_map.get(clip_id, {}).get("understandability", 8)
                        scored_pool.append(item)
                    if rejected:
                        stats["rejections"] += len(rejected)
                        rejected_list.extend(rejected)
            except Exception as exc:
                err_msg = str(exc).lower()
                if "429" in err_msg or "rate limit" in err_msg:
                    raise PipelineError(
                        f"Rate limit exceeded from provider for clip {clip_id}.",
                        "Wait a few moments and retry, or switch to 'Rules only' mode to complete the evaluation offline.",
                    ) from exc
                # For non-fatal clip errors, log and continue with fallback score
                score_val = min(10, max(1, int(round(clip.get("prefilter_score", 5) / 2))))
                dialogue = clip.get("dialogue", [])
                fallback_item = {
                    **clip,
                    "episode_number": episode.get("episode_number", "1"),
                    "scores": {"hook": score_val, "cliffhanger": score_val, "emotion": score_val},
                    "standalone": 8,
                    "hook_line": dialogue[0]["text"] if dialogue else "",
                    "cliffhanger_line": dialogue[-1]["text"] if dialogue else "",
                    "why_it_works": clip.get("reason", "Deterministic pre-filter signal"),
                    "low_confidence": True,
                }
                scored_pool.append(fallback_item)
    else:
        # Rules-only deterministic scoring
        notify("Applying rules-based scoring")
        for clip in top_candidates:
            dialogue = clip.get("dialogue", [])
            hook_line = dialogue[0]["text"] if dialogue else ""
            cliff_line = dialogue[-1]["text"] if dialogue else ""
            score_val = min(10, max(1, int(round(clip.get("prefilter_score", 5) / 2))))
            scored_pool.append({
                **clip,
                "episode_number": episode.get("episode_number", "1"),
                "scores": {
                    "hook": score_val,
                    "cliffhanger": score_val,
                    "emotion": score_val,
                },
                "standalone": 8,
                "hook_line": hook_line,
                "cliffhanger_line": cliff_line,
                "why_it_works": clip.get("reason", "Deterministic pre-filter signals"),
            })

    # 5. Ranking
    notify("Ranking and deduplicating")
    if not scored_pool:
        raise PipelineError(
            f"All candidate windows failed validation in Episode {episode.get('episode_number', '')}.",
            "Review the rejection reasons in the Quality tab, or adjust the duration and hook filters.",
        )

    enriched = add_weighted_scores(scored_pool, blind_map, weights=weights)
    ranked = rank_clips(enriched, max_per_episode=max_promos)

    for rank_idx, promo in enumerate(ranked, 1):
        promo["rank"] = rank_idx

    elapsed_s = round(time.time() - start_time, 2)
    stats["elapsed_s"] = elapsed_s

    ep_num = episode.get("episode_number", "00")
    import re
    m = re.search(r"\d+", str(ep_num))
    ep_id = f"ep{int(m.group()):02d}" if m else "ep00"

    return {
        "stopped": False,
        "episode_id": ep_id,
        "filename": filename,
        "episode": episode,
        "funnel": funnel,
        "candidate_count": len(candidates),
        "scored_candidates": enriched,
        "promos": ranked,
        "rejected_scores": rejected_list,
        "stats": stats,
    }
