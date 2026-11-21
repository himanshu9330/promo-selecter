"""Deterministic weighted ranking and duplicate removal."""

import os
from pathlib import Path
import yaml

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


def _overlap_ratio(a: dict, b: dict) -> float:
    start = max(float(a["start_s"]), float(b["start_s"]))
    end = min(float(a["end_s"]), float(b["end_s"]))
    overlap = max(0.0, end - start)
    union = float(a["duration_s"]) + float(b["duration_s"]) - overlap
    return overlap / union if union > 0 else 0.0


def _clip_text(clip: dict) -> str:
    return " ".join(event["text"] for event in clip["dialogue"])


_EMBED_MODEL = None


def get_embedding_model():
    """Module-level singleton for sentence transformer model."""
    global _EMBED_MODEL
    if _EMBED_MODEL is not None:
        return _EMBED_MODEL
    try:
        from sentence_transformers import SentenceTransformer
        try:
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2", local_files_only=True)
            return _EMBED_MODEL
        except Exception:
            _EMBED_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
            return _EMBED_MODEL
    except Exception:
        return None



def rank_clips(
    clips: list[dict],
    max_per_episode: int = 5,
    duplicate_threshold: float = 0.82,
    max_overlap: float = 0.45,
) -> list[dict]:
    """Apply weights, overlap filtering, semantic deduplication, and episode limit."""
    ranked = sorted(clips, key=lambda clip: clip["weighted_score"], reverse=True)

    embeddings = None
    np = None
    if len(ranked) > 1:
        try:
            import numpy as numpy
            np = numpy
            model = get_embedding_model()
            if model is not None:
                embeddings = model.encode(
                    [_clip_text(clip) for clip in ranked],
                    normalize_embeddings=True,
                )
        except Exception:
            embeddings = None


    accepted: list[dict] = []
    accepted_indexes: list[int] = []
    episode_counts: dict[str, int] = {}

    for ranked_index, clip in enumerate(ranked):
        episode = str(clip["episode_number"])
        if episode_counts.get(episode, 0) >= max_per_episode:
            continue

        overlap = any(
            _overlap_ratio(clip, other) > max_overlap
            for other in accepted
            if str(other["episode_number"]) == episode
        )
        if overlap:
            continue

        duplicate = False
        if embeddings is not None:
            for accepted_index in accepted_indexes:
                if str(ranked[accepted_index]["episode_number"]) != episode:
                    continue
                similarity = float(np.dot(embeddings[ranked_index], embeddings[accepted_index]))
                if similarity >= duplicate_threshold:
                    duplicate = True
                    break
        if duplicate:
            continue

        episode_counts[episode] = episode_counts.get(episode, 0) + 1
        accepted_indexes.append(ranked_index)
        accepted.append(dict(clip))

    return accepted


def normalize_weights(hook: float, cliffhanger: float, standalone: float, emotion: float) -> dict[str, float]:
    """Normalize 4 ranking weights so they sum to 1.0."""
    total = float(hook) + float(cliffhanger) + float(standalone) + float(emotion)
    if total <= 0:
        return {"hook": 0.25, "cliffhanger": 0.25, "standalone": 0.25, "emotion": 0.25}
    return {
        "hook": round(hook / total, 4),
        "cliffhanger": round(cliffhanger / total, 4),
        "standalone": round(standalone / total, 4),
        "emotion": round(emotion / total, 4),
    }


def add_weighted_scores(clips: list[dict], standalone: dict[str, dict],
                       weights: dict[str, float] | None = None) -> list[dict]:
    """Add weighted scores using custom weights or config weights, and standalone understandability."""
    if weights is None:
        with CONFIG_PATH.open("r", encoding="utf-8") as handle:
            weights_data = yaml.safe_load(handle)["ranking_weights"]
    else:
        weights_data = weights

    norm = normalize_weights(
        float(weights_data.get("hook", 0.30)),
        float(weights_data.get("cliffhanger", 0.30)),
        float(weights_data.get("standalone", 0.25)),
        float(weights_data.get("emotion", 0.15)),
    )

    enriched = []
    for clip in clips:
        blind = standalone.get(clip["clip_id"])
        standalone_score = (blind["understandability"] / 10) if blind else 0
        weighted = (
            (clip["scores"]["hook"] / 10) * float(norm["hook"])
            + (clip["scores"]["cliffhanger"] / 10) * float(norm["cliffhanger"])
            + standalone_score * float(norm["standalone"])
            + (clip["scores"]["emotion"] / 10) * float(norm["emotion"])
        )
        item = dict(clip)
        item["weighted_score"] = round(weighted, 4)
        enriched.append(item)
    return enriched


def rerank_existing_run(run: dict, new_weights: dict[str, float], max_promos: int = 5) -> dict:
    """Recompute weighted scores and ranking for an existing run with new weights, zero LLM calls."""
    pool = run.get("scored_candidates") or run.get("promos", [])
    if not pool:
        return run

    standalone_map = {
        clip["clip_id"]: {"understandability": clip.get("standalone", 8)}
        for clip in pool
    }
    enriched = add_weighted_scores(pool, standalone_map, weights=new_weights)
    new_ranked = rank_clips(enriched, max_per_episode=max_promos)

    updated_run = dict(run)
    for rank, promo in enumerate(new_ranked, 1):
        promo["rank"] = rank
    updated_run["promos"] = new_ranked
    return updated_run

