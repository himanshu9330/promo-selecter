"""Main results section for Promo Finder.

Renders Top Pick Hero card, 4-metric summary row, remaining compact cards,
and analysis tabs using 100% native Streamlit components.
"""

import streamlit as st

from ui.components import top_pick_card, clip_card
from ui.tabs import (
    render_all_candidates_tab,
    render_timeline_tab,
    render_quality_tab,
    render_export_tab,
)


def render_results(runs: list[dict], model_name: str = "") -> None:
    """Render the primary results area according to the design specification."""
    if not runs:
        return

    # Aggregate all promos across runs
    all_promos = [p for r in runs for p in r.get("promos", [])]
    if not all_promos:
        st.warning("No promo clips matched the filter rules for the evaluated screenplay(s).")
        return

    # Sort initially by rank and weighted score
    sorted_all = sorted(all_promos, key=lambda p: (p.get("rank", 999), -float(p.get("weighted_score", 0))))

    # Deduplicate strictly by (episode_number, clip_id) so each card is a distinct promo
    seen_keys: set[tuple[str, str]] = set()
    sorted_promos: list[dict] = []
    for p in sorted_all:
        cid = p.get("clip_id", "")
        ep = str(p.get("episode_number", "1"))
        key = (ep, cid) if cid else (ep, f"s{p.get('start_s')}_e{p.get('end_s')}")
        if key in seen_keys:
            continue
        seen_keys.add(key)
        sorted_promos.append(p)

    # Re-index ranks 1..N
    for idx, p in enumerate(sorted_promos, 1):
        p["rank"] = idx

    # Calm success banner (No balloons, no snow)
    st.success(
        f"Promo discovery complete: {len(all_promos)} high-retention cuts identified across {len(runs)} episode(s).",
        icon=":material/check_circle:",
    )

    # Low-confidence Rules-only persistent banner
    if any(r.get("mode") == "Rules only" for r in runs):
        st.info(
            "**:violet[Rules-Only Mode Active:]** Promo scores and clip bounds are derived deterministically from screenplay dialogue density, punctuation, and tone keywords without LLM semantic evaluation. Switch to **Rules + LLM** for deep semantic hook validation.",
            icon=":material/info:",
        )

    # Partial results warning if run was interrupted or stopped
    if any(r.get("stopped") for r in runs):
        st.warning(
            "**Partial Results Displayed:** The run was ended early by user request. Partial evaluated clips are preserved below.",
            icon=":material/pause_circle:",
        )

    # ---------------------------------------------------------
    # a. Top Pick Hero Card (Rank 1 only)
    # ---------------------------------------------------------
    top_promo = sorted_promos[0]
    top_pick_card(top_promo, key_prefix="hero_pick", model_name=model_name)

    # ---------------------------------------------------------
    # b. Metrics Row (4 summary metrics)
    # ---------------------------------------------------------
    promos_count = len(sorted_promos)
    episodes_count = len({str(p.get("episode_number", "1")) for p in sorted_promos})
    avg_score = sum(float(p.get("weighted_score", 0.0)) for p in sorted_promos) / promos_count
    low_conf_count = sum(1 for p in sorted_promos if p.get("low_confidence"))
    low_conf_share = (low_conf_count / promos_count) * 100.0

    m1, m2, m3, m4 = st.columns(4, gap="small")
    m1.metric("Promos Found", promos_count)
    m2.metric("Episodes Covered", episodes_count)
    m3.metric("Average Score", f"{avg_score:.2f}")
    m4.metric("Low Confidence Share", f"{low_conf_share:.0f}%")

    st.divider()

    # ---------------------------------------------------------
    # c. Remaining Ranked Promos (Rank 2 and up)
    # ---------------------------------------------------------
    if len(sorted_promos) > 1:
        st.subheader("Remaining Ranked Promos")
        for rank_num, promo in enumerate(sorted_promos[1:], 2):
            clip_id_slug = promo.get("clip_id", f"idx_{rank_num}")
            clip_card(promo, rank=rank_num, key_prefix=f"list_{rank_num}_{clip_id_slug}", model_name=model_name)

        st.divider()

    # ---------------------------------------------------------
    # d. Detailed Analysis Tabs
    # ---------------------------------------------------------
    tab_all, tab_timeline, tab_quality, tab_export = st.tabs([
        "All Candidates",
        "Timeline",
        "Quality",
        "Export",
    ])

    with tab_all:
        render_all_candidates_tab(runs)

    with tab_timeline:
        render_timeline_tab(runs)

    with tab_quality:
        render_quality_tab(runs)

    with tab_export:
        render_export_tab(runs)
