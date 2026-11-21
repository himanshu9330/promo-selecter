"""Native Streamlit tab views for All Candidates, Timeline, Quality, and Export.

Strictly native: no unsafe_allow_html, no altair, no plotly, no matplotlib.
Uses native st.dataframe, st.bar_chart, st.area_chart, and st.download_button.
"""

import json
from pathlib import Path
import pandas as pd
import streamlit as st

from src.parse import format_timestamp
from src.report import build_export_zip, write_all_outputs
from ui.components import script_table


def render_all_candidates_tab(runs: list[dict]) -> None:
    """Render Candidate Explorer with native st.dataframe and column configs."""
    if not runs:
        st.info("Run an episode to populate candidates.")
        return

    # Gather all candidate clips (from top, candidates, or promos)
    records = []
    for r in runs:
        ep_id = r.get("episode_id", "ep01")
        pool = r.get("scored_candidates") or r.get("top") or r.get("promos", [])
        for c in pool:
            score_val = float(c.get("weighted_score", c.get("prefilter_score", 0.0) / 10.0))
            if score_val > 1.0:
                score_val = score_val / 10.0

            dialogue = c.get("dialogue", [])
            hook_txt = dialogue[0]["text"] if dialogue else ""
            cliff_txt = dialogue[-1]["text"] if dialogue else ""

            records.append({
                "Episode": ep_id,
                "Clip ID": c.get("clip_id", ""),
                "Start (s)": round(float(c.get("start_s", 0.0)), 1),
                "End (s)": round(float(c.get("end_s", 0.0)), 1),
                "Duration (s)": round(float(c.get("duration_s", 0.0)), 1),
                "Score": round(score_val, 3),
                "Confidence": "Low" if c.get("low_confidence") else "High",
                "Hook Text": hook_txt[:50] + "..." if len(hook_txt) > 50 else hook_txt,
                "Cliffhanger Text": cliff_txt[:50] + "..." if len(cliff_txt) > 50 else cliff_txt,
                "_raw_clip": c,
            })

    if not records:
        st.caption("No candidate windows found.")
        return

    # Filters
    f1, f2, f3 = st.columns([2, 3, 3], gap="small")
    with f1:
        ep_list = ["All"] + sorted(list({r["Episode"] for r in records}))
        selected_ep = st.selectbox("Episode", ep_list)
    with f2:
        min_score = st.slider("Minimum score", 0.0, 1.0, 0.0, step=0.05)
    with f3:
        conf_filter = st.pills("Confidence", ["All", "High", "Low"], default="All")

    filtered = [
        r for r in records
        if (selected_ep == "All" or r["Episode"] == selected_ep)
        and r["Score"] >= min_score
        and (conf_filter == "All" or r["Confidence"] == conf_filter)
    ]

    st.caption(f"Showing {len(filtered)} of {len(records)} candidate windows.")

    display_df = pd.DataFrame([
        {k: v for k, v in r.items() if k != "_raw_clip"}
        for r in filtered
    ])

    st.dataframe(
        display_df,
        hide_index=True,
        width="stretch",
        column_config={
            "Score": st.column_config.ProgressColumn(
                "Score",
                format="%.2f",
                min_value=0.0,
                max_value=1.0,
            ),
            "Start (s)": st.column_config.NumberColumn("Start (s)", format="%.1f"),
            "End (s)": st.column_config.NumberColumn("End (s)", format="%.1f"),
            "Duration (s)": st.column_config.NumberColumn("Duration (s)", format="%.1f"),
        },
    )

    # Inspect dialogue for any candidate
    if filtered:
        with st.expander("Inspect Candidate Script Evidence", icon=":material/search:"):
            clip_choices = {r["Clip ID"]: r["_raw_clip"] for r in filtered}
            chosen_id = st.selectbox("Choose Clip ID to view script lines", list(clip_choices.keys()))
            if chosen_id and chosen_id in clip_choices:
                script_table(clip_choices[chosen_id])


def render_timeline_tab(runs: list[dict]) -> None:
    """Render native st.area_chart showing dialogue density and promo blocks."""
    if not runs:
        st.info("Run an episode to view the timeline.")
        return

    ep_options = [r["episode_id"] for r in runs]
    selected_ep = st.selectbox("Select Episode for Timeline", ep_options)
    run = next((r for r in runs if r["episode_id"] == selected_ep), None)
    if not run:
        return

    events = run.get("episode", {}).get("events", [])
    promos = run.get("promos", [])

    if not events:
        st.caption("No events found in this episode.")
        return

    max_sec = int(max((float(e.get("end_s", 0.0)) for e in events), default=60.0)) + 2

    # Build second-by-second activity DataFrame
    seconds = list(range(max_sec))
    dialogue_density = [0.0] * max_sec

    for e in events:
        if e.get("type") == "dialogue":
            s_idx = max(0, int(float(e.get("start_s", 0.0))))
            e_idx = min(max_sec - 1, int(float(e.get("end_s", 0.0))))
            for t in range(s_idx, e_idx + 1):
                dialogue_density[t] += 1.0

    chart_data = {"Second": seconds, "Dialogue Activity": dialogue_density}

    # Overlay each promo rank as an independent track with density value inside its window
    for promo in promos:
        rank = promo.get("rank", 1)
        start_s = max(0, int(float(promo.get("start_s", 0.0))))
        end_s = min(max_sec - 1, int(float(promo.get("end_s", 0.0))))
        track_col = f"Promo #{rank} ({promo.get('clip_id', '')})"

        promo_vals = [0.0] * max_sec
        for t in range(start_s, end_s + 1):
            promo_vals[t] = max(dialogue_density[t], 1.0)
        chart_data[track_col] = promo_vals

    df_chart = pd.DataFrame(chart_data).set_index("Second")

    st.subheader("Episode Dialogue Density & Promo Window Coverage")
    st.area_chart(df_chart, stack=False, width="stretch")

    if promos:
        segments_summary = " | ".join(
            f"#{p.get('rank', 1)}: {format_timestamp(float(p.get('start_s', 0.0)))}-{format_timestamp(float(p.get('end_s', 0.0)))} ({p.get('duration_s', 0.0):.1f}s)"
            for p in promos
        )
        st.caption(f"**Promo Segments:** {segments_summary}")
        st.caption("Note: The first 3 seconds of each highlighted promo window are evaluated for opening hook intensity.")


def render_quality_tab(runs: list[dict]) -> None:
    """Render Funnel chart, Rejections table, Score distribution, and Model comparison."""
    if not runs:
        st.info("Run an episode to view quality metrics.")
        return

    q_col1, q_col2 = st.columns([1, 1], gap="medium")

    with q_col1:
        st.subheader("Candidate Filtering Funnel")
        # Aggregate funnels across runs
        total_events = sum(len(r.get("episode", {}).get("events", [])) for r in runs)
        raw_windows = sum(r.get("funnel", {}).get("raw_windows", 0) for r in runs)
        after_length = sum(r.get("funnel", {}).get("after_length", 0) for r in runs)
        after_hook = sum(r.get("funnel", {}).get("after_hook", 0) for r in runs)
        after_cliff = sum(r.get("funnel", {}).get("after_cliffhanger", 0) for r in runs)
        promos_count = sum(len(r.get("promos", [])) for r in runs)

        funnel_df = pd.DataFrame([
            {"Stage": "1. Parsed Events", "Count": total_events},
            {"Stage": "2. Raw Windows", "Count": raw_windows},
            {"Stage": "3. Valid Duration (20-60s)", "Count": after_length},
            {"Stage": "4. Hook Passed", "Count": after_hook},
            {"Stage": "5. Cliffhanger Passed", "Count": after_cliff},
            {"Stage": "6. Final Promos", "Count": promos_count},
        ]).set_index("Stage")

        st.bar_chart(funnel_df, width="stretch")

    with q_col2:
        st.subheader("Score Distribution")
        all_promos = [p for r in runs for p in r.get("promos", [])]
        scores = [float(p.get("weighted_score", 0.0)) for p in all_promos]

        buckets = {"0.0 - 0.2": 0, "0.2 - 0.4": 0, "0.4 - 0.6": 0, "0.6 - 0.8": 0, "0.8 - 1.0": 0}
        for s in scores:
            if s < 0.2:
                buckets["0.0 - 0.2"] += 1
            elif s < 0.4:
                buckets["0.2 - 0.4"] += 1
            elif s < 0.6:
                buckets["0.4 - 0.6"] += 1
            elif s < 0.8:
                buckets["0.6 - 0.8"] += 1
            else:
                buckets["0.8 - 1.0"] += 1

        dist_df = pd.DataFrame([{"Score Range": k, "Promos": v} for k, v in buckets.items()]).set_index("Score Range")
        st.bar_chart(dist_df, width="stretch")

    st.divider()

    # Validator rejections
    st.subheader("Validator Diagnostics & Rejections")
    all_rejections = []
    for r in runs:
        ep_id = r.get("episode_id", "ep01")
        # Funnel rejections
        f = r.get("funnel", {})
        for item in f.get("length_rejected", [])[:3]:
            all_rejections.append({"Episode": ep_id, "Type": "Duration", "Reason": item.get("reason", "Outside 20-60s")})
        for item in f.get("cliff_rejected", [])[:3]:
            all_rejections.append({"Episode": ep_id, "Type": "Cliffhanger", "Reason": item.get("cliffhanger_reason", "Neutral ending")})

    m1, m2 = st.columns(2)
    m1.metric("Total Rejected Windows", len(all_rejections))
    rejection_rate = (len(all_rejections) / (raw_windows or 1)) * 100
    m2.metric("Pre-Filter Rejection Rate", f"{rejection_rate:.1f}%")

    if all_rejections:
        st.dataframe(pd.DataFrame(all_rejections), hide_index=True, width="stretch")
    else:
        st.caption("No validator rejection events recorded.")

    st.divider()

    # Two-model compare
    st.subheader("Two-Model Comparison")
    st.caption("Compare top clips when testing different evaluation models or weight presets.")
    if len(runs) >= 2:
        p1_ids = {p.get("clip_id") for p in runs[0].get("promos", [])}
        p2_ids = {p.get("clip_id") for p in runs[1].get("promos", [])}
        overlap_ids = p1_ids.intersection(p2_ids)
        overlap_pct = (len(overlap_ids) / max(len(p1_ids), 1)) * 100

        st.metric("Top Promos Model Overlap", f"{overlap_pct:.0f}%", delta=f"{len(overlap_ids)} common clips")

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            st.write(f"**Model Run 1:** {runs[0].get('episode_id', '')} ({runs[0].get('mode', 'Rules')})")
            df1 = pd.DataFrame([
                {"Rank": p.get("rank"), "Clip ID": p.get("clip_id"), "Score": p.get("weighted_score"), "In Both": p.get("clip_id") in p2_ids}
                for p in runs[0].get("promos", [])
            ])
            st.dataframe(df1, hide_index=True, width="stretch")
        with c2:
            st.write(f"**Model Run 2:** {runs[1].get('episode_id', '')} ({runs[1].get('mode', 'Rules')})")
            df2 = pd.DataFrame([
                {"Rank": p.get("rank"), "Clip ID": p.get("clip_id"), "Score": p.get("weighted_score"), "In Both": p.get("clip_id") in p1_ids}
                for p in runs[1].get("promos", [])
            ])
            st.dataframe(df2, hide_index=True, width="stretch")
    else:
        st.info("Execute a second run or evaluate another episode to enable side-by-side model comparison.")


def render_export_tab(runs: list[dict]) -> None:
    """Render export buttons for CSV, JSON, Markdown, and raw ZIP with file paths."""
    if not runs:
        st.info("No runs to export. Execute promo discovery first.")
        return

    all_promos = [p for r in runs for p in r.get("promos", [])]
    if not all_promos:
        st.warning("No promo records available for export.")
        return

    # Clean promo export data
    clean_promos = []
    for p in all_promos:
        clean_promos.append({
            "rank": p.get("rank", 1),
            "episode": p.get("episode_number", "1"),
            "clip_id": p.get("clip_id", ""),
            "start_seconds": round(float(p.get("start_s", 0.0)), 2),
            "end_seconds": round(float(p.get("end_s", 0.0)), 2),
            "duration_seconds": round(float(p.get("duration_s", 0.0)), 2),
            "start_timestamp": format_timestamp(float(p.get("start_s", 0.0))),
            "end_timestamp": format_timestamp(float(p.get("end_s", 0.0))),
            "weighted_score": round(float(p.get("weighted_score", 0.0)), 3),
            "hook_line": p.get("hook_line", ""),
            "cliffhanger_line": p.get("cliffhanger_line", ""),
            "why_it_works": p.get("why_it_works", ""),
            "low_confidence": bool(p.get("low_confidence", False)),
        })

    # Prepare formats
    df_export = pd.DataFrame(clean_promos)
    csv_bytes = df_export.to_csv(index=False).encode("utf-8")
    json_bytes = json.dumps({"promos": clean_promos}, indent=2, ensure_ascii=False).encode("utf-8")

    # Markdown report
    md_lines = ["# Promo Finder - Evaluation Report", "", f"**Total Promos Discovered:** {len(clean_promos)}", ""]
    for p in clean_promos:
        md_lines.extend([
            f"## #{p['rank']} {p['clip_id']} (Episode {p['episode']})",
            f"- **Time Window:** {p['start_timestamp']} - {p['end_timestamp']} ({p['duration_seconds']:.1f}s)",
            f"- **Score:** {p['weighted_score']:.2f} | Confidence: {'Low' if p['low_confidence'] else 'High'}",
            f"- **Opening Hook:** {p['hook_line']}",
            f"- **Closing Cliffhanger:** {p['cliffhanger_line']}",
            f"- **Editorial Why:** {p['why_it_works']}",
            "",
        ])
    md_bytes = "\n".join(md_lines).encode("utf-8")

    # In-memory ZIP
    zip_bytes = build_export_zip(clean_promos, runs, Path("outputs/raw"))

    # Write files to outputs/ directory
    written_paths = write_all_outputs(clean_promos, base_dir="outputs")

    st.subheader("Export Promo Clips & Reports")
    st.caption("Download structured findings for video editors, social media schedulers, and review archives.")

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        st.download_button(
            "Download promos.csv",
            data=csv_bytes,
            file_name="promos.csv",
            mime="text/csv",
            icon=":material/table_view:",
            width="stretch",
        )
    with c2:
        st.download_button(
            "Download promos.json",
            data=json_bytes,
            file_name="promos.json",
            mime="application/json",
            icon=":material/data_object:",
            width="stretch",
        )
    with c3:
        st.download_button(
            "Download report.md",
            data=md_bytes,
            file_name="report.md",
            mime="text/markdown",
            icon=":material/description:",
            width="stretch",
        )
    with c4:
        st.download_button(
            "Download outputs.zip",
            data=zip_bytes,
            file_name="promo_finder_bundle.zip",
            mime="application/zip",
            icon=":material/archive:",
            width="stretch",
        )

    st.divider()
    st.caption("Persisted local file paths:")
    for path in (written_paths.values() if isinstance(written_paths, dict) else written_paths):
        st.caption(f"• `{path}`")
