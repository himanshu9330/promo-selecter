"""Reusable native Streamlit components for Promo Finder.

Adheres strictly to native Streamlit: no unsafe_allow_html, no external charting
libraries, consistent typography, semantic Streamlit markdown colors, and clean
spacing patterns.
"""

from typing import Callable
import pandas as pd
import streamlit as st

from src.parse import format_timestamp
from src.app_core import (
    snap_to_line_boundary,
    save_editor_feedback,
    read_editor_feedback,
)


def badge_row(badges: list[dict]) -> None:
    """Render a horizontal row of semantic badges using native st.badge."""
    if not badges:
        return
    cols = st.columns(len(badges), gap="small")
    for col, badge in zip(cols, badges):
        with col:
            st.badge(
                label=badge["label"],
                color=badge.get("color", "gray"),
                icon=badge.get("icon"),
            )


def score_row(item: dict) -> None:
    """Render 4 native progress bars for hook, cliffhanger, standalone, and emotion."""
    scores = item.get("scores", {})
    hook = float(scores.get("hook", item.get("prefilter_score", 5) / 3))
    cliff = float(scores.get("cliffhanger", scores.get("hook", 5)))
    standalone = float(item.get("standalone", 8))
    emotion = float(scores.get("emotion", 5))

    c1, c2, c3, c4 = st.columns(4, gap="small")
    with c1:
        st.progress(min(1.0, max(0.0, hook / 10.0)), text=f"Hook {hook:.1f}/10")
    with c2:
        st.progress(min(1.0, max(0.0, cliff / 10.0)), text=f"Cliffhanger {cliff:.1f}/10")
    with c3:
        st.progress(min(1.0, max(0.0, standalone / 10.0)), text=f"Standalone {standalone:.1f}/10")
    with c4:
        st.progress(min(1.0, max(0.0, emotion / 10.0)), text=f"Emotion {emotion:.1f}/10")


def script_table(clip: dict) -> None:
    """Render the exact lines of a clip inside a native st.dataframe with role markers."""
    dialogue = clip.get("dialogue", [])
    if not dialogue:
        st.caption("No dialogue events recorded for this segment.")
        return

    rows = []
    first_idx = 0
    last_idx = len(dialogue) - 1

    for idx, event in enumerate(dialogue):
        start_fmt = format_timestamp(float(event.get("start_s", 0.0)))
        end_fmt = format_timestamp(float(event.get("end_s", 0.0)))

        role = ""
        if idx == first_idx:
            role = "HOOK"
        elif idx == last_idx:
            role = "CLIFFHANGER"

        rows.append({
            "Time": f"{start_fmt} - {end_fmt}",
            "Role": role,
            "Speaker": event.get("speaker", "UNKNOWN"),
            "Tone": event.get("tone", ""),
            "Line": event.get("text", ""),
        })

    df = pd.DataFrame(rows)
    st.dataframe(
        df,
        hide_index=True,
        width="stretch",
        column_config={
            "Time": st.column_config.TextColumn("Time", width="small"),
            "Role": st.column_config.TextColumn("Role", width="small"),
            "Speaker": st.column_config.TextColumn("Speaker", width="medium"),
            "Tone": st.column_config.TextColumn("Tone", width="small"),
            "Line": st.column_config.TextColumn("Line", width="large"),
        },
    )


def editor_actions(clip: dict, key_prefix: str, model_name: str = "") -> None:
    """Render Accept, Reject, and Snap-to-Line Trim popover controls."""
    clip_id = clip.get("clip_id", "clip")
    decision_key = f"decision_{clip_id}"

    # Read feedback state from session or disk
    current_decision = st.session_state.get(decision_key)
    if not current_decision:
        feedback = read_editor_feedback()
        for row in reversed(feedback):
            if row.get("clip_id") == clip_id:
                current_decision = row.get("decision")
                st.session_state[decision_key] = current_decision
                break

    c1, c2, c3, c4 = st.columns([2, 2, 2, 3], gap="small")

    with c1:
        if st.button(
            "Accept",
            key=f"{key_prefix}_accept_{clip_id}",
            type="primary" if current_decision == "accept" else "secondary",
            icon=":material/check:",
            width="stretch",
        ):
            save_editor_feedback(clip_id, "accept", float(clip["start_s"]), float(clip["end_s"]), model_name)
            st.session_state[decision_key] = "accept"
            st.toast(f"Accepted clip {clip_id}", icon=":material/check_circle:")
            st.rerun()

    with c2:
        if st.button(
            "Reject",
            key=f"{key_prefix}_reject_{clip_id}",
            type="primary" if current_decision == "reject" else "secondary",
            icon=":material/close:",
            width="stretch",
        ):
            save_editor_feedback(clip_id, "reject", float(clip["start_s"]), float(clip["end_s"]), model_name)
            st.session_state[decision_key] = "reject"
            st.toast(f"Rejected clip {clip_id}", icon=":material/cancel:")
            st.rerun()

    with c3:
        with st.popover("Trim", icon=":material/content_cut:", width="stretch"):
            st.subheader("Trim Boundaries")
            st.caption("Values snap to nearest dialogue line start and end.")
            orig_start = float(clip["start_s"])
            orig_end = float(clip["end_s"])

            t_start = st.number_input(
                "Start offset (seconds)",
                value=0.0,
                step=0.5,
                key=f"{key_prefix}_tstart_{clip_id}",
            )
            t_end = st.number_input(
                "End offset (seconds)",
                value=0.0,
                step=0.5,
                key=f"{key_prefix}_tend_{clip_id}",
            )

            snapped_start = snap_to_line_boundary(orig_start + t_start, clip, "start")
            snapped_end = snap_to_line_boundary(orig_end + t_end, clip, "end")
            new_duration = snapped_end - snapped_start

            st.write(
                f"**New Window:** {format_timestamp(snapped_start)} - {format_timestamp(snapped_end)} "
                f"(:violet[{new_duration:.1f}s])"
            )

            if new_duration < 20.0 or new_duration > 60.0:
                st.warning(f"Duration is {new_duration:.1f}s. Recommended promo length is 20s-60s.")

            if st.button("Apply and Save Trim", key=f"{key_prefix}_savetrim_{clip_id}", type="primary"):
                save_editor_feedback(clip_id, f"trimmed_{new_duration:.1f}s", snapped_start, snapped_end, model_name)
                st.session_state[decision_key] = "trimmed"
                st.toast(f"Trim saved ({new_duration:.1f}s)", icon=":material/content_cut:")
                st.rerun()

    with c4:
        if current_decision:
            color = "green" if current_decision == "accept" else "red" if current_decision == "reject" else "orange"
            st.badge(f"Decision: {current_decision.capitalize()}", color=color)
            if st.button("Undo", key=f"{key_prefix}_undo_{clip_id}", icon=":material/undo:"):
                st.session_state[decision_key] = None
                from src.app_core import ROOT
                feedback_file = ROOT / "outputs" / "editor_feedback.csv"
                if feedback_file.exists():
                    import csv
                    rows = [r for r in read_editor_feedback() if r.get("clip_id") != clip_id]
                    with feedback_file.open("w", newline="", encoding="utf-8") as h:
                        fieldnames = ["clip_id", "decision", "start_trim_seconds", "end_trim_seconds", "model", "timestamp"]
                        w = csv.DictWriter(h, fieldnames=fieldnames)
                        w.writeheader()
                        if rows:
                            w.writerows(rows)
                st.toast(f"Undid decision for {clip_id}", icon=":material/undo:")
                st.rerun()


def storyboard_table(clip: dict) -> None:
    """Render camera and action lines as a storyboard table when no video is provided."""
    events = [e for e in clip.get("events", []) if e.get("type") in {"camera", "action"}]
    if not events:
        st.caption("No explicit camera/action cues recorded for this segment.")
        return

    rows = []
    for e in events:
        s_fmt = format_timestamp(float(e.get("start_s", 0.0)))
        e_fmt = format_timestamp(float(e.get("end_s", 0.0)))
        rows.append({
            "Time": f"{s_fmt} - {e_fmt}",
            "Type": e.get("type", "").upper(),
            "Cue": e.get("text", ""),
        })

    st.caption("Storyboard Visual Cues (Camera & Action):")
    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
        column_config={
            "Time": st.column_config.TextColumn("Time", width="small"),
            "Type": st.column_config.TextColumn("Type", width="small"),
            "Cue": st.column_config.TextColumn("Cue", width="large"),
        },
    )


def top_pick_card(promo: dict, key_prefix: str = "top", model_name: str = "") -> None:
    """Render the primary rank-1 hero container."""
    clip_id = promo.get("clip_id", "")
    start_fmt = format_timestamp(float(promo.get("start_s", 0.0)))
    end_fmt = format_timestamp(float(promo.get("end_s", 0.0)))
    duration = float(promo.get("duration_s", 0.0))
    score = float(promo.get("weighted_score", 0.0))

    with st.container(border=True):
        head_col, metric_col = st.columns([3, 2])
        with head_col:
            st.header(f"#1 Top Pick :violet[{clip_id}]")
            st.caption(f"Episode {promo.get('episode_number', '1')} | Master Promo Recommendation")
        with metric_col:
            m1, m2 = st.columns(2)
            with m1:
                st.metric("Segment Range", f"{start_fmt} - {end_fmt}")
            with m2:
                st.metric("Duration / Score", f"{duration:.1f}s", delta=f"{score:.2f} score")

        st.divider()

        # Quotes with semantic bold labels and native markdown blockquotes
        st.markdown(f"> **Opening Hook:** {promo.get('hook_line', 'No hook line text available.')}")
        st.markdown(f"> **Closing Cliffhanger:** {promo.get('cliffhanger_line', 'No cliffhanger text available.')}")

        st.caption(f"Reason: {promo.get('why_it_works', 'Strong narrative momentum and unanswered question.')}")

        # Scores
        st.subheader("Component Scores")
        score_row(promo)

        # Badges
        badges = []
        if promo.get("low_confidence"):
            badges.append({"label": "Low confidence", "color": "orange", "icon": ":material/warning:"})
        else:
            badges.append({"label": "High confidence", "color": "green", "icon": ":material/verified:"})

        if promo.get("cross_episode"):
            badges.append({"label": "Cross-episode join", "color": "violet", "icon": ":material/join_full:"})

        badge_row(badges)

        st.divider()

        with st.expander("Show Script Evidence", icon=":material/menu_book:"):
            script_table(promo)

        with st.expander("Preview (Optional Video or Storyboard)", icon=":material/videocam:"):
            vid_file = st.file_uploader(
                f"Upload episode video for {clip_id} (.mp4)",
                type=["mp4", "mov"],
                key=f"{key_prefix}_video_{clip_id}",
            )
            if vid_file is not None:
                st.video(vid_file, start_time=int(float(promo.get("start_s", 0))), end_time=int(float(promo.get("end_s", 60))))
            else:
                storyboard_table(promo)

        editor_actions(promo, key_prefix=key_prefix, model_name=model_name)


def clip_card(promo: dict, rank: int, key_prefix: str = "", model_name: str = "") -> None:
    """Render a compact promo card with an expandable detail view."""
    clip_id = promo.get("clip_id", "")
    ep_num = promo.get("episode_number", "1")
    start_fmt = format_timestamp(float(promo.get("start_s", 0.0)))
    end_fmt = format_timestamp(float(promo.get("end_s", 0.0)))
    duration = float(promo.get("duration_s", 0.0))
    score = float(promo.get("weighted_score", 0.0))
    hook_text = promo.get("hook_line", "")
    short_hook = (hook_text[:60] + "...") if len(hook_text) > 60 else hook_text

    with st.container(border=True):
        c_rank, c_meta, c_hook, c_score = st.columns([1, 4, 6, 2], gap="small")
        with c_rank:
            st.subheader(f"#{rank}")
        with c_meta:
            st.write(f"**Ep {ep_num}** (:violet[{clip_id}]) | {start_fmt} - {end_fmt}")
            st.caption(f"{duration:.1f}s duration")
        with c_hook:
            st.write(f'"{short_hook}"')
            st.caption(promo.get("why_it_works", ""))
        with c_score:
            st.metric("Score", f"{score:.2f}")

        with st.expander("Details and Script", icon=":material/unfold_more:"):
            st.markdown(f"> **Opening Hook:** {promo.get('hook_line', '')}")
            st.markdown(f"> **Closing Cliffhanger:** {promo.get('cliffhanger_line', '')}")

            st.caption("Component Scores:")
            score_row(promo)

            badges = []
            if promo.get("low_confidence"):
                badges.append({"label": "Low confidence", "color": "orange", "icon": ":material/warning:"})
            else:
                badges.append({"label": "Validated", "color": "green", "icon": ":material/verified:"})
            badge_row(badges)

            st.divider()
            script_table(promo)

            with st.expander("Preview (Optional Video or Storyboard)", icon=":material/videocam:"):
                vid_file = st.file_uploader(
                    f"Upload video for #{rank} (.mp4)",
                    type=["mp4", "mov"],
                    key=f"{key_prefix}_vid_{clip_id}",
                )
                if vid_file is not None:
                    st.video(vid_file, start_time=int(float(promo.get("start_s", 0))), end_time=int(float(promo.get("end_s", 60))))
                else:
                    storyboard_table(promo)

            editor_actions(promo, key_prefix=f"{key_prefix}_{rank}", model_name=model_name)


def empty_state(on_load_sample: Callable[[], None], on_demo_run: Callable[[], None]) -> None:
    """Render the Linear-style calm onboarding state."""
    with st.container(border=True):
        st.subheader("Promo Clip Discovery")
        st.caption(
            "Detect high-retention 20s-60s microdrama promotional cuts with strict timestamp validation."
        )

        st.divider()

        c1, c2, c3 = st.columns(3, gap="medium")
        with c1:
            st.write("**:violet[1. Provide Screenplay]**")
            st.caption(
                "Upload one or more episode .txt scripts in the sidebar, paste a script directly, or load the built-in sample."
            )
        with c2:
            st.write("**:violet[2. Select Mode]**")
            st.caption(
                "Use **Fast Rules Engine** for instant offline candidate cuts, or **AI Judge** for semantic evaluation."
            )
        with c3:
            st.write("**:violet[3. Review and Export]**")
            st.caption(
                "Inspect the top pick, verify exact script dialogue lines, fine-tune boundaries, and download CSV/JSON/Markdown."
            )

        st.divider()

        b1, b2, _ = st.columns([2, 2, 4], gap="small")
        with b1:
            if st.button("Load Sample Episode", icon=":material/folder_open:", width="stretch"):
                on_load_sample()
        with b2:
            if st.button("Demo Run (Offline)", type="primary", icon=":material/play_arrow:", width="stretch"):
                on_demo_run()


def error_box(title: str, message: str, fix: str = "") -> None:
    """Render a structured error alert with an actionable suggested fix."""
    st.error(f"**{title}**\n\n{message}")
    if fix:
        st.info(f"**Suggested fix:** {fix}", icon=":material/lightbulb:")


def render_style_showcase() -> None:
    """Render the hidden ?style=1 design system preview."""
    st.title("Promo Finder Design System Preview")
    st.caption("Visual audit of all native UI components in light and dark mode.")

    st.header("1. Semantic Badges")
    badge_row([
        {"label": "Validated", "color": "green", "icon": ":material/verified:"},
        {"label": "Low confidence", "color": "orange", "icon": ":material/warning:"},
        {"label": "Rejected by rules", "color": "red", "icon": ":material/block:"},
        {"label": "Cross-episode join", "color": "violet", "icon": ":material/join_full:"},
        {"label": "Neutral signal", "color": "gray"},
    ])

    st.header("2. Empty State")
    empty_state(on_load_sample=lambda: None, on_demo_run=lambda: None)

    st.header("3. Error and Guidance Box")
    error_box(
        title="Screenplay Parse Warning",
        message="Episode 7 contained 3 timestamps with unaligned millisecond boundaries.",
        fix="Ensure lines follow the format 'Start: (mm:ss.s) | End: (mm:ss.s)'.",
    )

    st.header("4. Component Score Rows")
    score_row({
        "scores": {"hook": 9.2, "cliffhanger": 8.8, "emotion": 7.5},
        "standalone": 9.0,
    })

    st.header("5. Top Pick Hero Card")
    mock_clip = {
        "clip_id": "ep01_w137",
        "episode_number": "1",
        "start_s": 46.8,
        "end_s": 81.2,
        "duration_s": 34.4,
        "weighted_score": 0.88,
        "hook_line": "JULIANA (Shocked): You were the broken soldier all along?",
        "cliffhanger_line": "DUKE (Whisper): And now that you know... you cannot leave.",
        "why_it_works": "Immediate identity reveal followed by high-stakes ultimatum.",
        "scores": {"hook": 9.5, "cliffhanger": 9.0, "emotion": 8.5},
        "standalone": 9.0,
        "dialogue": [
            {"start_s": 46.8, "end_s": 50.2, "speaker": "JULIANA", "tone": "Shocked", "text": "You were the broken soldier all along?"},
            {"start_s": 50.5, "end_s": 55.0, "speaker": "DUKE", "tone": "Calm", "text": "I never lied to you about the war."},
            {"start_s": 75.0, "end_s": 81.2, "speaker": "DUKE", "tone": "Whisper", "text": "And now that you know... you cannot leave."},
        ],
    }
    top_pick_card(mock_clip, key_prefix="showcase_hero")

    st.header("6. Compact Clip Card")
    clip_card(mock_clip, rank=2, key_prefix="showcase_compact")
