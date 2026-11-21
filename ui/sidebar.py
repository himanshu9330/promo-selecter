"""Native Streamlit sidebar controls for Promo Finder."""

from pathlib import Path
import pandas as pd
import streamlit as st

from src.parse import inspect_screenplay
from src.llm import check_llm_connection
from src.rank import normalize_weights
from src.app_core import read_editor_feedback


DEFAULT_WEIGHTS = {
    "hook": 0.30,
    "cliffhanger": 0.30,
    "standalone": 0.25,
    "emotion": 0.15,
}

SAMPLE_PATH = Path("data/I Married a Broken Soldier. He was a Duke/episode-001-Episode 1671703.txt")


def render_sidebar() -> dict:
    """Render calm, professional sidebar controls using native Streamlit."""
    st.sidebar.header("Controls")

    # Accept rate metric
    feedback = read_editor_feedback()
    if feedback:
        accepted_count = sum(1 for row in feedback if row.get("decision") == "accept")
        rate_val = f"{(accepted_count / len(feedback)) * 100:.0f}%"
    else:
        rate_val = "N/A"
    st.sidebar.metric("Accept rate", rate_val)

    loaded_scripts: list[dict] = []

    # ---------------------------------------------------------
    # 1. Input Expander (open by default)
    # ---------------------------------------------------------
    with st.sidebar.expander("Input", expanded=True):
        uploaded_files = st.file_uploader(
            "Screenplay files (.txt)",
            type=["txt"],
            accept_multiple_files=True,
            help="Upload one or more screenplay .txt files.",
        )

        b_sample_col, _ = st.columns([3, 1])
        with b_sample_col:
            if st.button("Load sample", icon=":material/folder_open:", width="stretch"):
                sample_file = SAMPLE_PATH
                if not sample_file.exists():
                    candidates = list(Path("data").glob("**/*.txt"))
                    if candidates:
                        sample_file = candidates[0]
                if sample_file.exists():
                    st.session_state["sample_loaded_text"] = sample_file.read_text(encoding="utf-8-sig")
                    st.session_state["sample_loaded_name"] = sample_file.name
                    st.toast("Loaded sample episode", icon=":material/check:")
                    st.rerun()
                else:
                    st.error("Sample screenplay not found in data/.")

        with st.expander("Paste a script"):
            pasted_text = st.text_area(
                "Paste screenplay text",
                height=140,
                placeholder="EPISODE NUMBER: 1\n---\n00:00.0 Speaker: Text...",
                help="Paste an unseen script for a live demo run.",
            )

        # Collect scripts
        if uploaded_files:
            for f in uploaded_files:
                try:
                    txt = f.getvalue().decode("utf-8-sig", errors="replace")
                    insp = inspect_screenplay(txt, filename=f.name)
                    loaded_scripts.append({"filename": f.name, "text": txt, "inspection": insp})
                except Exception as exc:
                    st.error(f"Error reading {f.name}: {exc}")

        if pasted_text.strip():
            insp = inspect_screenplay(pasted_text, filename="pasted_script.txt")
            loaded_scripts.append({"filename": "pasted_script.txt", "text": pasted_text, "inspection": insp})

        if st.session_state.get("sample_loaded_text") and not uploaded_files and not pasted_text.strip():
            sample_txt = st.session_state["sample_loaded_text"]
            sample_name = st.session_state.get("sample_loaded_name", "episode-001-sample.txt")
            insp = inspect_screenplay(sample_txt, filename=sample_name)
            loaded_scripts.append({"filename": sample_name, "text": sample_txt, "inspection": insp})

        # Table of loaded files
        if loaded_scripts:
            st.caption(f"{len(loaded_scripts)} episode(s) ready:")
            file_rows = [
                {
                    "File": s["filename"],
                    "Ep": s["inspection"].get("episode_number", "?"),
                    "Events": s["inspection"].get("event_count", 0),
                    "Warnings": len(s["inspection"].get("warnings", [])),
                }
                for s in loaded_scripts
            ]
            st.dataframe(
                pd.DataFrame(file_rows),
                hide_index=True,
                width="stretch",
                column_config={
                    "File": st.column_config.TextColumn("File", width="medium"),
                    "Ep": st.column_config.TextColumn("Ep", width="small"),
                    "Events": st.column_config.NumberColumn("Events", width="small"),
                    "Warnings": st.column_config.NumberColumn("Warnings", width="small"),
                },
            )

    # ---------------------------------------------------------
    # 2. Model Expander (closed by default)
    # ---------------------------------------------------------
    with st.sidebar.expander("Model", expanded=False):
        mode = st.segmented_control(
            "Evaluation mode",
            options=["Rules only", "Rules + LLM"],
            default=st.session_state.get("mode", "Rules only"),
            help="Rules only runs 100% locally with zero latency. Rules + LLM validates candidate hooks with an LLM judge.",
        )
        if mode is None:
            mode = "Rules only"
        st.session_state.mode = mode

        provider = st.selectbox(
            "Provider",
            ["OpenRouter", "Gemini", "Ollama"],
            index=["OpenRouter", "Gemini", "Ollama"].index(st.session_state.get("provider", "OpenRouter")),
            disabled=(mode == "Rules only"),
        )
        st.session_state.provider = provider

        model = st.text_input(
            "Model identifier",
            value=st.session_state.get("model", "nvidia/nemotron-3-ultra-550b-a55b"),
            disabled=(mode == "Rules only"),
        )
        st.session_state.model = model

        api_key = st.text_input(
            "API key",
            type="password",
            value=st.session_state.get("api_key", ""),
            disabled=(mode == "Rules only"),
            help="Stored only in memory for this session.",
        )
        st.session_state.api_key = api_key

        if mode == "Rules + LLM":
            if st.button("Test connection", icon=":material/wifi:"):
                with st.spinner("Testing API connection..."):
                    ok, msg = check_llm_connection(provider=provider, model=model, api_key=api_key)
                    if ok:
                        st.success("API connection verified.")
                    else:
                        st.error(f"Connection failed: {msg}")

    # ---------------------------------------------------------
    # 3. Clip Rules Expander (closed by default)
    # ---------------------------------------------------------
    with st.sidebar.expander("Clip rules", expanded=False):
        min_s, max_s = st.slider(
            "Duration window (seconds)",
            20, 60,
            (st.session_state.get("min_seconds", 20), st.session_state.get("max_seconds", 60)),
            help="Target duration range for promotional cuts.",
        )
        st.session_state.min_seconds = min_s
        st.session_state.max_seconds = max_s

        top_n = st.slider(
            "Candidate pool size",
            5, 25,
            st.session_state.get("top_n", 15),
            help="Pre-filter candidate pool size per episode.",
        )
        st.session_state.top_n = top_n

        max_promos = st.slider(
            "Max promos per episode",
            1, 10,
            st.session_state.get("max_promos", 5),
            help="Maximum ranked promo cuts output per episode.",
        )
        st.session_state.max_promos = max_promos

        allow_cross = st.toggle(
            "Allow cross-episode joins",
            value=st.session_state.get("allow_cross_ep", False),
            help="Allow promo candidate windows to span episode transitions.",
        )
        st.session_state.allow_cross_ep = allow_cross

    # ---------------------------------------------------------
    # 4. Ranking Weights Expander (closed by default)
    # ---------------------------------------------------------
    with st.sidebar.expander("Ranking weights", expanded=False):
        current_w = st.session_state.get("weights", DEFAULT_WEIGHTS)

        hook_w = st.slider("Hook weight", 0.0, 1.0, float(current_w["hook"]), step=0.05)
        cliff_w = st.slider("Cliffhanger weight", 0.0, 1.0, float(current_w["cliffhanger"]), step=0.05)
        stand_w = st.slider("Standalone weight", 0.0, 1.0, float(current_w["standalone"]), step=0.05)
        emot_w = st.slider("Emotion weight", 0.0, 1.0, float(current_w["emotion"]), step=0.05)

        norm_w = normalize_weights(hook_w, cliff_w, stand_w, emot_w)
        st.session_state.weights = norm_w

        st.caption(
            f"Effective: **Hook {norm_w['hook']*100:.0f}%** | "
            f"**Cliff {norm_w['cliffhanger']*100:.0f}%** | "
            f"**Standalone {norm_w['standalone']*100:.0f}%** | "
            f"**Emotion {norm_w['emotion']*100:.0f}%**"
        )

        if st.button("Reset to defaults", icon=":material/refresh:"):
            st.session_state.weights = dict(DEFAULT_WEIGHTS)
            st.rerun()

    # ---------------------------------------------------------
    # 5. Primary Action Button
    # ---------------------------------------------------------
    can_run = True
    disabled_reason = ""

    if not loaded_scripts:
        can_run = False
        disabled_reason = "Provide at least one screenplay via upload, sample, or paste."
    elif mode == "Rules + LLM" and not api_key and provider.lower() != "ollama":
        can_run = False
        disabled_reason = "Enter an API key in the Model section, or switch to Rules only mode."

    find_clicked = st.sidebar.button(
        "Find promos",
        type="primary",
        disabled=not can_run,
        width="stretch",
        icon=":material/search:",
    )

    if not can_run and disabled_reason:
        st.sidebar.caption(f":orange[{disabled_reason}]")

    return {
        "mode": mode,
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "min_seconds": min_s,
        "max_seconds": max_s,
        "top_n": top_n,
        "max_promos": max_promos,
        "allow_cross_ep": allow_cross,
        "weights": norm_w,
        "find_promos_clicked": find_clicked,
        "can_run": can_run,
        "loaded_scripts": loaded_scripts,
    }
