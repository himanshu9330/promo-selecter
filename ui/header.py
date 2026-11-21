"""Header, quiet status strip, and footer components for Promo Finder using native Streamlit."""

import streamlit as st


def render_header(mode: str, model_name: str, episodes_count: int, promos_count: int = 0) -> None:
    """Render application header and quiet status strip using native columns and typography."""
    title_col, status_col = st.columns([3, 2], gap="medium")

    with title_col:
        st.title("Promo Finder")
        st.caption("High-retention 20s-60s promotional clip discovery for microdrama screenplays.")

    with status_col:
        st.write("")  # Align with title baseline
        model_display = model_name if mode == "Rules + LLM" else "Rules Engine"
        st.caption(
            f"Mode: **:violet[{mode}]** | Model: **{model_display}** | Episodes: **{episodes_count}**"
        )

    st.divider()


def render_footer(mode: str = "Rules only", model_name: str = "", elapsed_s: float = 0.0) -> None:
    """Render calm accessible native footer with required disclaimer."""
    st.divider()
    st.caption("Scores come from the script text only and cannot see the visuals.")
