"""Promo Finder - Main Streamlit application and entry point (under 100 lines)."""

import time
from pathlib import Path
import streamlit as st

from src.pipeline import run_episode_pipeline, PipelineError
from src.rank import rerank_existing_run
from src.app_core import save_run
from ui.header import render_header, render_footer
from ui.sidebar import render_sidebar
from ui.results import render_results
from ui.components import empty_state, error_box, render_style_showcase

SAMPLE_PATH = Path("data/I Married a Broken Soldier. He was a Duke/episode-001-Episode 1671703.txt")


def init_state() -> None:
    for k, v in {"runs": [], "_prev_weights": None, "mode": "Rules only", "stop_run": False}.items():
        st.session_state.setdefault(k, v)


def handle_sample(cfg: dict) -> None:
    sample = SAMPLE_PATH if SAMPLE_PATH.exists() else next(Path("data").glob("**/*.txt"), None)
    if sample:
        txt = sample.read_text(encoding="utf-8-sig")
        st.session_state.update(sample_loaded_text=txt, sample_loaded_name=sample.name)
        res = run_episode_pipeline(txt, sample.name, mode="Rules only", weights=cfg["weights"], max_promos=5)
        st.session_state.runs = [res]
        save_run(res["episode_id"], sample.name, res)
        st.toast("Loaded and analyzed Episode 1.", icon=":material/check_circle:")
        st.rerun()


def run_streamlit() -> None:
    st.set_page_config(page_title="Promo Finder", layout="wide", page_icon=None)
    init_state()
    if st.query_params.get("style") == "1":
        render_style_showcase()
        return

    cfg = render_sidebar()

    if st.session_state.runs and cfg["weights"] != st.session_state._prev_weights:
        if st.session_state._prev_weights is not None:
            st.session_state.runs = [rerank_existing_run(r, cfg["weights"], max_promos=cfg["max_promos"]) for r in st.session_state.runs]
            st.toast("Instant re-rank applied (0 LLM calls).", icon=":material/tune:")
        st.session_state._prev_weights = dict(cfg["weights"])

    render_header(cfg["mode"], cfg["model"], len(cfg["loaded_scripts"]), sum(len(r.get("promos", [])) for r in st.session_state.runs))

    if cfg["find_promos_clicked"]:
        st.session_state.runs, st.session_state.stop_run = [], False
        with st.status("Executing promo discovery pipeline...", expanded=True) as status_box:
            counters, t0 = st.empty(), time.time()
            for idx, item in enumerate(cfg["loaded_scripts"], 1):
                if st.session_state.get("stop_run"):
                    status_box.update(label="Run stopped by user.", state="error", expanded=False)
                    break
                status_box.update(label=f"Scoring episode {idx} of {len(cfg['loaded_scripts'])}: {item['filename']}...")
                try:
                    res = run_episode_pipeline(
                        text=item["text"], filename=item["filename"], mode=cfg["mode"],
                        weights=cfg["weights"], min_seconds=cfg["min_seconds"],
                        max_seconds=cfg["max_seconds"], top_n=cfg["top_n"], max_promos=cfg["max_promos"],
                        provider=cfg["provider"], model=cfg["model"], api_key=cfg["api_key"],
                        on_step=lambda msg: status_box.write(f"• {msg}"),
                        check_stop=lambda: st.session_state.get("stop_run", False),
                    )
                    st.session_state.runs.append(res)
                    save_run(res["episode_id"], item["filename"], res)
                    s = res.get("stats", {})
                    counters.caption(f"LLM Calls: **{s.get('llm_calls', 0)}** | Cache Hits: **{s.get('cache_hits', 0)}** | Rejected: **{s.get('rejections', 0)}** | Elapsed: **{time.time()-t0:.1f}s**")
                except PipelineError as exc:
                    error_box(f"Pipeline Error in {item['filename']}", exc.message, exc.suggested_fix)
            if not st.session_state.get("stop_run"):
                status_box.update(label="Promo discovery complete.", state="complete", expanded=False)

    if st.session_state.runs:
        render_results(st.session_state.runs, model_name=cfg["model"])
    elif cfg["loaded_scripts"]:
        with st.container(border=True):
            st.subheader(f"Screenplay Ready: {cfg['loaded_scripts'][0]['filename']}")
            st.caption(f"{cfg['loaded_scripts'][0]['inspection'].get('event_count', 0)} screenplay events loaded and validated.")
            if st.button("Find Promo Clips Now", type="primary", icon=":material/search:"):
                item = cfg["loaded_scripts"][0]
                res = run_episode_pipeline(item["text"], item["filename"], mode=cfg["mode"], weights=cfg["weights"], max_promos=cfg["max_promos"])
                st.session_state.runs = [res]
                save_run(res["episode_id"], item["filename"], res)
                st.rerun()
    else:
        empty_state(on_load_sample=lambda: handle_sample(cfg), on_demo_run=lambda: handle_sample(cfg))

    render_footer(mode=cfg["mode"], model_name=cfg["model"])


if __name__ == "__main__":
    run_streamlit()
