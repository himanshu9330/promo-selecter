"""Command-line entry point for Promo Finder."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.pipeline import run_episode_pipeline, PipelineError
from src.report import write_all_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Find microdrama promo clips from timestamped screenplays")
    parser.add_argument("screenplay", help="Path to a timestamped screenplay .txt file")
    parser.add_argument("--mode", choices=["Rules only", "Rules + LLM"], default="Rules only", help="Evaluation mode")
    parser.add_argument("--max-promos", type=int, default=5, help="Maximum promos to return")
    parser.add_argument("--output-dir", default="outputs", help="Directory to save outputs (promos.csv, promos.json, report.md)")
    args = parser.parse_args()

    script_path = Path(args.screenplay)
    if not script_path.exists():
        sys.exit(f"Error: Screenplay file '{args.screenplay}' not found.")

    text = script_path.read_text(encoding="utf-8-sig")
    print(f"Analyzing {script_path.name} in '{args.mode}' mode...")

    try:
        result = run_episode_pipeline(
            text=text,
            filename=script_path.name,
            mode=args.mode,
            weights={"hook": 0.30, "cliffhanger": 0.30, "standalone": 0.25, "emotion": 0.15},
            max_promos=args.max_promos,
            on_step=lambda s: print(f"  • {s}"),
        )
    except PipelineError as exc:
        sys.exit(f"Pipeline error: {exc.message}\nSuggested fix: {exc.suggested_fix}")

    promos = result.get("promos", [])
    print(f"\nDiscovered {len(promos)} promo cut(s):")
    for p in promos:
        dur = float(p.get("duration_s", 0.0))
        s_s = float(p.get("start_s", 0.0))
        e_s = float(p.get("end_s", 0.0))
        print(f"  #{p.get('rank', 1)} {p.get('clip_id', 'clip')} | {s_s:.1f}s - {e_s:.1f}s ({dur:.1f}s) | Score: {p.get('weighted_score', 0.0):.2f}")

    written = write_all_outputs(promos, base_dir=args.output_dir)
    print(f"\nOutputs written to '{args.output_dir}/':")
    for k, path in written.items():
        print(f"  - {Path(path).name}")


if __name__ == "__main__":
    main()
