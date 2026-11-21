"""Write final clip reports under outputs/."""

import csv
import json
from pathlib import Path


def write_report(output_path: str, report: dict) -> None:
    """Write a JSON report and create its parent directory."""
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def write_promos_json(output_path: str, promos: list[dict]) -> None:
    write_report(output_path, {"promos": promos})


def write_promos_csv(output_path: str, promos: list[dict]) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "rank",
        "episode_number",
        "clip_id",
        "start_s",
        "end_s",
        "duration_s",
        "weighted_score",
        "hook_line",
        "cliffhanger_line",
        "why_it_works",
        "hook",
        "cliffhanger",
        "emotion",
        "standalone",
    ]
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for promo in promos:
            scores = promo.get("scores", {})
            writer.writerow(
                {
                    "rank": promo.get("rank", 1),
                    "episode_number": promo.get("episode_number", ""),
                    "clip_id": promo.get("clip_id", ""),
                    "start_s": promo.get("start_s", 0.0),
                    "end_s": promo.get("end_s", 0.0),
                    "duration_s": promo.get("duration_s", 0.0),
                    "weighted_score": promo.get("weighted_score", 0.0),
                    "hook_line": promo.get("hook_line", ""),
                    "cliffhanger_line": promo.get("cliffhanger_line", ""),
                    "why_it_works": promo.get("why_it_works", ""),
                    "hook": scores.get("hook", promo.get("hook", 8)),
                    "cliffhanger": scores.get("cliffhanger", promo.get("cliffhanger", 8)),
                    "emotion": scores.get("emotion", promo.get("emotion", 8)),
                    "standalone": promo.get("standalone", 8),
                }
            )


def write_report_md(output_path: str, promos: list[dict]) -> None:
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Promo Clip Report", ""]
    for promo in promos:
        scores = promo.get("scores", {})
        h = scores.get("hook", promo.get("hook", 8))
        c = scores.get("cliffhanger", promo.get("cliffhanger", 8))
        e = scores.get("emotion", promo.get("emotion", 8))
        s = promo.get("standalone", 8)
        lines.extend(
            [
                f"## Rank {promo.get('rank', 1)}: {promo.get('clip_id', '')}",
                "",
                f"- Episode: {promo.get('episode_number', '')}",
                f"- Segment: {float(promo.get('start_s', 0)):.3f}s–{float(promo.get('end_s', 0)):.3f}s ({float(promo.get('duration_s', 0)):.3f}s)",
                f"- Hook: {promo.get('hook_line', '')}",
                f"- Cliffhanger: {promo.get('cliffhanger_line', '')}",
                f"- Why it works: {promo.get('why_it_works', '')}",
                f"- Scores: hook {h}, cliffhanger {c}, emotion {e}, standalone {s}",
                f"- Weighted score: {float(promo.get('weighted_score', 0)):.4f}",
                "",
            ]
        )
    destination.write_text("\n".join(lines), encoding="utf-8")


def build_export_zip(
    promos_or_raw: list[dict] | str | Path | None = None,
    runs: list[dict] | None = None,
    raw_dir: str | Path = "outputs/raw",
) -> bytes:
    """Create an in-memory zip archive of promos, reports, and raw LLM response JSONs."""
    import io
    import zipfile

    buffer = io.BytesIO()

    # Determine whether first arg is raw_dir or promos list
    promos = None
    target_raw_dir = raw_dir

    if isinstance(promos_or_raw, (str, Path)):
        target_raw_dir = promos_or_raw
    elif isinstance(promos_or_raw, list):
        promos = promos_or_raw

    with zipfile.ZipFile(buffer, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        if promos:
            import pandas as pd
            # promos.csv
            df = pd.DataFrame(promos)
            zf.writestr("promos.csv", df.to_csv(index=False))
            # promos.json
            zf.writestr("promos.json", json.dumps({"promos": promos}, indent=2, ensure_ascii=False))

        source_dir = Path(target_raw_dir)
        if source_dir.exists():
            for json_file in source_dir.glob("*.json"):
                zf.write(json_file, arcname=f"raw/{json_file.name}")

        # If archive is empty, provide a fallback readme
        if not zf.namelist():
            zf.writestr("README.txt", "No export artifacts or raw LLM outputs logged yet.")

    return buffer.getvalue()


def write_all_outputs(
    arg1: list[dict] | str | Path,
    arg2: list[dict] | str | Path = "outputs",
    *args,
    base_dir: str | Path | None = None,
) -> dict[str, str]:
    """Write promos.json, promos.csv, and report.md to disk and return paths."""
    # Allow flexible argument order: (promos, base_dir) or (base_dir, promos)
    if isinstance(arg1, (str, Path)) and isinstance(arg2, list):
        promos = arg2
        out_dir = Path(arg1)
    elif isinstance(arg1, list):
        promos = arg1
        out_dir = Path(base_dir if base_dir is not None else arg2)
    else:
        promos = []
        out_dir = Path(base_dir if base_dir is not None else "outputs")

    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "promos.json"
    csv_path = out_dir / "promos.csv"
    md_path = out_dir / "report.md"

    write_promos_json(str(json_path), promos)
    write_promos_csv(str(csv_path), promos)
    write_report_md(str(md_path), promos)

    return {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "markdown": str(md_path.resolve()),
    }



def archive_raw_llm_response(directory: str, kind: str, clip_id: str, response: dict) -> None:
    """Save an unedited LLM response without modifying it."""
    destination = Path(directory)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / f"{clip_id}-{kind}.json"
    path.write_text(
        json.dumps(response, indent=2, ensure_ascii=False), encoding="utf-8"
    )
