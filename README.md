# 1. Title and Summary

Promo Finder discovers 20-second to 60-second promotional clip candidates from timestamped microdrama screenplays.
Deterministic code handles parsing, candidate window slicing, pre-filtering, and ranking; the large language model (LLM) only evaluates dialogue hooks, cliffhangers, and standalone understandability.
The LLM never writes or modifies timestamps; every start and end time is bound to exact dialogue event timestamps in the source screenplay.

## 2. Quick Start

1. Clone or download the repository and open the directory:
```bash
git clone https://github.com/himanshu9330/promo-selecter.git
cd "clip trimming"
```

2. Create a virtual environment:
```bash
python -m venv .venv          # Windows
python3 -m venv .venv         # macOS / Linux
```

3. Activate the virtual environment:
```bash
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # macOS / Linux
```

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Configure credentials:
```bash
copy .env.example .env        # Windows
cp .env.example .env          # macOS / Linux
```
*No API key?* Leave `.env` blank. The tool defaults to Rules-only mode, which runs locally without network calls. If using an LLM, open `.env` and set `OPENROUTER_API_KEY` or `GEMINI_API_KEY`.

6. Place screenplay files:
Place `.txt` screenplay files in the `data/` directory.

7. Run the web application:
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

8. Run the command-line interface:
```bash
python src/main.py "data/I Married a Broken Soldier. He was a Duke/episode-001-Episode 1671703.txt"
```
Expected output: The pipeline parses the screenplay, scores candidate windows, prints 5 ranked promo cuts, and writes outputs to `outputs/`.

## 3. What It Produces

Sample output format (using placeholder text, not confidential script data):
```
Rank: 1
Episode: 1
Clip ID: ep01_w012
Segment: 00:45.0 - 01:18.5 (33.5s)
Hook Line: "ALEX (Shouting): You knew the contract was void before we arrived!"
Cliffhanger Line: "ELENA (Whisper): There is one secret you have not uncovered..."
Why It Works: Immediate accusation opening followed by an unresolved threat.
Scores: Hook: 9/10, Cliffhanger: 9/10, Emotion: 8/10, Standalone: 8/10
Weighted Score: 0.86
```

Files written to `outputs/`:
- `outputs/promos.json`: Machine-readable array of ranked promo objects with dialogue events and scores.
- `outputs/promos.csv`: Flat table containing segment timestamps, hook lines, cliffhangers, and component scores.
- `outputs/report.md`: Markdown summary formatted for editorial review.
- `outputs/editor_feedback.csv`: Log of human accept, reject, and trim adjustments.
- `outputs/raw/`: Raw LLM response JSON files saved when running in Rules + LLM mode.

## 4. How It Works

Pipeline stages:
1. Parse Screenplay (Code): Extracts episode metadata, timestamps, dialogue lines, and visual cues.
2. Cut Windows (Code): Generates all combinations of dialogue start and end events between 20s and 60s.
3. Pre-Filter (Code): Scores windows on tone markers, punctuation, and pace; selects top diverse candidates using non-maximum suppression.
4. LLM Scoring (LLM): Evaluates candidate hook, cliffhanger, and emotional intensity on a 1-10 scale.
5. Standalone Blind Test (LLM): Tests clip understandability without series logline or continuity notes.
6. Validate Scores (Code): Rejects scores outside 1-10, hallucinated quotes, or mismatched clip IDs.
7. Rank and Deduplicate (Code): Calculates weighted composite scores and eliminates overlapping cuts (IoU > 0.45).
8. Report Generation (Code): Writes JSON, CSV, Markdown, and ZIP exports.
9. Editorial Review (Human + Code): Human editor accepts, rejects, or trims boundaries snapping to dialogue lines.

```
[Screenplay .txt] -> [Parse Events] -> [Cut 20-60s Windows] -> [Pre-Filter & NMS]
                                                                      |
                   +--------------------------------------------------+
                   | (Rules + LLM Mode)               | (Rules-Only Mode)
                   v                                  v
          [LLM Scoring & Blind Test]          [Deterministic Scoring]
                   |                                  |
                   +----------------+-----------------+
                                    v
                        [Validate Quote Integrity]
                                    v
                      [IoU Deduplication & Ranking]
                                    v
                        [outputs/ & Streamlit UI]
```

Timestamps originate exclusively in code from screenplay timestamps. The LLM receives text and returns scores and quotes; it never calculates or alters boundary numbers.

## 5. Choosing and Swapping the Model

Model configuration is set in `config.yaml`:
```yaml
llm:
  provider: openrouter
  model: nvidia/nemotron-3-ultra-550b-a55b
```

To switch providers:
- OpenRouter: Set `provider: openrouter` and `model: <model-identifier>`. Set `OPENROUTER_API_KEY` in `.env`.
- Gemini: Set `provider: gemini` and `model: gemini-2.5-flash` (or `gemini-2.5-pro`). Set `GEMINI_API_KEY` in `.env`.
- Ollama (Local): Set `provider: ollama` and `model: <local-model>`. Set `LLM_BASE_URL: http://localhost:11434/v1` in `.env`.

All LLM calls pass through the single `llm()` function in `src/llm.py`.
Responses are cached locally on disk in `.cache/llm/` using a SHA-256 hash of the provider, model, and prompt. Repeated runs on the same input make zero external API calls.

## 6. Using the App

The Streamlit app (`app.py`) provides five primary sections:
- Sidebar Controls: Upload `.txt` screenplays, paste script text, switch between Rules-only and Rules + LLM modes, set duration bounds, and adjust ranking weights.
- Top Pick Card: Displays the rank 1 recommendation with exact opening/closing quotes, component scores, storyboard visual cues, optional video preview, and Accept/Reject/Trim buttons.
- Candidate Explorer: Interactive table of all candidate windows with score progress bars and filtering.
- Timeline View: Visual mapping of dialogue density across the episode with promo block overlays.
- Quality Tab: Diagnostic candidate funnel tracking rejected windows, score distribution, and model comparison.
- Export Tab: Download buttons for `promos.csv`, `promos.json`, `report.md`, and an all-in-one ZIP archive.

To analyze a new script, paste it into the "Paste a script" sidebar expander or upload the file, then click "Find promos".

## 7. Results and Quality Checks

Measurements from real runs on the input dataset:

| Episode File | Raw Windows | After Length (20-60s) | After Hook Filter | After Cliff Filter | Promos Returned | Validator Rejections | LLM Calls | Cache Hits | Runtime | Source File |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| `episode-001` | 663 | 471 | 471 | 336 | 5 | 0 | 0 (Rules) | 0 | 18.2s | `outputs/promos.json` |
| `episode-010` | 343 | 242 | 242 | 131 | 2 | 0 | 0 (Rules) | 0 | 31.3s | `outputs/history/ep10-*.json` |
| `episode-048` | 366 | 260 | 260 | 141 | 3 | 0 | 0 (Rules) | 0 | 2.3s | `outputs/history/ep48-*.json` |
| `episode-049` | 345 | 272 | 272 | 147 | 4 | 0 | 0 (Rules) | 0 | 1.6s | `outputs/history/ep49-*.json` |
| `episode-050` | 159 | 118 | 118 | 63 | 2 | 0 | 0 (Rules) | 0 | 1.4s | `outputs/history/ep50-*.json` |

Manual review of output quality: manual review: not done.

## 8. Known Limitations and Failures

1. Text-only limitation: The pipeline evaluates written screenplays and cannot detect visual acting performances, cuts, actor chemistry, or audio stingers.
2. Action line exclusion at boundaries: Windows are anchored to dialogue events. If a dramatic slap or visual reveal precedes dialogue by 2 seconds in an action cue, the window starts on the spoken line unless trimmed manually.
3. Cold opens lacking context: In `episode-010`, candidate `ep10_w082` was scored high for dramatic confrontation keywords, but begins mid-dialogue on an unresolved pronoun reference.
4. Overlapping scene competition: In `episode-001`, candidates `ep01_w206` (63.4s - 119.0s) and `ep01_w208` (66.5s - 88.9s) share the same scene dialogue; IoU filtering limits but does not fully eliminate shared emotional beats across nearby ranks.
5. JSON parsing variance: Smaller local models (e.g., 7B-parameter models via Ollama) occasionally format nested JSON incorrectly; the validator retries once before dropping the clip.

## 9. Assumptions

1. Target promotional cut duration is between 20.0 and 60.0 seconds.
2. Dialogue start and end points represent natural cutting boundaries for video editors.
3. Screenplays follow the standard timestamped format `Start: (mm:ss.s) | End: (mm:ss.s)`.
4. Up to 5 non-overlapping promotional candidates per episode are sufficient for editorial selection.
5. When no LLM API key is present, offline rules-based scoring provides a reliable baseline without failing.

## 10. Scaling and Cost

- Candidate reduction: Prefiltering reduces hundreds of raw windows to 12 candidates per episode.
- API volume: At 12 candidates per episode, full LLM scoring requires 24 requests per episode (1 scoring call + 1 blind standalone test per candidate).
- Caching: Repeated runs or parameter tweaks on the same episode cost $0 and make 0 API requests due to disk caching in `.cache/llm/`.
- Cost per episode: not measured.
- Human in the loop: Editors use the trim and decision controls in the UI; adjustments persist to `outputs/editor_feedback.csv` to guide downstream video assembly.

## 11. Project Layout

```
.
|-- .env.example              Configuration template for API credentials
|-- .gitignore                Ignored paths (data/, .env, .cache/)
|-- README.md                 Project documentation and reproduction guide
|-- app.py                    Streamlit entry point (< 100 lines)
|-- config.yaml               Default provider, model, duration, and ranking weights
|-- requirements.txt          Python package dependencies
|-- outputs/                  Export directory (CSV, JSON, Markdown, feedback)
|-- src/
|   |-- app_core.py           Data structures, feedback storage, boundary snapping
|   |-- candidates.py         Exhaustive 20-60s window generator and candidate funnel
|   |-- llm.py                Single LLM gateway with provider routing and disk caching
|   |-- main.py               CLI entry point for running pipeline on single scripts
|   |-- parse.py              Screenplay parser and timestamp conversion
|   |-- pipeline.py           End-to-end orchestration, step callbacks, error handling
|   |-- prefilter.py          Deterministic scoring, tone keyword matching, NMS selection
|   |-- rank.py               IoU overlap deduplication, weight normalization, reranking
|   |-- report.py             JSON, CSV, Markdown, and in-memory ZIP export serialization
|   |-- score.py              LLM prompt construction, structured scoring, blind test
|   |-- validate.py           Quote verification and boundary integrity validation
|-- tests/
|   |-- test_parse.py         Unit tests for screenplay parser and timestamps
|   |-- test_pipeline.py      Pipeline error handling, stop conditions, rules mode
|   |-- test_report.py        Timestamp formatting and export file generation
|   |-- test_sidebar_helpers.py Re-ranking, connection testing, weight normalization
|   |-- test_skeleton.py      Smoke tests
|   |-- test_validate.py      LLM output validation and rejection logic
|-- ui/
|   |-- components.py         Cards, tables, empty states, error alerts, trim controls
|   |-- header.py             Header bar with quiet status indicators and disclaimer
|   |-- results.py            Results section rendering Top Pick, metrics, and cards
|   |-- sidebar.py            Collapsible sidebar sections and file ingestion
|   |-- styles.py             Color palette and theme constants
|   |-- tabs.py               All Candidates, Timeline, Quality, and Export views
```

## 12. Tests

Run the full automated test suite using `pytest`:
```bash
pytest tests/ -v
```
Test coverage includes:
- Screenplay parsing, header detection, and millisecond timestamp alignment (`test_parse.py`).
- Pipeline execution, missing file handling, and stop conditions (`test_pipeline.py`).
- Quote integrity, duration limits, and hallucination rejection (`test_validate.py`).
- Output generation, CSV serialization, and ZIP archiving (`test_report.py`).
- Interactive re-ranking with custom weights and weight normalization (`test_sidebar_helpers.py`).

Result: 19 passed in ~1.0s.

## 13. Data and Confidentiality

The screenplay text files provided for this evaluation are confidential proprietary content of Kuku.
Screenplays are not committed to this repository. The `data/` directory, the `.cache/` folder, and local `.env` credential files are explicitly ignored in `.gitignore`.
No screenplay text or dialogues are stored in the documentation, source files, or test fixtures.

## 14. AI Tools Used

This project was built using Google Antigravity with Gemini 2.5 Pro as an engineering assistant.
One specific failure encountered during development:
In the initial candidate generation stage, the AI assistant implemented a greedy selection without non-maximum suppression (NMS). This caused 12 contiguous 1-second shifts of the same emotional scene (46s to 88s) to crowd out the candidate pool. The overlap filter then discarded the remaining 11 windows, leaving only 1 promo for the entire episode. This was diagnosed and resolved by adding NMS window selection in `src/prefilter.py` and updating overlap pruning to use Intersection-over-Union (IoU <= 0.45) in `src/rank.py`, restoring 5 diverse promo cuts per episode.
