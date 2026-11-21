"""Parser tests using a representative microdrama screenplay."""

from src.parse import parse_episode, parse_timestamp, parse_screenplay_text


def test_parse_screenplay_text() -> None:
    result = parse_screenplay_text(SCREENPLAY, "memory_script.txt")
    assert result["episode_number"] == "1"
    assert len(result["events"]) == 3
    assert result["events"][1]["speaker"] == "MARA (O.S.)"


SCREENPLAY = """EPISODE NUMBER: 1
EPISODE LOGLINE: A stolen key changes everything.
CONTINUITY NOTES: Keep the red coat.
--- SCRIPT START ---
SCENE 01
Scene heading: Exterior - Night
Scene start: 00:10.0
Scene end: 00:40.0
**ACTION:**
Start: (00:00.0) | End: (00:03.0)
Mara runs toward the gate.
**MARA (O.S.) (Whisper):**
Start: (00:03.0) | End: (00:08.0)
"I have the key."
**CAMERA:**
Start: (00:08.0) | End: (00:12.0)
Close-up on Mara's shaking hand.
"""


def test_parse_episode(tmp_path) -> None:
    screenplay = tmp_path / "episode.txt"
    screenplay.write_text(SCREENPLAY, encoding="utf-8")

    result = parse_episode(str(screenplay))

    assert result["episode_number"] == "1"
    assert result["logline"] == "A stolen key changes everything."
    assert result["continuity_notes"] == "Keep the red coat."
    assert [event["type"] for event in result["events"]] == ["action", "dialogue", "camera"]
    assert result["events"][1]["speaker"] == "MARA (O.S.)"
    assert result["events"][1]["tone"] == "Whisper"
    assert result["events"][1]["text"] == "I have the key."


def test_parse_timestamp() -> None:
    assert parse_timestamp("01:02.5") == 62.5
    assert parse_timestamp("1:02:03.25") == 3723.25


def test_real_episode_1():
    episode_path = "data/I Married a Broken Soldier. He was a Duke/episode-001-Episode 1671703.txt"
    result = parse_episode(episode_path)

    assert len(result["events"]) == 95

    matriarch = next(
        event for event in result["events"]
        if event["type"] == "dialogue"
        and event["speaker"] == "MATRIARCH"
        and event["text"] == "You should turn around, Juliana."
    )
    assert matriarch["start_s"] == 11.3
    assert matriarch["end_s"] == 13.1


def test_real_episode_1_windows():
    from src.candidates import generate_candidates

    episode_path = "data/I Married a Broken Soldier. He was a Duke/episode-001-Episode 1671703.txt"
    episode = parse_episode(episode_path)
    windows = generate_candidates(episode)

    valid = [window for window in windows if 20 <= window["duration_s"] <= 60]
    assert len(valid) >= 5
    assert any(window["start_s"] == 11.3 for window in valid)
