"""Single gateway for all LLM calls and disk-cached responses."""

import hashlib
import json
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
CACHE_DIR = ROOT / ".cache" / "llm"


def _load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _cache_path(model: str, prompt: str) -> Path:
    key = hashlib.sha256(f"{model}\n{prompt}".encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{key}.json"


def _read_cache(path: Path) -> dict | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _write_cache(path: Path, value: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _gemini_call(model: str, prompt: str, schema: dict) -> dict:
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError("Gemini support requires: google-genai") from exc

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY (or GOOGLE_API_KEY) for Gemini")

    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_json_schema=schema,
            temperature=0,
        ),
    )
    return json.loads(response.text)


def _openai_compatible_call(model: str, prompt: str, schema: dict) -> dict:
    """OpenAI-compatible endpoint, including OpenRouter, NVIDIA NIM, and Groq."""
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("OpenAI-compatible support requires: openai") from exc

    api_key = (
        os.getenv("OPENROUTER_API_KEY")
        or os.getenv("LLM_API_KEY")
        or os.getenv("NVIDIA_API_KEY")
    )
    base_url = os.getenv("LLM_BASE_URL") or "https://openrouter.ai/api/v1"
    if not api_key:
        raise RuntimeError("Set OPENROUTER_API_KEY, LLM_API_KEY, or NVIDIA_API_KEY")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0,
    )
    content = response.choices[0].message.content
    return json.loads(content)


def llm(prompt: str, schema: dict, provider: str | None = None,
        model: str | None = None, api_key: str | None = None,
        base_url: str | None = None) -> dict:
    """Call exactly one provider and cache by provider, model, and prompt."""
    load_dotenv(ROOT / ".env")
    config = _load_config()
    provider = (provider or str(config["llm"]["provider"])).lower()
    model = model or str(config["llm"]["model"])

    cache_file = _cache_path(f"{provider}:{model}", prompt)
    cached = _read_cache(cache_file)
    if cached is not None:
        return {**cached, "__cache_hit__": True}

    if provider == "gemini":
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
        result = _gemini_call(model, prompt, schema)
    elif provider in {"openrouter", "openai-compatible", "nvidia", "groq", "ollama"}:
        if api_key:
            os.environ["OPENROUTER_API_KEY"] = api_key
        if base_url:
            os.environ["LLM_BASE_URL"] = base_url
        result = _openai_compatible_call(model, prompt, schema)
    else:
        raise RuntimeError(f"Unsupported LLM provider: {provider}")

    _write_cache(cache_file, result)
    return {**result, "__cache_hit__": False}


def check_llm_connection(provider: str, model: str, api_key: str | None = None,
                         base_url: str | None = None) -> tuple[bool, str]:
    """Test LLM connectivity and credentials with a minimal structured call."""
    try:
        prompt = "Reply with exactly this JSON object: {\"ok\": true}"
        schema = {"type": "object", "properties": {"ok": {"type": "boolean"}}, "required": ["ok"]}
        res = llm(prompt, schema, provider=provider, model=model, api_key=api_key, base_url=base_url)
        if isinstance(res, dict) and (res.get("ok") is True or "__cache_hit__" in res):
            return True, "Connected successfully"
        return True, "Connected and received response"
    except Exception as exc:
        return False, str(exc)


