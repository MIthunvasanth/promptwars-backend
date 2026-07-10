"""LLM judge for PromptClash rounds. Pluggable provider (OpenAI / Gemini /
Azure OpenAI) via httpx, with a deterministic heuristic fallback so a round can
never hang or crash. Provider + key + model are chosen per-room in the UI."""
import json
import os
import re

import httpx

# Judge config comes per-room from the frontend (provider + api_key + model).
# If none is supplied, we fall back to env vars, then to the heuristic scorer.
# Any failure / unparseable response also falls back to the heuristic — a round
# can never hang or crash.
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
AZURE_OPENAI_DEPLOYMENT = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o-mini")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Default model per provider (used when the frontend doesn't send one).
DEFAULT_MODELS = {
    "openai": "gpt-4o-mini",
    "gemini": "gemini-2.0-flash",
    "azure": "gpt-4o-mini",  # Azure deployment name
}

RUBRIC = (
    "Score each submission 0-100 using this rubric (25 points each):\n"
    "- clarity: is the prompt clear and unambiguous?\n"
    "- specificity: is it concrete and detailed?\n"
    "- constraint coverage: does it cover the task's required elements?\n"
    "- robustness: does it handle edge cases / missing data / misuse?"
)


def _build_prompt(challenge, submissions):
    lines = [
        "You are judging a prompt-engineering battle. Players wrote prompts "
        "for the task below.",
        "",
        f"TASK: {challenge['task_text']}",
        f"INPUT SHOWN TO PLAYERS: {challenge['shown_input']}",
        f"REQUIRED ELEMENTS: {', '.join(challenge['required_elements'])}",
        "",
        RUBRIC,
        "",
        "SUBMISSIONS:",
    ]
    for i, s in enumerate(submissions):
        lines.append(f"[{i}] nickname={s['nickname']!r}: {s['prompt']}")
    lines.append("")
    lines.append(
        'Return ONLY a strict JSON object of the form {"results": [...]}, with '
        "one entry per submission in the same order. Each entry: "
        '{"nickname": str, "score": int 0-100, "feedback": str <=20 words}. '
        "No prose, no code fences."
    )
    return "\n".join(lines)


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _parse(text: str, submissions):
    data = json.loads(_strip_fences(text))
    if isinstance(data, dict):
        data = data.get("results", data.get("scores", []))
    if not isinstance(data, list):
        raise ValueError("not a list")
    by_nick = {}
    for item in data:
        nick = item.get("nickname")
        score = int(item.get("score", 0))
        score = max(0, min(100, score))
        fb = str(item.get("feedback", ""))[:200]
        if nick is not None:
            by_nick[nick] = {"nickname": nick, "score": score, "feedback": fb}
    # ensure every submission gets a result (fall back per-missing)
    results = []
    for s in submissions:
        if s["nickname"] in by_nick:
            results.append(by_nick[s["nickname"]])
        else:
            results.append(_heuristic_one(None, s))  # single fallback
    return results


def _heuristic_one(challenge, sub):
    """Deterministic score for one submission."""
    prompt = (sub.get("prompt") or "").lower()
    score = 0
    hits = []
    required = (challenge or {}).get("required_elements", [])
    if required:
        # coverage: keyword tokens from each required element
        per = 60 / len(required)
        for elem in required:
            tokens = [t for t in re.split(r"[^a-z0-9]+", elem.lower()) if len(t) > 2]
            if any(t in prompt for t in tokens):
                score += per
                hits.append(elem)
    # length band (0-25)
    words = len(prompt.split())
    if 30 <= words <= 200:
        score += 25
    elif 15 <= words < 30 or 200 < words <= 300:
        score += 15
    elif words >= 5:
        score += 8
    # structure bonus (0-15): line breaks, numbering, or format cues
    if re.search(r"\n|[-*]\s|\d\.\s|:", prompt):
        score += 10
    if "json" in prompt or "format" in prompt or "example" in prompt:
        score += 5
    score = int(max(0, min(100, round(score))))
    if hits:
        fb = f"Covered: {', '.join(h.split('(')[0].strip() for h in hits[:2])}."
    else:
        fb = "Add the task's required elements and be more specific."
    return {"nickname": sub["nickname"], "score": score, "feedback": fb[:120]}


def _heuristic(challenge, submissions):
    return [_heuristic_one(challenge, s) for s in submissions]


SYSTEM_MSG = (
    "You are a strict prompt-engineering judge. Respond with a JSON object only."
)


def _resolve_config(config):
    """Merge per-room config with env fallbacks. Returns a normalized dict with
    a valid provider+key, or None to signal 'use the heuristic scorer'."""
    config = config or {}
    provider = (config.get("provider") or "").lower()
    key = (config.get("api_key") or "").strip()
    model = (config.get("model") or "").strip()
    endpoint = (config.get("endpoint") or "").strip().rstrip("/")

    if provider in ("openai", "gemini", "azure") and key:
        if provider == "azure" and not endpoint:
            return None  # azure needs an endpoint
        return {
            "provider": provider,
            "api_key": key,
            "model": model or DEFAULT_MODELS[provider],
            "endpoint": endpoint,
            "api_version": config.get("api_version") or AZURE_OPENAI_API_VERSION,
        }

    # env fallbacks
    if OPENAI_API_KEY:
        return {"provider": "openai", "api_key": OPENAI_API_KEY,
                "model": DEFAULT_MODELS["openai"], "endpoint": ""}
    if GEMINI_API_KEY:
        return {"provider": "gemini", "api_key": GEMINI_API_KEY,
                "model": DEFAULT_MODELS["gemini"], "endpoint": ""}
    if AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT:
        return {"provider": "azure", "api_key": AZURE_OPENAI_API_KEY,
                "model": AZURE_OPENAI_DEPLOYMENT, "endpoint": AZURE_OPENAI_ENDPOINT,
                "api_version": AZURE_OPENAI_API_VERSION}
    return None


async def _call_openai(cfg, prompt, client):
    resp = await client.post(
        "https://api.openai.com/v1/chat/completions",
        headers={"Authorization": f"Bearer {cfg['api_key']}"},
        json={
            "model": cfg["model"],
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _call_azure(cfg, prompt, client):
    url = (
        f"{cfg['endpoint']}/openai/deployments/{cfg['model']}"
        f"/chat/completions?api-version={cfg['api_version']}"
    )
    resp = await client.post(
        url,
        headers={"api-key": cfg["api_key"]},
        json={
            "messages": [
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        },
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


async def _call_gemini(cfg, prompt, client):
    url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/"
        f"{cfg['model']}:generateContent?key={cfg['api_key']}"
    )
    resp = await client.post(
        url,
        json={
            "contents": [{"parts": [{"text": f"{SYSTEM_MSG}\n\n{prompt}"}]}],
            "generationConfig": {
                "temperature": 0.2,
                "response_mime_type": "application/json",
            },
        },
    )
    resp.raise_for_status()
    return resp.json()["candidates"][0]["content"]["parts"][0]["text"]


_CALLERS = {"openai": _call_openai, "azure": _call_azure, "gemini": _call_gemini}


async def judge_round(challenge, submissions, config=None):
    """Return [{nickname, score, feedback}] for all submissions.
    `config` = per-room {provider, api_key, model, endpoint}. Never raises —
    falls back to the heuristic scorer on any failure."""
    if not submissions:
        return []
    cfg = _resolve_config(config)
    if not cfg:
        return _heuristic(challenge, submissions)

    prompt = _build_prompt(challenge, submissions)
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            text = await _CALLERS[cfg["provider"]](cfg, prompt, client)
        return _parse(text, submissions)
    except Exception as e:  # noqa: BLE001 — judging must never crash the round
        print(f"[judge] {cfg['provider']} failed ({e!r}), using heuristic fallback")
        return _heuristic(challenge, submissions)
