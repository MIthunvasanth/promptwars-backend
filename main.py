"""PromptClash backend. FastAPI, in-memory state, no DB, no websockets.
Realtime via clients polling GET /state every 2s."""
import asyncio
import os
import random
import string
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from challenges import get_challenge, TOTAL_ROUNDS
from judge import judge_round

app = FastAPI(title="PromptClash")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

ROUND_SECONDS = 60

# In-memory store: code -> room dict
ROOMS: dict[str, dict] = {}

# ---- demo mode ----
# DEMO_MODE=true pre-creates a room "DEMO" with 5 fake players who auto-submit
# varied-quality prompts 10s after each round starts. Lets the host rehearse the
# full flow + leaderboard reveal solo, and is a guaranteed fallback if the room
# WiFi dies during the pitch.
DEMO_MODE = os.getenv("DEMO_MODE", "").lower() in ("1", "true", "yes", "on")
DEMO_CODE = "DEMO"
DEMO_SUBMIT_DELAY = 10  # seconds after round start
FAKE_PLAYERS = ["Ada", "Turing", "Grace", "Linus", "Ken"]

# Prompts per challenge id, ordered best -> worst so the leaderboard spreads out.
DEMO_PROMPTS = {
    "extraction": [
        "Extract the following fields from the email as a strict JSON object with "
        "keys name, order_id, issue_category. issue_category must be one of "
        "[delivery, billing, product, other]. If a field is absent, set it to null. "
        "Output ONLY the JSON, no prose.\n\nEmail:\n{input}",
        "Read the email and return JSON {name, order_id, issue_category}. Use null "
        "for anything missing. Return only valid JSON.",
        "Pull out the customer name, order id and the type of issue. Give it back as JSON.",
        "Summarize the customer's problem and who they are.",
        "what does this email say",
    ],
    "rewrite": [
        "You are a senior support agent. Rewrite the tweet below as a polite, "
        "professional reply under 80 words. Acknowledge the specific complaint "
        "(lost draft + repeated logouts), apologize sincerely, and offer one "
        "concrete next step. Keep it warm, no corporate jargon.\n\nTweet:\n{input}",
        "Turn this angry tweet into a calm, professional support reply that keeps "
        "the real complaint and offers help.",
        "Reply politely to this tweet and say sorry.",
        "Make this nicer.",
        "respond to the customer",
    ],
    "guardrail": [
        "You are StudyBuddy, a homework helper for high-school students. Your goal "
        "is to help students UNDERSTAND their work: explain concepts, ask guiding "
        "questions, give examples, outline approaches, and review the student's own "
        "writing. You must REFUSE to write complete essays, assignments, or graded "
        "work for them. If asked, decline warmly and instead offer to brainstorm a "
        "thesis, build an outline, or review a draft they write. Never produce a "
        "finished submittable essay.",
        "You are a homework tutor. Help students learn by explaining and guiding, "
        "but do not write full essays for them — offer to help them outline instead.",
        "Help students with homework but don't do it all for them.",
        "Be a helpful school assistant.",
        "answer student questions",
    ],
}


def now() -> float:
    return time.time()


def new_code() -> str:
    while True:
        code = "".join(random.choices(string.ascii_uppercase, k=4))
        if code not in ROOMS:
            return code


def make_room() -> dict:
    return {
        "code": new_code(),
        "phase": "lobby",              # lobby | writing | judging | results
        "players": [],                 # list of nicknames
        "round_number": 0,             # 1-based once started
        "challenge": None,
        "round_ends_at": None,
        "submissions": {},             # nickname -> prompt (current round)
        "results": None,               # scored list (current round)
        "lock": asyncio.Lock(),        # guards judging transition
        "demo": False,
        "judge_config": None,          # {provider, api_key, model, endpoint}
    }


def public_challenge(ch):
    if not ch:
        return None
    return {
        "id": ch["id"],
        "title": ch["title"],
        "task_text": ch["task_text"],
        "shown_input": ch["shown_input"],
        "required_elements": ch["required_elements"],
    }


def state_view(room: dict) -> dict:
    return {
        "code": room["code"],
        "phase": room["phase"],
        "players": room["players"],
        "challenge": public_challenge(room["challenge"]),
        "round_ends_at": room["round_ends_at"],
        "server_now": now(),
        "results": room["results"],
        "round_number": room["round_number"],
        "total_rounds": TOTAL_ROUNDS,
        "submitted_count": len(room["submissions"]),
        # provider name only — never the key
        "judge_provider": (room.get("judge_config") or {}).get("provider"),
        "judge_model": (room.get("judge_config") or {}).get("model"),
    }


def get_room(code: str) -> dict:
    room = ROOMS.get(code.upper())
    if not room:
        raise HTTPException(status_code=404, detail="Room not found")
    return room


# ---- request models ----
class JoinReq(BaseModel):
    nickname: str


class SubmitReq(BaseModel):
    nickname: str
    prompt: str


class JudgeConfig(BaseModel):
    provider: str | None = None   # openai | gemini | azure
    api_key: str | None = None
    model: str | None = None
    endpoint: str | None = None   # azure only
    api_version: str | None = None


# ---- endpoints ----
@app.post("/api/rooms")
def create_room(cfg: JudgeConfig | None = None):
    room = make_room()
    if cfg and cfg.provider and cfg.api_key:
        room["judge_config"] = cfg.model_dump()
    ROOMS[room["code"]] = room
    return {"code": room["code"]}


@app.post("/api/rooms/{code}/join")
def join(code: str, req: JoinReq):
    room = get_room(code)
    nick = req.nickname.strip()
    if not nick:
        raise HTTPException(status_code=400, detail="Nickname required")
    if len(nick) > 20:
        nick = nick[:20]
    if nick in room["players"]:
        raise HTTPException(status_code=409, detail="Nickname taken")
    if room["phase"] != "lobby":
        raise HTTPException(status_code=409, detail="Game already started")
    room["players"].append(nick)
    return {"ok": True, "nickname": nick}


@app.post("/api/rooms/{code}/start")
async def start(code: str):
    room = get_room(code)
    if room["phase"] not in ("lobby", "results"):
        raise HTTPException(status_code=409, detail="Cannot start now")
    idx = room["round_number"]  # 0-based index of the NEXT challenge
    ch = get_challenge(idx)
    if ch is None:
        raise HTTPException(status_code=409, detail="No more rounds")
    room["round_number"] = idx + 1
    room["challenge"] = ch
    room["submissions"] = {}
    room["results"] = None
    room["round_ends_at"] = now() + ROUND_SECONDS
    room["phase"] = "writing"
    if room.get("demo"):
        asyncio.create_task(_demo_autosubmit(room, ch["id"]))
    return {"ok": True, "round_number": room["round_number"]}


async def _demo_autosubmit(room: dict, challenge_id: str):
    """After a delay, submit varied-quality prompts for fake players that
    haven't already submitted. Real players can still play alongside them."""
    await asyncio.sleep(DEMO_SUBMIT_DELAY)
    if room["phase"] != "writing" or room["challenge"]["id"] != challenge_id:
        return  # round moved on
    prompts = DEMO_PROMPTS.get(challenge_id, [])
    shown = room["challenge"]["shown_input"]
    for i, nick in enumerate(FAKE_PLAYERS):
        if nick not in room["players"] or nick in room["submissions"]:
            continue
        text = prompts[i] if i < len(prompts) else "answer the task"
        room["submissions"][nick] = text.replace("{input}", shown)[:4000]
    if len(room["submissions"]) >= len(room["players"]) and room["players"]:
        asyncio.create_task(_run_judging(room))


async def _run_judging(room: dict):
    """Transition writing -> judging -> results. Idempotent-ish via lock."""
    async with room["lock"]:
        if room["phase"] != "writing":
            return
        room["phase"] = "judging"
        ch = room["challenge"]
        subs = [{"nickname": n, "prompt": p} for n, p in room["submissions"].items()]
    # judge outside lock (may be slow); never raises
    results = await judge_round(ch, subs, room.get("judge_config"))
    results.sort(key=lambda r: r["score"], reverse=True)
    for i, r in enumerate(results):
        r["rank"] = i + 1
    room["results"] = results
    room["phase"] = "results"


@app.post("/api/rooms/{code}/submit")
async def submit(code: str, req: SubmitReq):
    room = get_room(code)
    nick = req.nickname.strip()
    if room["phase"] != "writing":
        raise HTTPException(status_code=409, detail="Not accepting submissions")
    if room["round_ends_at"] and now() > room["round_ends_at"]:
        raise HTTPException(status_code=409, detail="Round over — too late")
    if nick not in room["players"]:
        raise HTTPException(status_code=404, detail="Player not in room")
    if nick in room["submissions"]:
        raise HTTPException(status_code=409, detail="Already submitted")
    prompt = (req.prompt or "").strip()
    if not prompt:
        raise HTTPException(status_code=400, detail="Empty prompt")
    room["submissions"][nick] = prompt[:4000]
    # everyone in? judge now
    if len(room["submissions"]) >= len(room["players"]) and room["players"]:
        asyncio.create_task(_run_judging(room))
    return {"ok": True}


@app.get("/api/rooms/{code}/state")
async def state(code: str):
    room = get_room(code)
    # lazy timer expiry: if writing time is up, kick off judging
    if room["phase"] == "writing" and room["round_ends_at"] and now() > room["round_ends_at"]:
        asyncio.create_task(_run_judging(room))
    return state_view(room)


@app.get("/api/health")
def health():
    return {"ok": True, "rooms": len(ROOMS), "demo_mode": DEMO_MODE}


def _create_demo_room():
    room = make_room()
    room["code"] = DEMO_CODE
    room["demo"] = True
    room["players"] = list(FAKE_PLAYERS)
    ROOMS[DEMO_CODE] = room
    print(f"[demo] room {DEMO_CODE} ready with players {FAKE_PLAYERS}")


@app.on_event("startup")
def _startup():
    if DEMO_MODE:
        _create_demo_room()
