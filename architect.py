"""Архитектор: MF слушает микс -> JSON-задача для Jev-контроллера.

Спека задачи узкая, под текущий словарь ручек (render.DEFAULT_KNOBS):
  {"track": vocals|drums|bass|other,
   "knob":  gain_db|lowmid_db|pres_db|shelf_db,
   "direction": up|down,
   "reason": str}
"""
import base64
import json
import re
import urllib.request

import judge

TRACKS = ["vocals", "drums", "bass", "other"]
KNOBS = ["gain_db", "lowmid_db", "pres_db", "shelf_db", "hpf_hz"]

PROMPT = """You are a mixing engineer listening to a 7-second fragment of a song mix (stems: vocals, drums, bass, other).
Find the single most important fixable problem with the balance or EQ.
Answer with ONLY a JSON object on one line, no prose:
{"track": one of vocals/drums/bass/other, "knob": one of gain_db/lowmid_db/pres_db/shelf_db, "direction": "up" or "down", "reason": short phrase}
Rules: gain_db = track loudness, lowmid_db = 150-600Hz mud/body, pres_db = 1.5-5kHz presence/harshness, shelf_db = brightness above 5kHz.
If the mix sounds fine, answer {"track": "none"}.
JSON:"""


def ask_mf(server, clip, prompt, fs=16000, timeout=300, max_new_tokens=200):
    body = {"audio_base64": base64.b64encode(judge.wav_bytes(clip, fs)).decode(),
            "prompt": prompt, "max_new_tokens": max_new_tokens}
    req = urllib.request.Request(
        server.rstrip("/"),
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())["text"]


FUSE_PROMPT = """You are the architect of a mixing-assistant system. You get:
1) METER REPORT — objective measurements of the mix and stems.
2) ARTISTIC NOTES — free-form observations from a music-understanding model (hypotheses, may be wrong; validate against meters).
The stems are: vocals, drums, bass, other. Available knobs per stem:
gain_db (level), lowmid_db (150-600Hz), pres_db (1.5-5kHz), shelf_db (>5kHz), hpf_hz (high-pass).
Decide the single most important correction. Prefer problems confirmed by BOTH sources or by meters alone; treat artistic notes that contradict meters as unreliable.
Answer with ONLY JSON on one line:
{"track": vocals|drums|bass|other|none, "knob": gain_db|lowmid_db|pres_db|shelf_db|hpf_hz, "direction": "up"|"down", "reason": short}
METER REPORT:
{meters}
ARTISTIC NOTES:
{notes}"""


def helium_key():
    import os
    a = json.load(open(os.path.expanduser("~/.local/share/opencode/auth.json")))
    return a["helium"]["key"]


def ask_llm(prompt, model="antigravity/claude-sonnet-4-6", timeout=120):
    body = {"model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 4096, "temperature": 0.2}
    req = urllib.request.Request(
        "https://router.y7.hk/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": "Bearer " + helium_key()})
    r = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    return r["choices"][0]["message"]["content"]


def fuse(meter_text, artistic_notes, model="antigravity/claude-sonnet-4-6"):
    prompt = FUSE_PROMPT.replace("{meters}", meter_text) \
                        .replace("{notes}", artistic_notes)
    return parse_task(ask_llm(prompt, model=model))


def parse_task(text):
    """Извлекает JSON-объект из ответа MF; None если не распарсилось."""
    m = re.search(r"\{[^{}]*\}", text)
    if not m:
        return None
    try:
        t = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if t.get("track") == "none":
        return {"track": "none", "raw": text}
    if t.get("track") in TRACKS and t.get("knob") in KNOBS and \
            t.get("direction") in ("up", "down"):
        t["raw"] = text
        return t
    return None
