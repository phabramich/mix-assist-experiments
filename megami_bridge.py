"""Вызов MEGAMI (отдельный интерпретатор с ROCm) из оркестратора."""
import json
import os
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MEGAMI = ROOT / "megami"
PY = Path(os.environ.get(
    "MEGAMI_PYTHON",
    Path.home() / ".unsloth/studio/unsloth_studio/Scripts/python.exe"))
FS = 44100
SEC_PER_AUDIO_SEC = 2.5


def _run(*args):
    env = {**os.environ, "PYTHONPATH": str(MEGAMI / ".pydeps"),
           "MEGAMI_INFERENCE": "1", "PYTHONUNBUFFERED": "1"}
    subprocess.run([str(PY), str(MEGAMI / "job.py"), *map(str, args)],
                   cwd=MEGAMI, env=env, check=True)


def bank(session, n=8):
    out = Path(session) / "megami" / "bank.pt"
    out.parent.mkdir(parents=True, exist_ok=True)
    _run("bank", "--dry", Path(session) / "dry", "--n", n, "--out", out)


def bank_meta(session):
    p = Path(session) / "megami" / "bank.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def wet_path(session, stem, z, t0, t1):
    return Path(session) / "megami" / "wet" / stem / f"z{z}_{t0:.2f}_{t1:.2f}.wav"


def missing(session, items):
    return [it for it in items if not wet_path(session, *it).exists()]


def ensure_wet(session, items):
    todo = missing(session, items)
    if todo:
        jobs = [{"stem": s, "z": z, "start": t0, "end": t1,
                 "out": str(wet_path(session, s, z, t0, t1))} for s, z, t0, t1 in todo]
        jp = Path(session) / "megami" / "jobs" / f"render-{int(time.time())}.json"
        jp.parent.mkdir(parents=True, exist_ok=True)
        jp.write_text(json.dumps(jobs), encoding="utf-8")
        _run("render", "--dry", Path(session) / "dry",
             "--bank", Path(session) / "megami" / "bank.pt", "--jobs", jp)
    return {it: wet_path(session, *it) for it in items}


def estimate_min(items):
    return round(sum(t1 - t0 for _, _, t0, t1 in items) * SEC_PER_AUDIO_SEC / 60, 1)
