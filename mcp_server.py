"""MCP-сервер mix-assist: те же инструменты, что и mixctl.py (stdio).

Каждый tool = вызов `.venv/bin/python mixctl.py ...`, вывод = JSON-строка.
Запуск: .venv/bin/python mcp_server.py
"""
import json
import subprocess

from mcp.server.fastmcp import FastMCP

PY = "/Users/phabramich/dev/mix-assist/.venv/bin/python"
CWD = "/Users/phabramich/dev/mix-assist"
mcp = FastMCP("mix-assist")


def run(*args):
    r = subprocess.run([PY, "mixctl.py", *args], cwd=CWD,
                       capture_output=True, text=True, timeout=900)
    return r.stdout.strip() or r.stderr.strip()[:2000]


@mcp.tool()
def list_sessions() -> str:
    """Список доступных мультитрек-сессий (индексы и названия)."""
    return run("tracks")


@mcp.tool()
def measure(track: int, knobs_json: str = "{}") -> str:
    """Метры микса: LUFS по стемам, полосовая энергия, crest, маскировка.
    knobs_json — обработка стемов перед измерением, напр.
    '{"vocals":{"gain_db":3}}'."""
    return run("meters", "--track", str(track), "--knobs", knobs_json)


@mcp.tool()
def describe(track: int, prompt: str = "Describe the mix: what's working, "
             "what feels weak or missing? 3-4 sentences.",
             knobs_json: str = "{}") -> str:
    """Music Flamingo слушает микс и отвечает на prompt (семантика:
    жанр, инструменты, настроение, что не так; уровни не судит)."""
    return run("caption", "--track", str(track), "--prompt", prompt,
               "--knobs", knobs_json)


@mcp.tool()
def jev_ask(track: int, question_json: str, knobs_json: str = "{}") -> str:
    """Рендер с knobs_json и типизированный вопрос AudioJev:
    score {"id":{"type":"score","instructions":"...","criteria":["от","до"]}},
    noul  {"id":{"type":"noul","instructions":"да/нет"}},
    choice{"id":{"type":"choice","instructions":"...","criteria":{"k":"опис"}}}.
    Слышит уровни (~±3-4.5 дБ JND), слаб на EQ, вкус смещён."""
    return run("jev", "--track", str(track), "--knobs", knobs_json,
               "--question", question_json)


@mcp.tool()
def compare(track: int, knobs_a_json: str, knobs_b_json: str,
            question_json: str) -> str:
    """A/B: тот же вопрос на двух наборах knobs → ответы + дельта B−A."""
    return run("compare", "--track", str(track), "--knobs", knobs_a_json,
               "--knobs2", knobs_b_json, "--question", question_json)


@mcp.tool()
def fix_level(track: int, stem: str = "vocals", target: float = 2.48) -> str:
    """Бисекция AudioJev: какая gain_db-коррекция приведёт стем к
    «идеалу» модели (target — откалиброванная точка score)."""
    return run("fix", "--track", str(track), "--track-knob", stem,
               "--target", str(target))


@mcp.tool()
def bounce(track: int, knobs_json: str = "{}", out: str = "") -> str:
    """Баунс микса с обработкой knobs_json в WAV-файл."""
    args = ["render", "--track", str(track), "--knobs", knobs_json]
    if out:
        args += ["--out", out]
    return run(*args)


if __name__ == "__main__":
    mcp.run()
