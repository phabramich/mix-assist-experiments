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


def _sess(session):
    return ["--session", session] if session else []


def _win(from_s, to_s):
    a = []
    if from_s is not None:
        a += ["--from_s", str(from_s)]
    if to_s is not None:
        a += ["--to_s", str(to_s)]
    return a


@mcp.tool()
def list_sessions() -> str:
    """Список доступных мультитрек-сессий (индексы и названия)."""
    return run("tracks")


@mcp.tool()
def measure_file(path: str) -> str:
    """Метры WAV-файла (баунс/соло): LUFS, полосы, crest."""
    return run("meters", "--file", path)


@mcp.tool()
def describe_file(path: str, prompt: str = "Describe this audio: genre, "
                  "instruments, mood, mix issues. 3-4 sentences.",
                  from_s: float = None, to_s: float = None) -> str:
    """Music Flamingo слушает WAV-файл (или его участок from_s..to_s
    секунд) и отвечает на prompt."""
    return run("caption", "--file", path, "--prompt", prompt,
               *_win(from_s, to_s))


@mcp.tool()
def listen_file(path: str, question_json: str, from_s: float = None,
                to_s: float = None) -> str:
    """AudioJev отвечает на типизированный вопрос про WAV-файл (или
    участок from_s..to_s сек). Форматы вопросов — как у jev_ask.
    Ответ включает window_s — что именно услышал судья."""
    return run("jev", "--file", path, "--question", question_json,
               *_win(from_s, to_s))


@mcp.tool()
def compare_files(path_a: str, path_b: str, question_json: str,
                  from_s: float = None, to_s: float = None) -> str:
    """A/B двух WAV-баунсов одним вопросом AudioJev → ответы + дельта B−A.
    Режим «послушай микс до/после правки в DAW»."""
    return run("compare", "--file", path_a, "--file2", path_b,
               "--question", question_json, *_win(from_s, to_s))


@mcp.tool()
def measure(track: int = 0, knobs_json: str = "{}",
            session: str = "") -> str:
    """Метры микса: LUFS по стемам, полосовая энергия, crest, маскировка.
    knobs_json — обработка стемов перед измерением, напр.
    '{"vocals":{"gain_db":3}}'. session — папка со стемами WAV."""
    return run("meters", "--track", str(track), "--knobs", knobs_json,
               *_sess(session))


@mcp.tool()
def describe(track: int = 0, prompt: str = "Describe the mix: what's working, "
             "what feels weak or missing? 3-4 sentences.",
             knobs_json: str = "{}", session: str = "") -> str:
    """Music Flamingo слушает микс и отвечает на prompt (семантика:
    жанр, инструменты, настроение, что не так; уровни не судит)."""
    return run("caption", "--track", str(track), "--prompt", prompt,
               "--knobs", knobs_json, *_sess(session))


@mcp.tool()
def jev_ask(track: int = 0, question_json: str = "",
            knobs_json: str = "{}", session: str = "") -> str:
    """Рендер с knobs_json и типизированный вопрос AudioJev:
    score {"id":{"type":"score","instructions":"...","criteria":["от","до"]}},
    noul  {"id":{"type":"noul","instructions":"да/нет"}},
    choice{"id":{"type":"choice","instructions":"...","criteria":{"k":"опис"}}}.
    Слышит уровни (~±3-4.5 дБ JND), слаб на EQ, вкус смещён."""
    return run("jev", "--track", str(track), "--knobs", knobs_json,
               "--question", question_json, *_sess(session))


@mcp.tool()
def compare(track: int = 0, knobs_a_json: str = "{}",
            knobs_b_json: str = "{}", question_json: str = "",
            session: str = "") -> str:
    """A/B: тот же вопрос на двух наборах knobs → ответы + дельта B−A."""
    return run("compare", "--track", str(track), "--knobs", knobs_a_json,
               "--knobs2", knobs_b_json, "--question", question_json,
               *_sess(session))


@mcp.tool()
def fix_level(track: int = 0, stem: str = "vocals", target: float = 2.48,
              session: str = "") -> str:
    """Бисекция AudioJev: какая gain_db-коррекция приведёт стем к
    «идеалу» модели (target — откалиброванная точка score)."""
    return run("fix", "--track", str(track), "--track-knob", stem,
               "--target", str(target), *_sess(session))


@mcp.tool()
def bounce(track: int = 0, knobs_json: str = "{}", out: str = "",
           session: str = "") -> str:
    """Баунс микса с обработкой knobs_json в WAV-файл."""
    args = ["render", "--track", str(track), "--knobs", knobs_json,
            *_sess(session)]
    if out:
        args += ["--out", out]
    return run(*args)


if __name__ == "__main__":
    mcp.run()
