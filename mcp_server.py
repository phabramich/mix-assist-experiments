"""MCP-сервер mix-assist: те же инструменты, что и mixctl.py (stdio).

Каждый tool = вызов `.venv/bin/python mixctl.py ...`, вывод = JSON-строка.
Запуск: .venv/bin/python mcp_server.py
"""
import json
import subprocess
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

PY = sys.executable
CWD = str(Path(__file__).resolve().parent)
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


PLAN_DOC = """Записать plan.json. Схема:
{"z_default": 0, "xfade_s": 0.4, "stems": {"vox": {"z": 2, "wet": 1.0,
 "knobs": {"gain_db": 1.5}, "sections": {"verse": {"wet": 0.4}, "s07": {"z": 1}}}},
 "_master": {"gain_db": 0}}
Разрешение для (стем, секция): дефолт z_default/wet=1/пустые knobs, затем поля стема,
затем sections[label], затем sections[id]. wet 0..1: 0 = сухой сигнал, 1 = MEGAMI.
Ручки: gain_db, hpf_hz, lowmid_f, lowmid_db, pres_f, pres_db, shelf_db, pan, haas_ms, rev_db.
Стемы вне плана получают дефолты. z должен существовать в банке, если wet > 0."""


@mcp.tool()
def session_init(src: str, name: str) -> str:
    """Скопировать WAV стемов в sessions/<name>/dry (44.1 кГц, общая длина)."""
    return run("session-init", "--src", src, "--name", name)


@mcp.tool()
def detect_sections(session: str, bpm: float = 0, force: bool = False,
                    use_mf: bool = True) -> str:
    """Найти секции (биты, новизна, активность стемов) и записать sections.json.
    bpm=0 — оценить темп. use_mf — попросить Music Flamingo описать кластеры.
    Существующий файл не перезаписывается без force."""
    args = ["sections", "--session", session]
    if bpm:
        args += ["--bpm", str(bpm)]
    if force:
        args += ["--force"]
    if not use_mf:
        args += ["--no-mf"]
    return run(*args)


@mcp.tool()
def save_sections(session: str, sections_json: str) -> str:
    """Записать sections.json (после ручной правки label) и проверить целостность."""
    return run("save-sections", "--session", session, "--json", sections_json)


@mcp.tool()
def save_plan(session: str, plan_json: str) -> str:
    """Записать plan.json и проверить его. """ + PLAN_DOC
    return run("save-plan", "--session", session, "--json", plan_json)


@mcp.tool()
def measure_space(session: str, stem: str) -> str:
    """diffuse_db и width_db стема по каждой секции. Пространство слышат только эти метры."""
    return run("space", "--session", session, "--stem", stem)


@mcp.tool()
def job_status(job_id: str) -> str:
    """Состояние фоновой задачи: running | done | failed, хвост лога, JSON-результат."""
    return run("job-status", "--id", job_id)


@mcp.tool()
def render_plan_estimate(session: str) -> str:
    """Сколько кусков MEGAMI ещё не в кэше и оценка минут. Ничего не рендерит."""
    return run("render-plan", "--session", session, "--dry-run")


@mcp.tool()
def megami_bank(session: str, n: int = 8) -> str:
    """Фоном сгенерировать банк из n эмбеддингов стиля. Вернёт job id, статус — job_status."""
    return run("job-start", "--", "megami-bank", "--session", session, "--n", str(n))


@mcp.tool()
def megami_candidates(session: str, stem: str, labels: str) -> str:
    """Фоном отрендерить 12 с каждого z для стема на label (через запятую) и измерить пространство.
    labels например 'verse,chorus'. Результат — megami/candidates_<stem>.json."""
    return run("job-start", "--", "megami-candidates", "--session", session,
               "--stem", stem, "--labels", labels)


@mcp.tool()
def render_plan(session: str, out: str = "", from_s: float = None,
                to_s: float = None) -> str:
    """Фоном собрать микс по plan.json. Недостающие куски MEGAMI дорендерит.
    Повторный вызов после правки ручек берёт кэш."""
    args = ["job-start", "--", "render-plan", "--session", session]
    if out:
        args += ["--out", out]
    if from_s is not None:
        args += ["--from_s", str(from_s)]
    if to_s is not None:
        args += ["--to_s", str(to_s)]
    return run(*args)


if __name__ == "__main__":
    mcp.run()
