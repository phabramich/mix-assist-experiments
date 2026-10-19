"""mixctl — инструментарий микс-агента (CLI -> JSON).

Модель мира: --spoil скрыто деградирует стемы (состояние «сессии»),
--knobs — текущая обработка поверх. Агент зовёт команды, читает JSON.

  tracks
  meters  --track 6 [--spoil '{"vocals":{"gain_db":-6}}']
  caption --track 6 --prompt "..." [--spoil ..]
  jev     --track 6 --knobs '{"vocals":{"gain_db":3}}' --question '{...}'
  fix     --track 6 --track-knob vocals [--spoil ..]
  render  --track 6 --knobs '{..}' --out /tmp/mix.wav

Файл-режим (баунсы/соло из DAW вместо стемов):
  meters  --file mix.wav | caption/jev --file x.wav --question ..
  compare --file A.wav --file2 B.wav --question ..
  --from_s/--to_s — явное окно в секундах (иначе 12с вокруг пика).
  Ответы jev/caption/compare включают window_s — что именно слышал судья.

knobs на стем: gain_db, hpf_hz, lowmid_f/lowmid_db (bell),
pres_f/pres_db (bell), shelf_db (5к), pan (-1..1), haas_ms (0..30),
rev_db (send в общий ревер, <=-80 выкл). "_master": {"gain_db":..} —
пост-фейдер всей суммы.
"""
import argparse
import base64
import enum
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import scipy.signal as sig
import soundfile as sf

import architect
import judge
import megami_bridge
import meters
import plan as planmod
import render
import sections as sectionsmod
import fix as fixmod
from probe import stems_of, STATE

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent


class JobState(enum.Enum):
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"


def _job_start(rest):
    jid = f"{int(time.time())}-{os.urandom(2).hex()}"
    log = ROOT / "jobs" / f"{jid}.log"
    log.parent.mkdir(exist_ok=True)
    flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [sys.executable, str(ROOT / "mixctl.py"), "job-wrap", jid, "--", *rest],
        cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL, creationflags=flags, close_fds=True)
    print(json.dumps({"job": jid, "log": str(log)}))


def _job_wrap(jid, rest):
    log = ROOT / "jobs" / f"{jid}.log"
    log.parent.mkdir(exist_ok=True)
    with log.open("w", encoding="utf-8") as f:
        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        r = subprocess.run([sys.executable, str(ROOT / "mixctl.py"), *rest],
                           cwd=ROOT, stdout=f, stderr=subprocess.STDOUT,
                           text=True, encoding="utf-8", env=env)
    (ROOT / "jobs" / f"{jid}.done.json").write_text(
        json.dumps({"returncode": r.returncode}), encoding="utf-8")


def _job_status(jid):
    log = ROOT / "jobs" / f"{jid}.log"
    done = ROOT / "jobs" / f"{jid}.done.json"
    tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:] if log.exists() else []
    result = None
    if tail:
        try:
            result = json.loads(tail[-1])
        except json.JSONDecodeError:
            result = None
    if not done.exists():
        state, rc = JobState.RUNNING, None
    else:
        rc = json.loads(done.read_text(encoding="utf-8"))["returncode"]
        state = JobState.DONE if rc == 0 else JobState.FAILED
    print(json.dumps({"state": state.value, "returncode": rc,
                      "log_tail": tail, "result": result}, ensure_ascii=False))

DB = "data/MUSDB18/MUSDB18-7"
AJ = "https://phabramich--mix-assist-audiojev-serve.us-east.modal.run"
MF = "https://phabramich--mix-assist-mf-mf-chat.us-east.modal.run"


def session_dir(s):
    if not s:
        raise SystemExit("нужен --session")
    p = Path(s)
    return p if p.exists() else ROOT / "sessions" / s


def _stem_name(stem):
    return re.sub(r"[^\w]+", "_", stem.lower()).strip("_")


def load(track, spoil=None, session=None):
    if session:                      # папка-сессия: dry/*.wav или плоские wav
        stems, fs = {}, None
        root = Path(session)
        src = root / "dry" if (root / "dry").is_dir() else root
        for p in sorted(src.glob("*.wav")):
            x, fs = sf.read(str(p), dtype="float32", always_2d=True)
            stems[p.stem] = x
        return stems, fs, Path(session).name
    import musdb
    tr = musdb.DB(root=DB, is_wav=False)[track]
    stems = stems_of(tr)
    for name, kn in (spoil or {}).items():
        if "gain_db" in kn:
            stems[name] = stems[name] * 10 ** (kn["gain_db"] / 20)
    return stems, tr.rate, tr.name


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "job-start":
        rest = sys.argv[2:]
        if rest[:1] == ["--"]:
            rest = rest[1:]
        _job_start(rest)
        return
    if len(sys.argv) > 1 and sys.argv[1] == "job-wrap":
        rest = sys.argv[3:]
        if rest[:1] == ["--"]:
            rest = rest[1:]
        _job_wrap(sys.argv[2], rest)
        return

    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tracks", "meters", "caption", "jev",
                                   "fix", "render", "compare",
                                   "session-init", "megami-bank", "megami-render",
                                   "sections", "save-sections", "save-plan",
                                   "render-plan", "space", "megami-candidates", "job-status"])
    ap.add_argument("--src", default="")
    ap.add_argument("--name", default="")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--stem", default="")
    ap.add_argument("--z", type=int, default=0)
    ap.add_argument("--bpm", type=float, default=None)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--no-mf", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--json", default="")
    ap.add_argument("--labels", default="")
    ap.add_argument("--zs", default="")
    ap.add_argument("--id", default="")
    ap.add_argument("--track", type=int, default=6)
    ap.add_argument("--session", default="", help="папка со стемами .wav")
    ap.add_argument("--spoil", default="{}")
    ap.add_argument("--knobs", default="{}")
    ap.add_argument("--knobs2", default="{}")
    ap.add_argument("--prompt", default="Describe the mix: what's working, "
                    "what feels weak or missing? 3-4 sentences.")
    ap.add_argument("--question", default="")
    ap.add_argument("--track-knob", default="vocals")
    ap.add_argument("--target", type=float, default=2.48)
    ap.add_argument("--out", default="")
    ap.add_argument("--file", default="", help="WAV-файл вместо стемов "
                    "(баунс из DAW) — meters/caption/jev/compare")
    ap.add_argument("--file2", default="", help="второй WAV для compare")
    ap.add_argument("--from_s", type=float, default=None)
    ap.add_argument("--to_s", type=float, default=None)
    a = ap.parse_args()
    spoil, knobs = json.loads(a.spoil), json.loads(a.knobs)

    if a.cmd == "tracks":
        import musdb
        db = musdb.DB(root=DB, is_wav=False)
        print(json.dumps([{"i": i, "name": t.name}
                          for i, t in enumerate(db[:20])]))
        return

    if a.cmd == "session-init":
        src = Path(a.src)
        if not src.is_dir():
            raise SystemExit(f"нет папки: {src}")
        dest = ROOT / "sessions" / a.name / "dry"
        dest.mkdir(parents=True, exist_ok=True)
        wavs = sorted(src.glob("*.wav"))
        if not wavs:
            raise SystemExit(f"нет wav в {src}")
        loaded = []
        fs_out = 44100
        for p in wavs:
            x, fs = sf.read(str(p), dtype="float32", always_2d=True)
            if fs != fs_out:
                g = np.gcd(fs, fs_out)
                x = sig.resample_poly(x, fs_out // g, fs // g, axis=0).astype(np.float32)
            loaded.append((_stem_name(p.stem), x))
        n = max(len(x) for _, x in loaded)
        names = []
        for name, x in loaded:
            if len(x) < n:
                x = np.pad(x, ((0, n - len(x)), (0, 0)))
            sf.write(dest / f"{name}.wav", x, fs_out, subtype="FLOAT")
            names.append(name)
        print(json.dumps({"session": str(ROOT / "sessions" / a.name),
                          "stems": names, "duration_s": round(n / fs_out, 3)}))
        return

    if a.cmd in ("megami-bank", "megami-render"):
        sess = session_dir(a.session)
        if a.cmd == "megami-bank":
            megami_bridge.bank(sess, a.n)
            print(json.dumps(megami_bridge.bank_meta(sess), ensure_ascii=False))
            return
        if not a.stem or a.to_s is None:
            raise SystemExit("megami-render: нужны --stem --from_s --to_s")
        t0, t1 = round(a.from_s or 0.0, 2), round(a.to_s, 2)
        megami_bridge.ensure_wet(sess, [(a.stem, a.z, t0, t1)])
        path = megami_bridge.wet_path(sess, a.stem, a.z, t0, t1)
        x, fs = sf.read(str(path), dtype="float32", always_2d=True)
        rms = float(np.sqrt(np.mean(x ** 2)))
        print(json.dumps({"wav": str(path), "seconds": round(len(x) / fs, 3),
                          "channels": int(x.shape[1]), "rms": round(rms, 5)}))
        return

    if a.cmd == "job-status":
        if not a.id:
            raise SystemExit("job-status: нужен --id")
        _job_status(a.id)
        return

    if a.cmd == "megami-candidates":
        sess = session_dir(a.session)
        sections = sectionsmod.load(sess, require_labels=True)
        meta = megami_bridge.bank_meta(sess)
        if not meta:
            raise SystemExit("нет банка z — сначала megami-bank")
        if not a.stem:
            raise SystemExit("megami-candidates: нужен --stem")
        labels = [s for s in a.labels.split(",") if s]
        if not labels:
            raise SystemExit("megami-candidates: нужен --labels")
        zs = [int(z) for z in a.zs.split(",") if z] if a.zs else list(range(meta["n"]))
        stems, fs, _ = load(a.track, session=sess)
        dry = stems[a.stem]
        windows, items = {}, []
        for label in labels:
            group = [s for s in sections["sections"] if s.get("label") == label]
            if not group:
                raise SystemExit(f"нет секций с label {label}")
            sec = max(group, key=lambda s: s["end"] - s["start"])
            mid = (sec["start"] + sec["end"]) / 2
            t0 = round(max(sec["start"], mid - 6), 2)
            t1 = round(min(sec["end"], mid + 6), 2)
            windows[label] = [t0, t1]
            items.extend((a.stem, z, t0, t1) for z in zs)
        megami_bridge.ensure_wet(sess, items)
        out = {"stem": a.stem, "window_s": windows, "candidates": {}}
        for label, (t0, t1) in windows.items():
            i0, i1 = int((t0 + 0.5) * fs), int((t1 - 0.5) * fs)
            rows = []
            for z in zs:
                w, _ = sf.read(megami_bridge.wet_path(sess, a.stem, z, t0, t1),
                               dtype="float32", always_2d=True)
                o0 = int(0.5 * fs)
                piece = planmod.align_wet(w)[o0:o0 + max(i1 - i0, 0)]
                row = {"z": z}
                row.update(meters.space(dry[i0:i1], piece, fs))
                row["lufs"] = round(float(meters.stem_lufs(piece, fs)), 2)
                row["bands"] = {
                    nm: round(meters.band_share(render.to_mono(piece), fs, lo, hi), 3)
                    for lo, hi, nm in meters.BANDS}
                rows.append(row)
            rows.sort(key=lambda r: r["diffuse_db"])
            out["candidates"][label] = rows
        dest = sess / "megami" / f"candidates_{a.stem}.json"
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps(out, ensure_ascii=False))
        return

    if a.cmd == "sections":
        sess = session_dir(a.session)
        outp = sess / "sections.json"
        if outp.exists() and not a.force:
            print(outp.read_text(encoding="utf-8"))
            return
        stems, fs, _ = load(a.track, session=sess)
        describe = None
        if not a.no_mf:
            def describe(clip):
                try:
                    return architect.ask_mf(MF, clip, sectionsmod.SECTION_PROMPT)
                except Exception as e:
                    return f"mf недоступен: {e}"
        data = sectionsmod.detect(stems, fs, bpm=a.bpm, describe=describe)
        outp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"sections": len(data["sections"]),
                          "clusters": list(data["clusters"]), "bpm": data["bpm"]}))
        return

    if a.cmd == "save-sections":
        sess = session_dir(a.session)
        data = json.loads(a.json)
        (sess / "sections.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        sectionsmod.load(sess, require_labels=False)
        print(json.dumps({"sections": len(data["sections"])}))
        return

    if a.cmd == "save-plan":
        sess = session_dir(a.session)
        (sess / "plan.json").write_text(json.dumps(json.loads(a.json), ensure_ascii=False, indent=2), encoding="utf-8")
        planmod.load(sess)
        print(json.dumps({"plan": str(sess / "plan.json")}))
        return

    if a.cmd in ("render-plan", "space"):
        sess = session_dir(a.session)
        sections = sectionsmod.load(sess, require_labels=True)
        plan = planmod.load(sess)
        stems, fs, _ = load(a.track, session=sess)
        dur = sections["duration_s"]
        items = planmod.wet_items(plan, sections, list(stems), dur)
        if a.cmd == "space":
            items = [it for it in items if it[0] == a.stem]
        miss = megami_bridge.missing(sess, items)
        if a.dry_run:
            print(json.dumps({"missing": len(miss), "estimate_min": megami_bridge.estimate_min(miss)}))
            return
        if a.cmd == "render-plan":
            megami_bridge.ensure_wet(sess, items)
            y = planmod.render_mix(sess, plan, sections, stems, fs)
            rows = []
            for sec in sections["sections"]:
                i0, i1 = int(sec["start"] * fs), int(sec["end"] * fs)
                piece = y[i0:i1]
                lufs = meters.stem_lufs(piece, fs) if len(piece) >= int(0.4 * fs) else None
                rows.append({"id": sec["id"], "label": sec["label"],
                             "lufs": None if lufs is None else round(float(lufs), 2)})
            mix_lufs = round(float(meters.stem_lufs(y, fs)), 2)
            if a.from_s is not None or a.to_s is not None:
                i0 = int((a.from_s or 0) * fs)
                i1 = int((a.to_s if a.to_s is not None else 1e9) * fs)
                y = y[i0:i1]
            out = Path(a.out) if a.out else ROOT / "renders" / f"plan-{int(__import__('time').time())}.wav"
            out.parent.mkdir(parents=True, exist_ok=True)
            render.save_wav(out, y, fs)
            (out.with_suffix(".plan.json")).write_text(
                json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({"wav": str(out), "lufs": mix_lufs, "sections": rows,
                              "rendered_megami": len(miss)}, ensure_ascii=False))
            return
        if not a.stem:
            raise SystemExit("space: нужен --stem")
        megami_bridge.ensure_wet(sess, items)
        y = planmod.render_mix(sess, plan, sections, stems, fs, only={a.stem})
        xf = plan.get("xfade_s", 0.4)
        rows = []
        for sec in sections["sections"]:
            a0 = int((sec["start"] + xf) * fs)
            b0 = int((sec["end"] - xf) * fs)
            if b0 <= a0:
                a0, b0 = int(sec["start"] * fs), int(sec["end"] * fs)
            row = {"id": sec["id"], "label": sec["label"]}
            row.update(meters.space(stems[a.stem][a0:b0], y[a0:b0], fs))
            rows.append(row)
        print(json.dumps(rows, ensure_ascii=False))
        return

    def validate_q(q):
        for qid, qq in q.items():
            t = qq.get("type")
            if t == "score" and not isinstance(qq.get("criteria"), list):
                raise SystemExit(f"score '{qid}': criteria = упорядоченный список строк")
            if t == "choice" and not isinstance(qq.get("criteria"), dict):
                raise SystemExit(f"choice '{qid}': criteria = объект ключ->описание")
            if t not in ("score", "choice", "noul"):
                raise SystemExit(f"'{qid}': type ∈ score|choice|noul")

    window = None
    if a.from_s is not None or a.to_s is not None:
        window = (a.from_s or 0.0, a.to_s if a.to_s is not None else 1e9)

    if a.file:  # файл-режим: судьи слушают WAV (баунс/соло из DAW)
        x, fs = sf.read(a.file, dtype="float32", always_2d=True)
        fname = Path(a.file).stem
        if a.cmd == "meters":
            r = meters.report({fname: x}, fs)
            print(json.dumps({"file": a.file, "report": r["text"],
                              "mix_lufs": r["mix_lufs"]}, ensure_ascii=False))
            return
        if a.cmd == "render":
            out = a.out or os.path.join(tempfile.gettempdir(), f"{fname}.wav")
            render.save_wav(out, x, fs)
            print(json.dumps({"wav": out, "lufs": meters.stem_lufs(x, fs)}))
            return
        stems = {fname: x}
        clip, win = render.render_for_judge(stems, {}, fs,
                                            window=window, return_window=True)
        if a.cmd == "caption":
            txt = architect.ask_mf(MF, clip, a.prompt)
            print(json.dumps({"text": txt, "window_s": win},
                             ensure_ascii=False))
            return
        if a.cmd == "jev":
            q = json.loads(a.question); validate_q(q)
            ans = judge.ask_audiojev(AJ, clip, STATE, q)
            print(json.dumps({"answers": ans["answers"], "window_s": win},
                             ensure_ascii=False))
            return
        if a.cmd == "compare":
            if not a.file2:
                raise SystemExit("compare --file требует --file2")
            x2, _ = sf.read(a.file2, dtype="float32", always_2d=True)
            c2, win2 = render.render_for_judge({Path(a.file2).stem: x2}, {},
                                               fs, window=window,
                                               return_window=True)
            q = json.loads(a.question); validate_q(q)
            out = {"A": {"file": a.file, "window_s": win,
                         "answers": judge.ask_audiojev(AJ, clip, STATE, q)["answers"]},
                   "B": {"file": a.file2, "window_s": win2,
                         "answers": judge.ask_audiojev(AJ, c2, STATE, q)["answers"]}}
            delta = {}
            for qid in q:
                sa, sb = out["A"]["answers"][qid], out["B"]["answers"][qid]
                for k in ("score", "noul", "confidence"):
                    if k in sa and k in sb:
                        delta.setdefault(qid, {})[k] = round(sb[k] - sa[k], 3)
            out["delta_B_minus_A"] = delta
            print(json.dumps(out, ensure_ascii=False))
            return
        raise SystemExit(f"--file: команда '{a.cmd}' не поддерживается "
                         "(meters/caption/jev/compare/render)")

    stems, fs, name = load(a.track, spoil, a.session or None)

    if a.cmd == "meters":
        stems = {n: render.apply_knobs(x, knobs.get(n, {}), fs)
                 for n, x in stems.items()}
        r = meters.report(stems, fs)
        print(json.dumps({"track": name, "report": r["text"],
                          "mix_lufs": r["mix_lufs"]}, ensure_ascii=False))
    elif a.cmd == "caption":
        clip, win = render.render_for_judge(stems, knobs, fs, window=window,
                                            return_window=True)
        txt = architect.ask_mf(MF, clip, a.prompt)
        print(json.dumps({"text": txt, "window_s": win}, ensure_ascii=False))
    elif a.cmd == "jev":
        clip, win = render.render_for_judge(stems, knobs, fs, window=window,
                                            return_window=True)
        q = json.loads(a.question); validate_q(q)
        ans = judge.ask_audiojev(AJ, clip, STATE, q)
        print(json.dumps({"answers": ans["answers"], "window_s": win},
                         ensure_ascii=False))
    elif a.cmd == "compare":
        q = json.loads(a.question)
        knobs2 = json.loads(a.knobs2)
        out = {}
        for tag, kn in (("A", knobs), ("B", knobs2)):
            clip, win = render.render_for_judge(stems, kn, fs, window=window,
                                                return_window=True)
            out[tag] = {"knobs": kn, "window_s": win,
                        "answers": judge.ask_audiojev(AJ, clip, STATE, q)["answers"]}
        delta = {}
        for qid in q:
            sa, sb = out["A"]["answers"][qid], out["B"]["answers"][qid]
            for k in ("score", "noul", "confidence"):
                if k in sa and k in sb:
                    delta.setdefault(qid, {})[k] = round(sb[k] - sa[k], 3)
        out["delta_B_minus_A"] = delta
        print(json.dumps(out, ensure_ascii=False))
    elif a.cmd == "fix":
        true_off = (spoil.get(a.track_knob) or {}).get("gain_db", 0.0)
        def ask(q, c):
            return judge.ask_audiojev(AJ, c, STATE, q)
        c, hist = fixmod.fix_one(ask, stems, fs, 0.0, a.target, 2, 5,
                                 stem=a.track_knob)
        print(json.dumps({"track_knob": a.track_knob,
                          "correction_db": round(c, 1),  # >JND — ложная точность
                          "runaway_at_boundary": abs(c) > 11.0,
                          "caller_spoil_db": true_off or None,
                          "residual_vs_reference_db": (true_off + c)
                              if true_off else None,
                          "history": hist}, ensure_ascii=False))
    elif a.cmd == "render":
        x = render.mix(stems, knobs, fs)
        out = a.out or os.path.join(tempfile.gettempdir(), f"mix-{a.track}.wav")
        render.save_wav(out, x, fs)
        print(json.dumps({"wav": out, "lufs": meters.stem_lufs(x, fs)}))


if __name__ == "__main__":
    main()
