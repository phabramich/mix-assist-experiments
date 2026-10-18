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
import json
import os
from pathlib import Path

import musdb
import numpy as np
import soundfile as sf

import architect
import judge
import meters
import render
import fix as fixmod
from probe import stems_of, STATE

DB = "data/MUSDB18/MUSDB18-7"
AJ = "https://phabramich--mix-assist-audiojev-serve.us-east.modal.run"
MF = "https://phabramich--mix-assist-mf-mf-chat.us-east.modal.run"


def load(track, spoil=None, session=None):
    if session:                      # папка-сессия: <stem>.wav
        stems, fs = {}, None
        for p in sorted(Path(session).glob("*.wav")):
            x, fs = sf.read(str(p), dtype="float32", always_2d=True)
            stems[p.stem] = x
        return stems, fs, Path(session).name
    tr = musdb.DB(root=DB, is_wav=False)[track]
    stems = stems_of(tr)
    for name, kn in (spoil or {}).items():
        if "gain_db" in kn:
            stems[name] = stems[name] * 10 ** (kn["gain_db"] / 20)
    return stems, tr.rate, tr.name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tracks", "meters", "caption", "jev",
                                   "fix", "render", "compare"])
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
        db = musdb.DB(root=DB, is_wav=False)
        print(json.dumps([{"i": i, "name": t.name}
                          for i, t in enumerate(db[:20])]))
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
            out = a.out or f"/tmp/{fname}.wav"
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
        out = a.out or f"/tmp/mix-{a.track}.wav"
        render.save_wav(out, x, fs)
        print(json.dumps({"wav": out, "lufs": meters.stem_lufs(x, fs)}))


if __name__ == "__main__":
    main()
