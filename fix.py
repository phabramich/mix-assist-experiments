"""Stage 1: контроллер «почини уровень вокала» бисекцией по Jev-score.

Сценарий: к миксу применяется скрытый сдвиг вокала (истина). Контроллер
прибавляет gain c дБ к вокалу и бисекцией ищет c, при котором
vocal_level score = target. Идеал: c* = -offset (микс как у оригинала).

Запуск:
  .venv/bin/python fix.py --judge audiojev --server <url> \
      --tracks 3,4,5 --offsets=-6,-3,3,6 --target 2.0
  калибровка: --calibrate --tracks 0,1,2  (score при offset=0)
"""
import argparse
import json
from pathlib import Path

import musdb
import numpy as np

import judge
import render
from probe import stems_of, STATE, SCALE


def _level_q(stem):
    return {"lvl": {"type": "score",
            "instructions": f"How loud is the {stem} relative to the rest of the mix?",
            "criteria": SCALE}}


def _quiet_q(stem):
    return {"lvl": {"type": "noul",
            "instructions": f"Is the {stem} too quiet relative to the rest of the mix?"}}


def vocal_score(ask, stems, total_gain_db, fs, repeats, mode="score",
                stem="vocals"):
    clip = render.render_for_judge(stems, {stem: {"gain_db": total_gain_db}}, fs)
    key = "lvl"
    if mode == "noul":
        vals = [ask(_quiet_q(stem), clip)["answers"][key]["noul"]
                for _ in range(repeats)]
    else:
        vals = [ask(_level_q(stem), clip)["answers"][key]["score"]
                for _ in range(repeats)]
    return float(np.mean(vals))


def fix_one(ask, stems, fs, offset, target, repeats, iters, rng=(-12.0, 12.0),
            mode="score", stem="vocals"):
    """offset — скрытый сдвиг. mode=noul: цель = P(too_quiet) 0.5
    (score ниже цели → вокал тихий → добавить gain)."""
    lo, hi = rng
    hist = []
    more = (lambda s, t: s > t) if mode == "noul" else (lambda s, t: s < t)
    for _ in range(iters):
        mid = (lo + hi) / 2
        s = vocal_score(ask, stems, offset + mid, fs, repeats, mode, stem)
        hist.append({"c": mid, "score": s})
        if more(s, target):   # вокал всё ещё тихий → поднять нижнюю границу
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2, hist


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="https://phabramich--mix-assist-audiojev-serve.us-east.modal.run")
    ap.add_argument("--judge", choices=["d1", "audiojev"], default="audiojev")
    ap.add_argument("--tracks", default="3,4,5")
    ap.add_argument("--offsets", default="-6,-3,3,6")
    ap.add_argument("--target", type=float, default=2.0,
                    help="score цели (2.0 = balanced); из калибровки ~2.3")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--mode", choices=["score", "noul"], default="score")
    ap.add_argument("--calibrate", action="store_true",
                    help="замерить score при offset=0 (базовая точка)")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if args.judge == "audiojev":
        def ask(q, clip):
            return judge.ask_audiojev(args.server, clip, STATE, q)
    else:
        def ask(q, clip):
            return judge.ask(args.server, STATE, q, clips=[clip])

    db = musdb.DB(root="data/MUSDB18/MUSDB18-7", is_wav=False)
    idxs = [int(i) for i in args.tracks.split(",")]
    offsets = [float(s) for s in args.offsets.split(",")]
    out = []

    if args.calibrate:
        for i in idxs:
            tr = db[i]
            s = vocal_score(ask, stems_of(tr), 0.0, tr.rate,
                            max(args.repeats, 3), args.mode)
            print(f"{tr.name[:40]:42} balanced score = {s:.2f}")
            out.append({"track": tr.name, "balanced_score": s})
        print(f"среднее: {np.mean([o['balanced_score'] for o in out]):.2f}")
    else:
        for i in idxs:
            tr = db[i]
            stems, fs = stems_of(tr), tr.rate
            for off in offsets:
                tgt = 0.5 if args.mode == "noul" else args.target
                c, hist = fix_one(ask, stems, fs, off, tgt,
                                  args.repeats, args.iters, mode=args.mode)
                err = abs(-c - off)          # оценка offset = -c; ошибка в дБ
                resid = abs(off + c)         # остаточный сдвиг после правки
                print(f"{tr.name[:36]:38} offset {off:+5.1f} -> c* {c:+5.2f} "
                      f"(est {-c:+5.2f}) ошибка {err:.2f} дБ | "
                      f"iter scores {[round(h['score'],2) for h in hist]}")
                out.append({"track": tr.name, "offset": off, "c": c,
                            "est_offset": -c, "err_db": err, "hist": hist})
        errs = [o["err_db"] for o in out]
        print(f"\nMAE оценки сдвига: {np.mean(errs):.2f} дБ  "
              f"(n={len(errs)}, медиана {np.median(errs):.2f})")

    if args.out:
        Path(args.out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
