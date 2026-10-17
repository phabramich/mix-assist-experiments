"""Этап 0, пробник T1/T2: слышит ли модель сдвиг уровня вокала.

Сервер:  llama-server -hf LiquidAI/d1-omni-600M-GGUF:Q8_0 -b 4096 -ub 4096
Запуск:  .venv/bin/python probe.py --tracks 3 --server http://127.0.0.1:8080
"""
import argparse
import json
from pathlib import Path

import musdb
import numpy as np

import judge
import render

SCALE = ["far too quiet", "a bit quiet", "balanced", "a bit loud", "far too loud"]

SANITY = {"kind": {
    "type": "choice",
    "instructions": "What kind of recording is this?",
    "criteria": {"music": "a music recording", "speech": "spoken word",
                 "noise": "noise or ambience"}}}

VOCAL_SCORE = {"vocal_level": {
    "type": "score",
    "instructions": "How loud is the lead vocal relative to the rest of the mix?",
    "criteria": SCALE}}

WHICH_LOUDER = {"louder_half": {
    "type": "choice",
    "instructions": "This clip contains the same song fragment twice, separated "
                    "by a short beep. In which half is the lead vocal louder "
                    "relative to the instruments?",
    "criteria": {"first": "the half before the beep",
                 "second": "the half after the beep"}}}

STATE = "A 7-second fragment of a song mix. Judge it as a mixing engineer would hear it."


def stems_of(track):
    return {n: track.targets[n].audio for n in ("vocals", "drums", "bass", "other")}


def beep(fs=16000, dur=0.3):
    t = np.arange(int(fs * dur)) / fs
    return (0.1 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)


def rendered(stems, vocal_db, fs):
    return render.render_for_judge(stems, {"vocals": {"gain_db": vocal_db}}, fs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8080")
    ap.add_argument("--judge", choices=["d1", "audiojev"], default="d1")
    ap.add_argument("--tracks", type=int, default=3)
    ap.add_argument("--offsets", default="-9,-6,-3,0,3,6,9")
    ap.add_argument("--afc-offsets", default="-6,6")
    ap.add_argument("--out", default="experiments/001-vocal-level.json")
    args = ap.parse_args()
    offsets = [float(s) for s in args.offsets.split(",")]
    afc_offsets = [float(s) for s in args.afc_offsets.split(",")]

    if args.judge == "audiojev":
        def ask(q, clip):
            return judge.ask_audiojev(args.server, clip, STATE, q)
    else:
        def ask(q, clip):
            return judge.ask(args.server, STATE, q, clips=[clip])

    db = musdb.DB(root="data/MUSDB18/MUSDB18-7", is_wav=False)
    tracks = db[: args.tracks]
    results = []

    first = rendered(stems_of(tracks[0]), 0.0, tracks[0].rate)
    sane = ask(SANITY, first)
    print("sanity:", sane["answers"]["kind"])
    if sane["answers"]["kind"]["choice"] != "music":
        print("ВНИМАНИЕ: сервер не распознал музыку — проверить пайплайн")

    for tr in tracks:
        stems = stems_of(tr)
        fs = tr.rate
        row = {"track": tr.name, "t1": {}, "t2": {}}

        # T1: абсолютная шкала уровня вокала на каждом сдвиге
        for off in offsets:
            clip = rendered(stems, off, fs)
            ans = ask(VOCAL_SCORE, clip)
            a = ans["answers"]["vocal_level"]
            row["t1"][off] = {"score": a["score"],
                              "probabilities": a["probabilities"],
                              "confidence": a.get("confidence")}
            probs = {k: round(v, 2) for k, v in a["probabilities"].items()}
            print(f"{tr.name[:40]:42} off {off:+5.1f} -> score {a['score']:.2f} {probs}")

        # T2: 2AFC склейка «ref | бип | испорченный» в обоих порядках
        ref = rendered(stems, 0.0, fs)
        for off in afc_offsets:
            deg = rendered(stems, off, fs)
            truth = "first" if off < 0 else "second"  # в какой половине вокал громче
            for order, (a, b) in {"AB": (ref, deg), "BA": (deg, ref)}.items():
                clip = np.concatenate([a, beep(), b])
                ans = ask(WHICH_LOUDER, clip)
                c = ans["answers"]["louder_half"]
                expected = truth if order == "AB" else {"first": "second", "second": "first"}[truth]
                key = f"{off:+g} {order}"
                row["t2"][key] = {"choice": c["choice"], "expected": expected,
                                  "probabilities": c["probabilities"],
                                  "confidence": c.get("confidence")}
                print(f"  2AFC {key}: -> {c['choice']} (правильно: {expected}) "
                      f"conf {c.get('confidence')}")
        results.append(row)

    Path(args.out).parent.mkdir(exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=1))

    # сводка по T1: медианный score по трекам на каждом сдвиге
    print("\nСводка T1 (медиана score по трекам):")
    for off in offsets:
        sc = [r["t1"][off]["score"] for r in results]
        print(f"  {off:+5.1f} дБ -> {np.median(sc):.2f}  {['%.2f' % s for s in sc]}")
    from scipy.stats import spearmanr
    all_off, all_sc = [], []
    for r in results:
        for off in offsets:
            all_off.append(off)
            all_sc.append(r["t1"][off]["score"])
    rho, p = spearmanr(all_off, all_sc)
    print(f"\nSpearman rho(offset, score) = {rho:.3f} (p={p:.4f})")

    hits, total = 0, 0
    for r in results:
        for v in r["t2"].values():
            total += 1
            hits += v["choice"] == v["expected"]
    print(f"2AFC точность: {hits}/{total}")


if __name__ == "__main__":
    main()
