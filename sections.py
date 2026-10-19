"""Авторазметка секций: биты, новизна, активность стемов, кластеры."""
import json
from pathlib import Path

import librosa
import numpy as np
import scipy.cluster.hierarchy as hier
import scipy.signal as sig

import render

SR = 22050
HOP = 512
KERNEL_BEATS = 16
MIN_BEATS = 8
SECTION_PROMPT = (
    "This is an excerpt of a song. In 1-2 sentences: which instruments and voices "
    "are present, the energy level, and whether it sounds like an intro, verse, "
    "pre-chorus, chorus, bridge, solo or outro."
)


def _activity(x, fs, t):
    db = np.array([
        20 * np.log10(np.sqrt(np.mean(x[int(a * fs):int(b * fs)] ** 2)) + 1e-9)
        for a, b in zip(t[:-1], t[1:])
    ])
    return (db > db.max() - 30).astype(float)


def _z(m):
    return (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-9)


def _foote(sim, length):
    g = sig.windows.gaussian(2 * length, std=length / 2)
    kernel = np.outer(g, g) * np.kron([[1, -1], [-1, 1]], np.ones((length, length)))
    padded = np.pad(sim, length)
    return np.array([
        (padded[i:i + 2 * length, i:i + 2 * length] * kernel).sum()
        for i in range(len(sim))
    ])


def _act_changes(act, hold=MIN_BEATS):
    out = []
    for i in range(1, act.shape[1] - hold + 1):
        changed = (act[:, i] != act[:, i - 1]).any()
        held = (act[:, i:i + hold] == act[:, [i]]).all()
        if changed and held:
            out.append(i)
    return out


def _merge(idx, nov, min_gap, n_beats):
    keep = []
    for i in sorted(idx, key=lambda k: -nov[k] if k < len(nov) else 0):
        if min_gap <= i <= n_beats - min_gap and all(abs(i - k) >= min_gap for k in keep):
            keep.append(i)
    return [0] + sorted(keep) + [n_beats]


def detect(stems, fs, bpm=None, describe=None):
    mono = {n: render.to_mono(x).astype(np.float32) for n, x in stems.items()}
    mix = sum(mono.values())
    dur = len(mix) / fs
    y = librosa.resample(mix, orig_sr=fs, target_sr=SR)
    tempo, beats = librosa.beat.beat_track(
        y=y, sr=SR, start_bpm=bpm or 120.0,
        tightness=400 if bpm else 100, units="time")
    bpm_out = float(bpm) if bpm else float(np.atleast_1d(tempo)[0])
    t = np.unique(np.concatenate([[0.0], np.atleast_1d(beats), [dur]]))
    if len(t) < MIN_BEATS * 2:
        bounds = [0, len(t) - 1]
        feats = np.zeros((1, 1))
        act = np.zeros((len(mono), max(len(t) - 1, 1)))
    else:
        fr = librosa.time_to_frames(t, sr=SR, hop_length=HOP)
        chroma = librosa.util.sync(
            librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP), fr, aggregate=np.median)
        mfcc = librosa.util.sync(
            librosa.feature.mfcc(y=y, sr=SR, n_mfcc=13, hop_length=HOP), fr)
        act = np.stack([_activity(mono[n], fs, t) for n in mono])
        m = min(chroma.shape[1], mfcc.shape[1], act.shape[1])
        chroma, mfcc, act = chroma[:, :m], mfcc[:, :m], act[:, :m]
        feats = np.vstack([_z(chroma), _z(mfcc), 2.0 * act]).T
        norm = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
        nov = _foote(norm @ norm.T, KERNEL_BEATS)
        peaks, _ = sig.find_peaks(nov, distance=KERNEL_BEATS, prominence=np.percentile(nov, 75))
        bounds = _merge(sorted(set(peaks.tolist()) | set(_act_changes(act))), nov, MIN_BEATS, len(t) - 1)

    names = list(mono)
    raw = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        i0, i1 = int(t[a] * fs), int(min(t[b], dur) * fs)
        piece = mix[i0:i1]
        lufs = None
        if len(piece) >= int(0.4 * fs):
            lufs = round(float(__import__("pyloudnorm").Meter(fs).integrated_loudness(
                piece.astype(np.float64))), 1)
        active = []
        if b > a and act.shape[1]:
            aa, bb = min(a, act.shape[1]), min(b, act.shape[1])
            if bb > aa:
                active = [names[i] for i in range(len(names)) if act[i, aa:bb].mean() > 0.5]
        raw.append({"start": round(float(t[a]), 3), "end": round(float(min(t[b], dur)), 3),
                    "lufs": lufs, "active": active, "_a": a, "_b": b})

    vectors = []
    for sec in raw:
        sl = feats[sec["_a"]:sec["_b"]]
        vectors.append(sl.mean(0) if len(sl) else np.zeros(feats.shape[1]))
    if len(raw) == 1:
        labels = [1]
    else:
        labels = hier.fcluster(
            hier.linkage(np.vstack(vectors), method="average", metric="cosine"),
            t=0.3, criterion="distance")
    letters, seen = {}, []
    for lab in labels:
        if lab not in letters:
            letters[lab] = chr(ord("A") + len(seen))
            seen.append(lab)
    sections, clusters = [], {}
    for i, (sec, lab) in enumerate(zip(raw, labels), start=1):
        letter = letters[lab]
        sec.pop("_a")
        sec.pop("_b")
        sec.update({"id": f"s{i:02d}", "cluster": letter, "label": ""})
        sections.append(sec)
        clusters.setdefault(letter, {"count": 0, "mf": ""})
        clusters[letter]["count"] += 1

    if describe:
        for letter in clusters:
            group = [s for s in sections if s["cluster"] == letter]
            sec = max(group, key=lambda s: s["end"] - s["start"])
            mid = (sec["start"] + sec["end"]) / 2
            t0, t1 = max(sec["start"], mid - 6), min(sec["end"], mid + 6)
            clip = render.render_for_judge(stems, {}, fs, window=(t0, t1))
            clusters[letter]["mf"] = describe(clip)

    return {"bpm": round(bpm_out, 2), "duration_s": round(dur, 3),
            "sections": sections, "clusters": clusters}


def load(session, require_labels=False):
    path = Path(session) / "sections.json"
    if not path.exists():
        raise SystemExit(f"нет {path}")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    secs = data.get("sections") or []
    if not secs:
        raise SystemExit("sections.json: нет секций")
    if [s["start"] for s in secs] != sorted(s["start"] for s in secs):
        raise SystemExit("sections.json: секции не по порядку")
    if abs(secs[0]["start"]) > 1e-3:
        raise SystemExit("sections.json: первая секция должна начинаться с 0")
    for a, b in zip(secs, secs[1:]):
        if abs(a["end"] - b["start"]) > 1e-3:
            raise SystemExit(f"sections.json: разрыв между {a['id']} и {b['id']}")
    if abs(secs[-1]["end"] - data["duration_s"]) > 1e-3:
        raise SystemExit("sections.json: последняя секция не доходит до конца")
    ids = [s["id"] for s in secs]
    if len(ids) != len(set(ids)):
        raise SystemExit("sections.json: повторяющиеся id")
    if require_labels and any(not s.get("label") for s in secs):
        raise SystemExit("sections.json: не у всех секций есть label")
    return data
