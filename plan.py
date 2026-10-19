"""План сведения: разрешение ручек по секциям, кроссфейды, dry/wet."""
import json
from pathlib import Path

import numpy as np
import soundfile as sf

import megami_bridge
import meters
import render
import sections as sections_mod

PAD = 2.0
LEAD = 0.5
# FxProcessor опережает dry; замер на vox, окно 60–72 с, лаг −60 сэмплов
WET_DELAY = 60


def align_wet(x):
    if WET_DELAY == 0:
        return x
    y = np.zeros_like(x)
    if WET_DELAY > 0:
        y[WET_DELAY:] = x[:-WET_DELAY]
    else:
        y[:WET_DELAY] = x[-WET_DELAY:]
    return y


def _check_knobs(knobs):
    unknown = set(knobs) - set(render.DEFAULT_KNOBS)
    if unknown:
        raise SystemExit(f"plan: неизвестные ручки {sorted(unknown)}")


def resolve(plan, sections, stem):
    spec = plan.get("stems", {}).get(stem, {})
    base = {
        "z": spec.get("z", plan.get("z_default", 0)),
        "wet": spec.get("wet", 1.0),
        "knobs": dict(spec.get("knobs", {})),
    }
    out = []
    for sec in sections["sections"]:
        p = {"z": base["z"], "wet": base["wet"], "knobs": dict(base["knobs"])}
        keys = [sec["label"]] if sec.get("label") else []
        keys.append(sec["id"])
        for key in keys:
            sub = spec.get("sections", {}).get(key)
            if not sub:
                continue
            if "z" in sub:
                p["z"] = sub["z"]
            if "wet" in sub:
                p["wet"] = sub["wet"]
            p["knobs"].update(sub.get("knobs", {}))
        out.append(p)
    return out


def runs(sections, params):
    out = []
    for sec, p in zip(sections["sections"], params):
        if p["wet"] <= 0:
            continue
        if out and out[-1]["z"] == p["z"] and out[-1]["end"] == sec["start"]:
            out[-1]["end"] = sec["end"]
        else:
            out.append({"z": p["z"], "start": sec["start"], "end": sec["end"]})
    return out


def _span(run, dur):
    return (round(max(0.0, run["start"] - PAD), 2), round(min(dur, run["end"] + PAD), 2))


def wet_items(plan, sections, stem_names, dur):
    items = []
    for stem in stem_names:
        for run in runs(sections, resolve(plan, sections, stem)):
            t0, t1 = _span(run, dur)
            items.append((stem, run["z"], t0, t1))
    return items


def load(session):
    session = Path(session)
    path = session / "plan.json"
    if not path.exists():
        raise SystemExit(f"нет {path}")
    plan = json.loads(path.read_text(encoding="utf-8-sig"))
    sections = sections_mod.load(session)
    dry = {p.stem for p in (session / "dry").glob("*.wav")}
    allowed = {s["id"] for s in sections["sections"]}
    allowed |= {s["label"] for s in sections["sections"] if s.get("label")}
    for name, spec in plan.get("stems", {}).items():
        if name not in dry:
            raise SystemExit(f"plan: неизвестный стем {name}")
        _check_knobs(spec.get("knobs", {}))
        if "wet" in spec and not 0 <= spec["wet"] <= 1:
            raise SystemExit(f"plan: wet вне [0, 1] у {name}")
        for key, sub in spec.get("sections", {}).items():
            if key not in allowed:
                raise SystemExit(f"plan: секция {key!r} не из sections.json")
            _check_knobs(sub.get("knobs", {}))
            if "wet" in sub and not 0 <= sub["wet"] <= 1:
                raise SystemExit(f"plan: wet вне [0, 1] у {name}/{key}")
    master = plan.get("_master", {})
    if set(master) - {"gain_db"}:
        raise SystemExit("plan: _master принимает только gain_db")
    meta = megami_bridge.bank_meta(session)
    for name in dry:
        for p in resolve(plan, sections, name):
            _check_knobs(p["knobs"])
            if not 0 <= p["wet"] <= 1:
                raise SystemExit(f"plan: wet вне [0, 1] у {name}")
            if p["wet"] > 0:
                n = None if meta is None else meta["n"]
                if n is None or not 0 <= int(p["z"]) < n:
                    raise SystemExit(f"plan: z={p['z']} вне банка (n={n})")
    return plan


def _fade(a, b, sec, xf, fs, first, last):
    if xf <= 0:
        return np.ones(b - a)
    t = (np.arange(a, b) + 0.5) / fs
    w = np.ones(b - a)
    if not first:
        w *= np.clip((t - (sec["start"] - xf / 2)) / xf, 0, 1)
    if not last:
        w *= np.clip(((sec["end"] + xf / 2) - t) / xf, 0, 1)
    return w


def _slice_wet(wet_runs, z, a0, b):
    for zz, i0, i1, arr in wet_runs:
        if zz == z and i0 <= a0 and i1 >= b:
            return arr[a0 - i0:b - i0]
    raise SystemExit(f"нет wet-прогона z={z} на сэмплах [{a0}, {b})")


def _load_runs(session, stem, sections, params, n, fs, dur):
    loaded = []
    for run in runs(sections, params):
        t0, t1 = _span(run, dur)
        x, _ = sf.read(megami_bridge.wet_path(session, stem, run["z"], t0, t1),
                       dtype="float32", always_2d=True)
        x = align_wet(x)
        i0 = round(t0 * fs)
        loaded.append((run["z"], i0, i0 + len(x), x))
    return loaded


def _level_match(dry, wet, fs):
    if len(dry) < int(0.4 * fs):
        return 1.0
    ld, lw = meters.stem_lufs(dry, fs), meters.stem_lufs(wet, fs)
    if not (np.isfinite(ld) and np.isfinite(lw)):
        return 1.0
    return 10 ** ((lw - ld) / 20)


def _source(dry, wet_runs, p, a0, b, sec, fs):
    d = dry[a0:b]
    if p["wet"] <= 0:
        return d
    w = _slice_wet(wet_runs, p["z"], a0, b)
    if p["wet"] >= 1:
        return w
    s0 = int(sec["start"] * fs) - a0
    s1 = int(sec["end"] * fs) - a0
    return (1 - p["wet"]) * _level_match(d[s0:s1], w[s0:s1], fs) * d + p["wet"] * w


def render_mix(session, plan, sections, stems, fs, only=None):
    n = len(next(iter(stems.values())))
    dur = n / fs
    xf = plan.get("xfade_s", 0.4)
    out = np.zeros((n, 2), np.float32)
    bus = np.zeros(n, np.float32)
    secs = sections["sections"]
    for stem, x in stems.items():
        if only and stem not in only:
            continue
        dry = np.repeat(render.to_mono(x)[:, None], 2, axis=1).astype(np.float32)
        if len(dry) < n:
            dry = np.pad(dry, ((0, n - len(dry)), (0, 0)))
        params = resolve(plan, sections, stem)
        wet_runs = _load_runs(session, stem, sections, params, n, fs, dur) if any(
            p["wet"] > 0 for p in params) else []
        for k, (sec, p) in enumerate(zip(secs, params)):
            a = max(0, int((sec["start"] - xf / 2) * fs))
            b = min(n, int((sec["end"] + xf / 2) * fs))
            a0 = max(0, a - int(LEAD * fs))
            y = render.apply_knobs(_source(dry, wet_runs, p, a0, b, sec, fs), p["knobs"], fs)
            if y.ndim == 1:
                y = np.stack([y, y], axis=1)
            y = y[a - a0:]
            y = y * _fade(a, b, sec, xf, fs, first=k == 0, last=k == len(secs) - 1)[:, None]
            out[a:b] += y[:b - a]
            send = 10 ** (p["knobs"].get("rev_db", -90.0) / 20)
            if send > 1e-4:
                bus[a:b] += render.to_mono(y[:b - a]) * send
    if bus.any():
        wet = render.reverb(bus, fs).astype(np.float32)
        out = np.concatenate([out, np.zeros((len(wet) - n, 2), np.float32)]) + wet
    gain = plan.get("_master", {}).get("gain_db", 0.0)
    return out * 10 ** (gain / 20)
