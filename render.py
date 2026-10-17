"""Рендер микса из стемов: gain + EQ -> сумма -> LUFS -> моно -> 16 кГц.

Модели-судьи всё равно слушают 16 кГц (см. docs/plan.md 2.1) — даунмикс
и ресемпл делаем сами, чтобы поведение было предсказуемым.
"""
import numpy as np
import pyloudnorm
import scipy.signal as sig
import soundfile as sf

JUDGE_FS = 16000

DEFAULT_KNOBS = {
    "gain_db": 0.0,
    "hpf_hz": 20.0,
    "lowmid_f": 350.0,
    "lowmid_db": 0.0,
    "pres_f": 3000.0,
    "pres_db": 0.0,
    "shelf_db": 0.0,  # полка на 5 кГц
}


def _lfilter(b, a, x):
    return sig.lfilter(b / a[0], a / a[0], x, axis=0)


def bell(x, f0, gain_db, q, fs):
    if gain_db == 0:
        return x
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
    return _lfilter(b, a, x)


def high_shelf(x, f0, gain_db, fs, s=0.7):
    if gain_db == 0:
        return x
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / 2 * np.sqrt((A + 1 / A) * (1 / s - 1) + 2)
    c = np.cos(w0)
    r = 2 * np.sqrt(A) * alpha
    b = np.array([
        A * ((A + 1) + (A - 1) * c + r),
        -2 * A * ((A - 1) + (A + 1) * c),
        A * ((A + 1) + (A - 1) * c - r),
    ])
    a = np.array([
        (A + 1) - (A - 1) * c + r,
        2 * ((A - 1) - (A + 1) * c),
        (A + 1) - (A - 1) * c - r,
    ])
    return _lfilter(b, a, x)


def highpass(x, fc, fs):
    if fc <= 20:
        return x
    sos = sig.butter(2, fc, "highpass", fs=fs, output="sos")
    return sig.sosfilt(sos, x, axis=0)


def apply_knobs(x, knobs, fs):
    unknown = set(knobs) - set(DEFAULT_KNOBS)
    if unknown:
        raise ValueError(f"неизвестные ручки: {unknown}")
    x = x * 10 ** (knobs.get("gain_db", 0.0) / 20)
    x = highpass(x, knobs.get("hpf_hz", 20.0), fs)
    x = bell(x, knobs.get("lowmid_f", 350.0), knobs.get("lowmid_db", 0.0), 1.5, fs)
    x = bell(x, knobs.get("pres_f", 3000.0), knobs.get("pres_db", 0.0), 1.5, fs)
    x = high_shelf(x, 5000.0, knobs.get("shelf_db", 0.0), fs)
    return x


def mix(stems, params, fs):
    """stems/params: {имя: массив (n, ch)} / {имя: knobs}."""
    out = None
    for name, x in stems.items():
        y = apply_knobs(x, params.get(name, {}), fs)
        out = y if out is None else out + y
    return out


def lufs_normalize(x, fs, target=-23.0):
    loud = pyloudnorm.Meter(fs).integrated_loudness(np.asarray(x, dtype=np.float64))
    return x * 10 ** ((target - loud) / 20)


def to_mono(x):
    return x.mean(axis=1) if x.ndim == 2 else x


def resample(x, fs, fs_out=JUDGE_FS):
    if fs == fs_out:
        return x
    g = np.gcd(fs, fs_out)
    return sig.resample_poly(x, fs_out // g, fs // g, axis=0)


def render_for_judge(stems, params, fs, target_lufs=-23.0, max_s=12.0):
    """Полная цепочка: микс -> LUFS -> моно -> 16 кГц float32.

    Длинные миксы режем в окно max_s вокруг пикового RMS — судья упирается
    в max_prompt_tokens, а JND не требует целой песни.
    """
    m = to_mono(lufs_normalize(mix(stems, params, fs), fs, target_lufs))
    n = int(max_s * fs)
    if len(m) > n:
        c = np.concatenate(([0.0], np.cumsum(m.astype(np.float64) ** 2)))
        i = int(np.argmax(c[n:] - c[:-n]))
        m = m[i:i + n]
    return resample(m.astype(np.float32), fs)


def save_wav(path, x, fs):
    sf.write(str(path), x, fs, subtype="FLOAT")
