"""Самопроверка EQ-математики и нормализации. Запуск: .venv/bin/python selftest.py"""
import numpy as np
import pyloudnorm

import render

FS = 44100


def rms_db(x):
    return 20 * np.log10(np.sqrt(np.mean(x**2)))


def sine(f, dur=1.0, fs=FS):
    t = np.arange(int(fs * dur)) / fs
    return np.stack([np.sin(2 * np.pi * f * t)] * 2, axis=1)


x300 = sine(300)
x3k = sine(3000)
x10k = sine(10000)

# колокол +6 дБ на 300 Гц: на частоте ≈+6 дБ, вдали ≈0 дБ
got = rms_db(render.bell(x300, 300, 6.0, 1.5, FS)) - rms_db(x300)
assert abs(got - 6.0) < 0.5, f"bell 300Hz: {got:+.2f} дБ"
got = rms_db(render.bell(x3k, 300, 6.0, 1.5, FS)) - rms_db(x3k)
assert abs(got) < 0.5, f"bell вне полосы: {got:+.2f} дБ"

# полка -6 дБ на 5к: на плато (10к) ≈-6 дБ, внизу ≈0 дБ
got = rms_db(render.high_shelf(x10k, 5000, -6.0, FS)) - rms_db(x10k)
assert -7.0 < got < -5.0, f"shelf 10k: {got:+.2f} дБ"
got = rms_db(render.high_shelf(x300, 5000, -6.0, FS)) - rms_db(x300)
assert abs(got) < 0.3, f"shelf на 300 Гц: {got:+.2f} дБ"

# HPF 100 Гц: синус 50 Гц давится сильно, 1 кГц проходит
got = rms_db(render.highpass(sine(50), 100, FS)) - rms_db(sine(50))
assert got < -6, f"hpf 50Hz: {got:+.2f} дБ"
got = rms_db(render.highpass(sine(1000), 100, FS)) - rms_db(sine(1000))
assert abs(got) < 0.2, f"hpf 1kHz: {got:+.2f} дБ"

# LUFS-нормализация попадает в цель
stems = {"a": sine(220) * 0.1, "b": sine(440) * 0.05}
m = render.mix(stems, {}, FS)
n = render.lufs_normalize(m, FS, -23.0)
loud = pyloudnorm.Meter(FS).integrated_loudness(np.asarray(n, dtype=np.float64))
assert abs(loud - -23.0) < 0.3, f"LUFS: {loud:.2f}"

# gain ручка: +6 дБ на стеме ≡ тот же микс с умноженным стемом
m2 = render.mix(stems, {"a": {"gain_db": 6.0}}, FS)
m_ref = render.mix({"a": stems["a"] * 10 ** 0.3, "b": stems["b"]}, {}, FS)
assert np.allclose(m2, m_ref, atol=1e-5)

print("selftest OK")
