"""Самопроверка EQ-математики и нормализации. Запуск: .venv/bin/python selftest.py"""
import numpy as np
import pyloudnorm

import meters
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

import plan as planmod

# кроссфейд двух секций с постоянным dry не меняет сигнал
secs = {"duration_s": 1.0, "sections": [
    {"id": "s01", "start": 0.0, "end": 0.5, "label": "a", "cluster": "A"},
    {"id": "s02", "start": 0.5, "end": 1.0, "label": "b", "cluster": "A"},
]}
tone = sine(220)
stems = {"a": tone}
pl = {"xfade_s": 0.1, "stems": {"a": {"wet": 0}}}
y = planmod.render_mix(".", pl, secs, stems, FS)
ref = np.repeat(render.to_mono(tone)[:, None], 2, axis=1)
assert np.max(np.abs(y - ref)) < 1e-5, "кроссфейд изменил постоянный dry"

# wet=0 и пустые ручки = сумма моно-дублей
stems2 = {"a": sine(220) * 0.1, "b": sine(440) * 0.05}
pl2 = {"stems": {"a": {"wet": 0}, "b": {"wet": 0}}}
y2 = planmod.render_mix(".", pl2, secs, stems2, FS)
ref2 = sum(np.repeat(render.to_mono(x)[:, None], 2, axis=1) for x in stems2.values())
assert np.allclose(y2, ref2, atol=1e-5), "сборка dry разошлась с суммой"

# diffuse_db растёт вместе с долей реверберации
rng = np.random.default_rng(0)
bursts = np.zeros((FS * 3, 1), np.float32)
for i in range(5):
    a = int((0.15 + i * 0.55) * FS)
    bursts[a:a + int(0.2 * FS)] = rng.standard_normal((int(0.2 * FS), 1)).astype(np.float32) * 0.2
rev = render.reverb(bursts[:, 0], FS).astype(np.float32)
base = np.pad(np.repeat(bursts, 2, axis=1), ((0, len(rev) - len(bursts)), (0, 0)))
scores = [meters.space(base, base + g * rev, FS)["diffuse_db"] for g in (0.0, 0.3, 1.0)]
assert scores[0] < scores[1] < scores[2], f"diffuse_db не растёт: {scores}"
mono = np.repeat(base.mean(axis=1)[:, None], 2, axis=1)
assert meters.space(base, mono, FS)["width_db"] < -60, "моно не дало width_db < -60"

print("selftest OK")
