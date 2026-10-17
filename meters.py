"""DSP-метры: физический отчёт о миксе для архитектора (текст).

LUFS по стемам, доля энергии по полосам, crest, маскировка (перекрытие
полосовой энергии стемов). Всё детерминированно, без моделей.
"""
import numpy as np
import pyloudnorm
import scipy.signal as sig

BANDS = [(20, 120, "sub"), (120, 600, "lowmid"), (600, 3000, "mid"),
         (3000, 8000, "pres"), (8000, 22050, "air")]


def stem_lufs(x, fs):
    return pyloudnorm.Meter(fs).integrated_loudness(np.asarray(x, np.float64))


def band_share(x, fs, lo, hi):
    """Доля энергии стема в полосе (0..1)."""
    sos = sig.butter(4, [lo, min(hi, fs / 2 - 1)], "bandpass", fs=fs,
                     output="sos")
    y = sig.sosfilt(sos, x, axis=0)
    tot = (x ** 2).mean() + 1e-12
    return float((y ** 2).mean() / tot)


def crest_db(x):
    peak = np.abs(x).max() + 1e-12
    return float(20 * np.log10(peak / (np.sqrt((x ** 2).mean()) + 1e-12)))


def report(stems, fs):
    """stems: {name: (n, ch)} -> dict + текст для промпта архитектора."""
    stereo_mix = np.sum(np.stack(list(stems.values())), axis=0)
    mix_lufs = stem_lufs(stereo_mix, fs)
    mono = {n: x.mean(axis=1) if x.ndim == 2 else x for n, x in stems.items()}
    mix = stereo_mix.mean(axis=1) if stereo_mix.ndim == 2 else stereo_mix

    rows, data = [], {}
    for n, x in mono.items():
        lufs = stem_lufs(x, fs)
        shares = {nm: band_share(x, fs, lo, hi) for lo, hi, nm in BANDS}
        cr = crest_db(x)
        data[n] = {"lufs": lufs, "delta_vs_mix": lufs - mix_lufs,
                   "bands": shares, "crest_db": cr}
        rows.append(f"  {n:8} LUFS {lufs:+6.1f} ({lufs - mix_lufs:+5.1f} vs mix) "
                    f"| sub {shares['sub']:.2f} lowmid {shares['lowmid']:.2f} "
                    f"mid {shares['mid']:.2f} pres {shares['pres']:.2f} "
                    f"air {shares['air']:.2f} | crest {cr:.1f} dB")
    # маскировка: кто с кем пересекается сильнее всего в lowmid/pres
    mask = []
    names = list(mono)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            for lo, hi, nm in BANDS:
                o = min(data[a]["bands"][nm], data[b]["bands"][nm])
                if o > 0.25:
                    mask.append(f"  {a}<->{b} в {nm}: {o:.2f}")
    text = (f"MIX LUFS {mix_lufs:+.1f} | crest {crest_db(mix):.1f} dB\n"
            + "\n".join(rows)
            + ("\nМАСКИРОВКА (общая энергия полосы >0.25):\n" + "\n".join(mask)
               if mask else ""))
    return {"mix_lufs": mix_lufs, "stems": data, "text": text}
