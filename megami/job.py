import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEPS = ROOT / ".pydeps"
STUDIO = Path.home() / ".unsloth" / "studio" / "unsloth_studio" / "Scripts" / "python.exe"
CKPT = ROOT / "checkpoints"

FILES = {
    "FxGenerator_public.pt": "https://github.com/SonyResearch/MEGAMI/releases/download/v0/FxGenerator_public.pt",
    "FxProcessor_public.pt": "https://github.com/SonyResearch/MEGAMI/releases/download/v0/FxProcessor_public.pt",
    "CLAP_DA_public-100000.pt": "https://github.com/SonyResearch/MEGAMI/releases/download/v0/CLAP_DA_public.pt",
    "music_audioset_epoch_15_esc_90.14.patched.pt": "https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt",
}


def _boot():
    DEPS.mkdir(exist_ok=True)
    sys.path.insert(0, str(DEPS))
    os.environ["PYTHONPATH"] = str(DEPS) + os.pathsep + os.environ.get("PYTHONPATH", "")
    os.environ["MEGAMI_INFERENCE"] = "1"
    if "--cpu" in sys.argv or not STUDIO.exists():
        return
    if Path(sys.executable).resolve() == STUDIO.resolve():
        return
    try:
        import torch
        if torch.cuda.is_available():
            return
    except Exception:
        pass
    raise SystemExit(subprocess.call([str(STUDIO), str(ROOT / "job.py"), *sys.argv[1:]]))


def _ensure_torchlibrosa():
    try:
        import torchlibrosa  # noqa: F401
    except ImportError:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--target", str(DEPS), "--no-deps", "torchlibrosa"]
        )


def _download(name, url):
    dest = CKPT / name
    CKPT.mkdir(exist_ok=True)
    if "huggingface.co" in url:
        from huggingface_hub import hf_hub_download

        repo, filename = url.split("huggingface.co/")[1].split("/resolve/main/")
        src = Path(hf_hub_download(repo, filename))
        if dest.exists() and dest.stat().st_size == src.stat().st_size:
            return
        if dest.exists():
            dest.unlink()
        try:
            os.link(src, dest)
        except OSError:
            shutil.copyfile(src, dest)
        return
    if dest.exists() and dest.stat().st_size > 0:
        return
    print(f"download {name}")
    part = dest.with_suffix(dest.suffix + ".part")
    subprocess.check_call(["curl.exe", "-L", "--fail", "--retry", "3", "-C", "-", "-o", str(part), url])
    part.replace(dest)


def _margs(a):
    import omegaconf

    return omegaconf.OmegaConf.create(
        {
            "FxGenerator_code": "public",
            "FxProcessor_code": "public",
            "T": a.steps,
            "Schurn": a.schurn,
            "cfg_scale": a.cfg,
        }
    )


def cmd_bank(a):
    import torch
    from inference.inference import Inference

    inf = Inference(method_args=_margs(a), processor=False)
    names, _, x_seg = inf.load_dry(str(a.dry))
    z = inf.generate_bank(x_seg, a.n).cpu()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"stems": names, "z": z}, a.out)
    a.out.with_suffix(".json").write_text(json.dumps({"stems": names, "n": a.n}), encoding="utf-8")
    print(json.dumps({"bank": str(a.out), "stems": names, "n": a.n}), flush=True)


def cmd_render(a):
    import numpy as np
    import soundfile as sf
    import torch
    from inference.inference import Inference

    inf = Inference(method_args=_margs(a), generator=False)
    names, x_dry, _ = inf.load_dry(str(a.dry))
    bank = torch.load(a.bank, weights_only=False)
    jobs = json.loads(Path(a.jobs).read_text(encoding="utf-8"))
    for i, j in enumerate(jobs):
        i0, i1 = round(j["start"] * 44100), round(j["end"] * 44100)
        out = Path(j["out"])
        out.parent.mkdir(parents=True, exist_ok=True)
        print(f"render {i + 1}/{len(jobs)} {j['stem']} z{j['z']} {j['start']}-{j['end']}", flush=True)
        want = i1 - i0
        if j["stem"] not in names or j["stem"] not in bank["stems"]:
            y = np.zeros((want, 2), np.float32)
        else:
            x = x_dry[names.index(j["stem"])][None, :, i0:i1]
            z = bank["z"][j["z"], bank["stems"].index(j["stem"])][None].to(inf.device)
            with torch.no_grad():
                y = inf.apply_effects(x, z)[0].T.cpu().numpy()
            if len(y) < want:
                y = np.pad(y, ((0, want - len(y)), (0, 0)))
            elif len(y) > want:
                y = y[:want]
        part = out.with_suffix(".part.wav")
        sf.write(part, y, 44100, subtype="FLOAT")
        part.replace(out)


def main():
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--steps", type=int, default=100)
    common.add_argument("--schurn", type=float, default=10)
    common.add_argument("--cfg", type=float, default=1.0)
    common.add_argument("--cpu", action="store_true")
    p = argparse.ArgumentParser(description="MEGAMI: банк z и рендер диапазонов")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank", parents=[common])
    b.add_argument("--dry", type=Path, required=True)
    b.add_argument("--n", type=int, default=8)
    b.add_argument("--out", type=Path, required=True)
    r = sub.add_parser("render", parents=[common])
    r.add_argument("--dry", type=Path, required=True)
    r.add_argument("--bank", type=Path, required=True)
    r.add_argument("--jobs", type=Path, required=True)
    args = p.parse_args()

    _ensure_torchlibrosa()
    for name, url in FILES.items():
        _download(name, url)
    os.chdir(ROOT)
    import torch

    if args.cpu:
        torch.cuda.is_available = lambda: False
    if args.cmd == "bank":
        cmd_bank(args)
    else:
        cmd_render(args)


if __name__ == "__main__":
    _boot()
    main()
