#!/usr/bin/env bash
# MEGAMI standalone deploy: Linux/Windows(WSL) + NVIDIA GPU.
# Ставит репо, зависимости, чекпоинты (~3 ГБ) и все наши патчи.
# Запуск: bash megami_standalone.sh  →  затем: python megami_run.py <папка_со_стемами> [N]
set -e

git clone --depth 1 https://github.com/SonyResearch/MEGAMI.git
cd MEGAMI
python3 -m venv .venv && source .venv/bin/activate
pip install -U pip
# audioop-lts нужен только py>=3.13, triton — только Linux; выкидываем оба при ошибке
pip install -r requirements.txt || pip install $(grep -v 'audioop-lts\|triton' requirements.txt | tr '\n' ' ')

mkdir -p checkpoints
wget -q https://github.com/SonyResearch/MEGAMI/releases/download/v0/CLAP_DA_public.pt     -O checkpoints/CLAP_DA_public.pt
wget -q https://github.com/SonyResearch/MEGAMI/releases/download/v0/FxGenerator_public.pt -O checkpoints/FxGenerator_public.pt
wget -q https://github.com/SonyResearch/MEGAMI/releases/download/v0/FxProcessor_public.pt -O checkpoints/FxProcessor_public.pt
wget -q https://huggingface.co/lukewys/laion_clap/resolve/main/music_audioset_epoch_15_esc_90.14.pt -O checkpoints/music_audioset_epoch_15_esc_90.14.patched.pt
wget -q https://huggingface.co/yytung/fxencoder-plusplus/resolve/main/fxenc_plusplus_default.pt      -O checkpoints/fxenc_plusplus_default.pt

# конфиг FxEncoder++ ищется рядом с чекпоинтом — берём их st_ito-конфиг
cp utils/st_ito/config.yaml checkpoints/config.yaml
# конфиг ждёт имя с суффиксом итерации
ln -sf CLAP_DA_public.pt checkpoints/CLAP_DA_public-100000.pt

# CLAP-чекпоинт надо пропатчить: выкинуть position_ids из state_dict
python - <<'EOF'
import torch, numpy as np
torch.serialization.add_safe_globals([np._core.multiarray.scalar])
p = "checkpoints/music_audioset_epoch_15_esc_90.14.patched.pt"
try:
    sd = torch.load(p, map_location="cpu")
except Exception:
    sd = torch.load(p, map_location="cpu", weights_only=False)
def strip(d):
    for k in list(d):
        if isinstance(k, str) and k.endswith("embeddings.position_ids"):
            del d[k]; print("removed", k)
        elif isinstance(d[k], dict):
            strip(d[k])
strip(sd); torch.save(sd, p)
EOF
echo "== OK: python megami_run.py <dir_with_wavs> [num_samples]"
