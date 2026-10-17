"""MEGAMI inference: python megami_run.py <папка_с_wav> [N_сэмплов].
Положить в корень клонированного MEGAMI, стемы — WAV 44.1кГц, до 14 штук.
Выход: <папка>/run/MEGAMI_inference_sample*.wav"""
import sys, omegaconf

sys.path.insert(0, ".")
from inference.inference import Inference  # noqa: E402

directory = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
args = omegaconf.OmegaConf.create({
    "FxGenerator_code": "public", "FxProcessor_code": "public",
    "T": 100, "Schurn": 10, "cfg_scale": 1.0})
Inference(method_args=args).run_inference_single_song(
    directory=directory, num_samples=n, exp_name="run")
