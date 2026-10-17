"""Music Flamingo 2601 (~8B, bf16) на Modal — архитектор диагнозов.

Деплой:  modal deploy modal_mf.py
Вызов:   POST https://<workspace>--mix-assist-mf-mf.chat.modal.run  — нет,
         эндпоинт даётся в выводе deploy (modal fastapi_endpoint).
Тело:    {"audio_base64": <wav>, "prompt": <str>, "max_new_tokens": 400}
Ответ:   {"text": <генерация>}
"""
import modal

app = modal.App("mix-assist-mf")
vol = modal.Volume.from_name("mix-assist-hf", create_if_missing=True)

img = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "ffmpeg")
    .pip_install("torch", "transformers>=5.20", "accelerate",
                 "soundfile", "librosa", "fastapi")
    .env({"HF_HOME": "/hf"})
)


@app.cls(image=img, gpu="L4", volumes={"/hf": vol},
         timeout=3600, scaledown_window=120)
class MF:
    @modal.enter()
    def load(self):
        import torch
        from transformers import (AutoProcessor,
                                  MusicFlamingoForConditionalGeneration)
        mid = "nvidia/music-flamingo-2601-hf"
        self.processor = AutoProcessor.from_pretrained(mid)
        self.model = MusicFlamingoForConditionalGeneration.from_pretrained(
            mid, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa").to("cuda")

    @modal.fastapi_endpoint(method="POST")
    def chat(self, request: dict):
        import base64
        import os
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            f.write(base64.b64decode(request["audio_base64"]))
            path = f.name
        try:
            conv = [{"role": "user", "content": [
                {"type": "text", "text": request["prompt"]},
                {"type": "audio", "path": path}]}]
            inputs = self.processor.apply_chat_template(
                conv, tokenize=True, add_generation_prompt=True,
                return_dict=True).to("cuda")
            inputs["input_features"] = inputs["input_features"].to(
                self.model.dtype)
            out = self.model.generate(
                **inputs, max_new_tokens=request.get("max_new_tokens", 400))
            text = self.processor.batch_decode(
                out[:, inputs.input_ids.shape[1]:],
                skip_special_tokens=True)[0]
            return {"text": text}
        finally:
            os.unlink(path)
