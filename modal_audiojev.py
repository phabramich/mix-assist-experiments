"""AudioJev (shlv/AudioJev, ~5B bf16) как /v1/systemone-сервис на Modal.

Деплой:  modal deploy modal_audiojev.py
URL:     https://<workspace>--mix-assist-audiojev-serve.modal.run
Стоп:    modal app stop mix-assist-audiojev
"""
import modal

app = modal.App("mix-assist-audiojev")
vol = modal.Volume.from_name("mix-assist-hf", create_if_missing=True)

img = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1", "git")
    .pip_install("AudioJev-Inference[server] @ git+https://github.com/SihanLv/AudioJev-Inference")
    .env({"HF_HOME": "/hf"})
)


@app.function(image=img, gpu="L4", volumes={"/hf": vol},
              timeout=3600, scaledown_window=120)
@modal.web_server(8000, startup_timeout=1200)
def serve():
    import subprocess
    subprocess.Popen(["audiojev-serve", "--device", "cuda:0",
                      "--host", "0.0.0.0", "--port", "8000"])
