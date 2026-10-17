"""Клиент Jev/SystemOne API.

d1-omni (llama-server): POST /v1/systemone, аудио — data URL в поле files,
один клип до 30 с на запрос. Совместимые серверы (Sori /v1/ask и др.)
подключаются сменой base_url — унификация клиента позже, когда будет
второй работающий бэкенд.
"""
import base64
import io
import json
import urllib.error
import urllib.request

import soundfile as sf


def wav_bytes(x, fs):
    buf = io.BytesIO()
    sf.write(buf, x, fs, format="WAV", subtype="FLOAT")
    return buf.getvalue()


def wav_data_url(x, fs):
    return "data:audio/wav;base64," + base64.b64encode(wav_bytes(x, fs)).decode()


def _post(server, body, timeout):
    req = urllib.request.Request(
        server.rstrip("/") + "/v1/systemone",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"judge {e.code}: {e.read()[:300]!r}") from e


def ask(server, state, questions, clips=None, fs=16000, timeout=120):
    """d1-omni/llama.cpp. clips — список моно-массивов."""
    body = {"state": state, "questions": questions}
    if clips:
        body["files"] = [wav_data_url(x, fs) for x in clips]
    return _post(server, body, timeout)


def ask_audiojev(server, clip, context, questions, fs=16000, timeout=300):
    """AudioJev-Inference: state = только audio_base64 (один клип),
    текстового state нет — контекст сливаем в instructions вопросов."""
    qs = {k: {**q, "instructions": context + " " + q["instructions"]}
          for k, q in questions.items()}
    body = {"state": {"audio_base64": base64.b64encode(wav_bytes(clip, fs)).decode()},
            "questions": qs}
    return _post(server, body, timeout)
