# 008 — Сквозной цикл MVP: метры + MF → архитектор → Jev → починено

Дата: 2026-10-16. Трек 6 (South Of The Water), скрытая порча: вокал −6 дБ.

## Цепочка

1. `meters.py`: vocal LUFS −29.9 (−11.9 vs mix) — физика порчи видна
   сразу; маскировка vocals↔other в lowmid, drums↔bass в sub.
2. MF (художник): «aggressive metal/metalcore, double-kick, thick
   bass, wide stereo» — атмосфера верна, вокал снова не замечен.
3. Архитектор `antigravity/claude-sonnet-4-6` (helium): фьюзит отчёты →
   `{"track":"vocals","knob":"gain_db","direction":"up"}` — правильно,
   с обоснованием по цифрам.
4. AudioJev бисекция: коррекция **+7.12 дБ**, остаток **+1.12 дБ** —
   то есть до своего «идеала» (reproduces 005 result).

**Первая работающая пара архитектор→рулевой.** Механика доказана;
остаточная ошибка = смещение вкуса модели, снимается калибровкой/
дообучением.

## Детали инфраструктуры

- helium `router.y7.hk/v1`: ключ валиден только для `ds/*`, `antigravity/*`,
  `xiaomi/*`; `vectide3/*`/`kimi` = invalid_api_key, `x/grok` = denied,
  `ds/*` сейчас Insufficient Balance → дефолт архитектора
  `antigravity/claude-sonnet-4-6` (чистый JSON), запасной
  `antigravity/gemini-3.8-flash-low`.
- reasoning-модели (deepseek-pro) жгут max_tokens в reasoning_content —
  `content` может прийти пустым; для JSON-выдачи брать не-рассуждающие
  или большой лимит.
- `architect.fuse(meter_text, notes)` + `FUSE_PROMPT` — формат входа
  зафиксирован (важно для будущего датасета дообучения).
