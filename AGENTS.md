# mix-assist

Личный ресёрч ради интереса (не продукт, не статья): агент сведения мультитрека.

Роли:
- Music Flamingo — архитектор (System 2);
- Jev-модели (Sori-1B-MCQ и др.) — рулевой (System 1).

План и все рассуждения — `docs/plan.md`, читать первым. Документы пишем на русском.

## Инварианты

- Перед любым сравнением кандидатов — выравнивание по LUFS.
- Не оценивать тем же судьёй, по которому оптимизировали: нужен hold-out судья + уши.
- Train/test делим по песням.
- Модели слышат 16 кГц (выше 8 кГц слепы) и, скорее всего, моно. Моно-даунмикс моделям делаем сами.
- Вывод MF — недоверенный: проверять по словарю проблем и списку дорожек.
- Замеры Sori — в fp32, dtype указывать рядом с результатом.
- Каждая модель — отдельный сервис со своим окружением; общий клиент Jev/SystemOne API (`judge.py`: `ask` = d1-omni/llama.cpp, `ask_audiojev` = AudioJev-формат).
- 2AFC-склейка «половина|бип|половина» у Jev-моделей сломана лейбл-перекосом (audio не влияет) — сравнивать T1-score по отдельным клипам, см. `experiments/002-003`.

## Железо

- Mac — разработка и оркестратор.
- Windows + AMD RX 6600M — оркестратор и MEGAMI (ROCm через Unsloth Studio). NVIDIA 8 ГБ — Sori, d1-omni, небольшие LoRA, если машина есть.
- Modal (Starter, $30/мес) — MF, AudioJev, clef-omni, LoRA для MF.

## Интерфейс MVP

CLI + файлы.

## Данные и окружение

- Стемы для старта: `data/MUSDB18/MUSDB18-7/` (144 отрывка по ~6.8 с,
  stems через пакет `musdb`; подробности — `data/README.md`).
- Python-окружение: `.venv` (uv), пакеты ставить через `uv pip install`.
- Локальный судья: `llama-server -hf LiquidAI/d1-omni-600M-GGUF:Q8_0` на :8080 — глух к уровням (`experiments/001`).
- Облачный судья: AudioJev на Modal — `modal deploy modal_audiojev.py`, слышит уровень вокала, JND ~3–4.5 дБ на один запрос (`experiments/002-003`).
- Stage-1 контроллер `fix.py`: бисекция по score/noul сходится на «идеале модели», но идеал ≠ референс (смещение 1–8 дБ, зависит от трека и формулировки) — `experiments/004-006`.
- EQ (bell 300 Гц ±9 дБ) AudioJev почти не слышит — `experiments/004-006`.
- Music Flamingo на Modal — `modal deploy modal_mf.py`; НЕ слышит уровни вокала (−6 дБ = «balanced», clean≡spoiled побайтово) — семантика есть, перцепции нет, `experiments/007`.
- Ключевой вывод: перцепция = Jev + DSP-метры (`meters.py`); рассуждение/архитектор = текстовый LLM поверх отчёта (`architect.fuse`, модель `antigravity/claude-sonnet-4-6`); MF — художественные заметки по аудио.
- **Сквозной цикл работает**: порча → метры+MF → Сонет → AudioJev-бисекция → +1.12 дБ остаток (`experiments/008`).
- **Архитектура — инструменты для агента**, не жёсткий оркестратор: `mixctl.py` (tracks/meters/caption/jev/fix/render/compare → JSON). Внешний агент-субагент протестирован (`experiments/009`).
- **MCP**: `mcp_server.py` (stdio, `mcp<2` FastMCP — в 2.x переименован в MCPServer). 7 тулзов — тонкие обёртки над mixctl, проверены реальным клиентом. Для Ableton-моста — та же поверхность, рендер заменяется вызовами DAW.
- **Сессии-папки**: `--session <dir>` грузит стемы из WAV (формат для своих мультитреков). Пилот v2 на запечённой порче: агент тройно подтвердил «vocals buried», починил +7.1 дБ, верифицировал compare/meters, не тронул неподтверждённое (`experiments/009` апдейт).
- Открытые UX-долги от агента: нет caption/jev на один стем; неизвестно, какой срез слышит судья; noul — голый скаляр без семантики шкалы.
- Ландшафт подходов: `docs/landscape.md` + `docs/path-b.md`. Коротко: «сумматор» = фронтир поля (MEGAMI поднимается на L4, веса открыты NC-SA; FxNorm-automix MIT под K=4; RoEx Tonn — единственный API с params-out, 1000 бесплатных кредитов); агент-в-DAW почти пустой; наша связка агент+типизированный судья — незанятая клетка.
- План наград по SPHERE (arXiv:2608.30559): Qwen2.5-Omni-3B → JSON-параметры → вычислимые награды (межстемовая маскировка, LUFS/спектр vs референс, role-уровни); NeMo-RL умеет audio-GRPO на этой модели. Гибрид «добавь струнные»: Stem-JEPA (gap detection) → ACE-Step 1.5 lego (генерация стема по контексту).
- MEGAMI живёт в `megami/` (vendored SonyResearch/MEGAMI, CC BY-NC-SA) и считается локально на Windows + AMD RX 6600M питоном Unsloth Studio. Оркестратор — `.venv\Scripts\python`. Секционный пайплайн — раздел ниже. Выводы eval'ов: `experiments/010`+`011` — 2/4 сэмпла попадают в баланс инженера, нужен внешний селектор. Модель ≤14 стемов, моно ок (дублируется), отдавать dry — EQ/comp/reverb/pan/Haas она делает сама, тюн/исполнительские FX печатать заранее.
- Честный eval на MedleyDB Sample (`experiments/011`): Phoenix_ScotchMorris, flute −6дБ. 2/4 сэмпла MEGAMI попали в баланс инженера (2.22–2.24 vs ref 2.26). Агентный fix утёк до +11.6дБ: target 2.48 откалиброван под другую сессию — «правильный баланс» имеет разный score на разном материале, target надо калибровать per-session. `render_for_judge` теперь режет окно 12с вокруг пикового RMS (cumsum) — иначе судья падает по max_prompt_tokens.
- Сырой пилот `experiments/012`: агент свёл 5 моно-каналов Rainfall (сам определил роли f0-анализом, починил «гармония громче лида»). Критичные находки: `fix` не резолвит не-семантические имена стемов (vox_a → runaway; нужны слова или описание в STATE), судья EQ-слеп (±6дБ → дельта <0.1), приминг-галлюцинации (слышит текст вопроса при заглушенном стеме, conf 0.92), окно 12с = только припев, нет master gain.
- Полная консоль `experiments/013`: добавлены `pan` (−1..1), `haas_ms` (0..30), `rev_db` (send в общий конволюционный ревер IR=шум+эксп. затухание, wet −12дБ от send), `_master`. Стемы сессии переименованы семантически (vocal1-3, acoustic/electric_guitar) — иначе fix/jev не резолвят. Баг моно+стерео суммы пофикшен. Ручной прогон: −21.8 LUFS, corr 0.90. Главная слепая зона: судья моно 12с — пространство не верифицируемо им вообще.
- Файл-режим для DAW-ценария: `mixctl meters/caption/jev --file x.wav`, `compare --file A --file2 B`, окно `--from_s/--to_s` (иначе 12с вокруг пикового RMS); jev/caption/compare возвращают `window_s` — срез, который реально слышал судья. MCP: +`measure_file/describe_file/listen_file/compare_files`, у стемовых тулзов параметр `session` (папка WAV).
- Перед экспериментами гонять `.venv\Scripts\python selftest.py`.

## Секционное сведение (MEGAMI + консоль)

Сессия — `sessions/<имя>/`: `dry/*.wav`, `sections.json`, `plan.json`, `megami/bank.pt` + `bank.json`, кэш `megami/wet/<stem>/z<k>_<t0>_<t1>.wav`.

Порядок: `session_init` → `detect_sections` (DSP; Music Flamingo подписывает кластеры) → агент ставит `label` и зовёт `save_sections` → `megami_bank` → `megami_candidates` по стему и label → `save_plan` → `render_plan_estimate` → `render_plan`. Долгие три команды — фоновые, статус через `job_status`. Повторный рендер после правки ручек идёт из кэша.

`plan.json`: дефолт `z_default` / `wet=1`, затем поля стема, затем `sections[label]`, затем `sections[id]`. `wet` 0 — сухой сигнал, 1 — выход MEGAMI. Стыки — линейный кроссфейд `xfade_s`.

Интент → рычаги:

| Интент | Рычаги |
|---|---|
| сухо / близко | z с минимальным `diffuse_db`; `wet` 0.3–0.6; `rev_db` −90 |
| большое пространство | z с максимальным `diffuse_db`; `rev_db` −16…−10; при желании `haas_ms` 8–15 |
| шире | больший `width_db` у z, `pan`, `haas_ms` |
| вперёд / громче | `gain_db`, `pres_db` +1…+3 |

Пространство проверять только `measure_space`. Баланс — `measure` и `jev_ask` с окном секции (`from_s`/`to_s`). Пространство судьёй не судить.
- LLM-ключи для архитектора: opencode-роутер helium `https://router.y7.hk/v1` (OpenAI-совместимый), ключ читать из `~/.local/share/opencode/auth.json` (helium.key); рабочие модели: `antigravity/claude-sonnet-4-6` (дефолт архитектора), `antigravity/gemini-3.8-flash-low`; `vectide3/*`, `kimi`, `x/*`, `ds/*` по этому ключу недоступны/без баланса.
