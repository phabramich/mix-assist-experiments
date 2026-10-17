# Путь B и гибриды — детальный ресёрч (второй заход)

Свежий срез. Четыре направления: сумматор-стек Sony/QMUL, коммерческие API,
SPHERE/RLVR-перенос на наш цикл, генеративная аранжировка.

## 1. Сумматор — что реально поднять руками

| Система | Веса | Лицензия | Вход | Выход | Поднять? |
|---|---|---|---|---|---|
| **MEGAMI** (Sony AI, ICASSP'26) | ✅ `FxGenerator_public/FxProcessor_public/CLAP_DA_public` + CLAP + FxEnc++ (~350M всего) | CC BY-NC-SA | папка dry wav 44.1k, ≤14 стемов, сегменты 11.9с | стерео-микс + wet-стемы; **сэмплирует варианты микса** (диффузия над FX-эмбеддингами) | **да, на L4 — `run_inference.sh`** |
| FxNorm-automix (Sony, ISMIR'22) | ✅ в репо (`ours_S_*`) | MIT | ровно K=4 stems — наш случай | микс waveform | да, но torch1.9-эпоха (legacy venv/CPU) + нужны IR реверба <1.5с |
| Diff-MST (QMUL) | ⚠️ чекпоинт не найден публично (issue #27) | NC-SA | стемы + **референс-песня** | интерпретируемые console params (gain/pan/EQ4/comp, без реверба) | тренировка на MedleyDB+MixingSecrets (~190M, дни GPU) |
| GRAFx-prune (SNU/Sony) | `pip install grafx` | — | dry stems + целевой микс | sparse mixing graph + params = **псевдо-разметка миксов** (1069 песен обработали) | да; нужен существующий целевой микс |
| **Diff2Mix** (QMUL, ISMIR'26) | нет | — | стемы (+optional ref) | **диффузия → параметры консоли** (MEGAMI-идея с интерпретируемым выходом) | следить |

MEGAMI-ускорение: инъекция `z_ref` (FxEncoder++ от референс-стема) прямо в
FxProcessor = per-track style transplant, минуя генератор. Text-conditioning
не встроен, но CLAP-текст-маппер — открытая модификация.

## 2. Коммерческое — RoEx как оракул

- **RoEx Tonn** — ЕДИНСТВЕННЫЙ API «стемы → параметры»: `retrieveFXSettings`
  возвращает per-track JSON (gain/pan/6-band EQ/comp), `applyaudioeffects`
  перерендеривает чужие параметры. ~$2.5/микс, **1000 бесплатных кредитов**
  при регистрации, OpenAPI-доки есть. Плюс C++ SDK (offline, deterministic,
  `settingsOnly` mode, 6 мес бесплатно стартапам).
- Интерналс: маскинг-минимизация Ronan/Reiss (патент WO2024134196A1 покрывает
  психоакустическую межтрековую метрику маскировки — наша meters.py-мешковая
  реализация это обходит, но при продуктовизации читать claims).
- Остальные API — только мастеринг/сепарация (Music AI реселлит RoEx).
- **DAWproject** (Bitwig+PreSonus): зрелый XML-формат, нативные
  BuiltinDevice EQ/Compressor/Limiter; писатель `roex-audio/dawproject-py`.
  **Ableton не читает** → для Ableton нужен `.als` (gzipped XML — можно
  генерить напрямую) или живое управление через AbletonOSC.

## 3. SPHERE — перенос на наш цикл

Полный рецепт из paper (Meta×QMUL, arXiv:2608.30559):
- Студент **Qwen2.5-Omni-3B** предсказывает JSON {azimuth,elevation,gain,role}
  на стем; рендер через HRTF; 6 наград считаются **по параметрам и loudness
  стемов — без рендера аудио**. SFT (LoRA r8, фильтр «все sub-реварды ≥0.5»)
  → GRPO (G=4, ~19k роллаутов, 8×H100/24ч). Студент побил учителя
  (Gemini 2.5 Pro) 4.98 vs 4.55; люди предпочли 76.8%.
- Анти-хак дизайн: взаимоограничивающие метрики (centering×spread),
  unweighted-термы, conjunction-фильтр; выбрасывание любой пары → деградация.
- Поддержка: **NeMo-RL официально имеет audio-GRPO для Qwen2.5-Omni-3B**
  (есть пример R1-AQA repro); veRL-omni; ms-swift с custom reward plugin.
- Прецеденты RLVR-аудио: R1-AQA, Omni-R1, SARI, EchoInk-R1 — GRPO на
  верифицируемых метках работает на аудио-LLM; Step-Audio-R1.5 предупреждает
  про «verifiable reward trap» (фичи ≠ вкус, нужен holdout-человек).

Перенос к нам: та же форма — модель выдаёт JSON-параметры (наш словарь ручек
= уже готовый action space), рендер = наш `render.py`, награда = вычислимая:
**R1 межстемовая маскировка** (наш meters уже считает пары; полный
Wilson&Fazenda masking-score ρ=0.838), **R2 LUFS/band-энергии/M-S/creat vs
референс** (auraloss MR-STFT), R3 role-уровни à la SPHERE, R4 валидность
JSON (verifiable). Студент: Qwen2.5-Omni-3B (бэкбон AudioJev! — можно либо
его, либо пост-тренить сам AudioJev как судью).

## 4. Гибрид «добавь струнные» — реален сегодня

Генерация стема по контексту микса — признанная задача, есть открытое:

| Модель | Что | Открыто? |
|---|---|---|
| **ACE-Step 1.5** (MIT) | задачи `lego`/`complete`: стем по аудио-контексту + тексту, 48k стерео | ✅ веса, diffusers |
| STAGE (MusicGen FT) | контекст + текст, темпо-лок | ✅ веса |
| Instruct-MusicGen | «add drums» инструкциями | ✅ |
| MusicGen-Stem (Meta) | 3 потока bass/drums/other | ⚠️ regen API сломан |
| StemGen, Diff-A-Riff (Sony), JEN-1 Composer | идеальные по формулировке | ❌ только paper |

**«Чего не хватает» — тоже есть чем:** Stem-JEPA (Sony CSL, код открыт) —
по эмбеддингу контекста предсказывает совместимый недостающий стем, zero-shot
вариант работает с произвольными именами инструментов; + роль-классификаторы
(main melody/pad/riff/...) + жанровые нормы (TiPS/Discogs-приоры) + наши
маскировочные карты (где спектральное место свободно).

**Петля-гибрид:** агент видит «пустая середина» (метры/MF) → Stem-JEPA ранжирует
кандидатов → ACE-Step lego генерит стем → mixctl вмиксовывает → AudioJev судит.
Это и миксинг, и аранжировка одним циклом.

## 5. MixParams для файнтюна — вердикт

Не как основной субстрат: 2345 строк single-track параметров, без
межтрекового контекста и без вкусовой правды. Место — либо как SFT-сиды для
формата вывода, либо как prior для «типичных» настроек по instrument_subtype.
Наше обучающее ядро — рендеримые миксы + вычислимые награды (SPHERE-формат)
и/или A/B-пары для калибровки судьи.
