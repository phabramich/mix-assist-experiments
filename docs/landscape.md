# Ландшафт подходов к AI-микшированию

Сводка ресёрча 4 субагентов (2026-…, текущий срез). Поле = «Intelligent Music
Production» (De Man, Stables & Reiss 2019; библиография
[csteinmetz1/AutomaticMixingPapers](https://github.com/csteinmetz1/AutomaticMixingPapers)).

## Путь B: «умный сумматор» (one-shot)

Реален и является фронтиром поля — почти все серьёзные системы именно такие:

| Система | Что предсказывает | Доказательства | Ограничения |
|---|---|---|---|
| FxNorm-automix (Sony, ISMIR'22, arXiv:2208.11428, код открыт) | waveform-микс через mixing-mask | тест: неотличим от живых инженеров | fixed stems, black-box, нет параметров |
| Diff-MST (QMUL+Steinberg, 2024, arXiv:2407.08889, код открыт) | gain/pan/4-band EQ/comp на трек + мастер, через дифференцируемую консоль; кондиционируется референс-песней | AF-метрики ≈ человеческие миксы | статичные параметры, нужен референс, нет реверба в консоли |
| MEGAMI (Sony+Aalto, arXiv:2511.08040, код открыт) | conditional diffusion над effect-эмбеддингами; permutation-equivariant, ест dry+wet данные | «approaching human-level» | один проход, embedding вместо читаемых params |
| Wave-U-Net drum mixing (JAES'21) | waveform | indistinguishable on drums | только барабаны |
| RoEx Automix/Tonn (QMUL-спинаут Ronan/Reiss) | params → рендер + экспорт Ableton-проекта; masking-minimization, 4M+ треков | лучший коммерческий образец | без автоматики/вокальных райдов |
| Cryo Mix | эвристики + чат-агент Nova | — | нет DAW-экспорта |

Консенсус поля: **предсказывать интерпретируемые параметры/эмбеддинги,
рендерить через явный FX-chain**, не waveform (артефакты, нередактируемость).

## Путь A: агент в DAW

Почти пусто: DAWZY (arXiv:2512.03289, REAPER/ReaScript MCP, код-генерация —
агент **не слушает** результат), Ableton-MCP-комьюнити. Никто не публиковал
полный цикл «агент слушает → правит параметры → переслушивает» на мультитреке.

## In-between (где живёт бóльшая часть серьёзной работы)

**Perception-in-the-loop (слушающая модель = судья в цикле оптимизации):**
- ST-ITO (Steinmetz+, ISMIR'24 best paper): CMA-ES по параметрам *любых*
  плагинов, судья = AFx-Rep embedding (self-supervised стиль). Минуты на fit.
- Text2FX (ICASSP'25): градиент по dasp-параметрам к цели в CLAP-эмбеддинге.
- FxSearcher: байесовская оптимизация, судья = CLAP.
- GRAFx-prune (Sony, DAFx'24): бэкпроп восстанавливает mixing graph по
  паре dry→mix; журнальный вариант arXiv:2509.15948.
- Wilson IGA (JAES'19): человек-слушатель как fitness → миксы уровня фейдеров.

**RL/post-training:** SPHERE (arXiv:2608.30559) — ALM предсказывает spatial
параметры, GRPO с детерминистическими перцептивными наградами; Conv-EQ
DPA (B&O, arXiv:2609.12607) — 1.5B на 90k предпочтениях = паритет с GPT-4o-mini.
Классического RL-агента по полному миксу нет (DeepFADE — отрицательный
результат: reward hacking).

**Agentic:** RIME+POEMS (Dartmouth, arXiv:2607.19605) — «agentic
post-production», тулкит + 15k синтетики, SFT помогает, но честно
констатирует: модели пока не дотягивают до такого цикла.

## Что слышат аудио-LLM (ключевые факты)

- **SonicBench** (arXiv:2601.11039): loudness-compare лучший 0.80 vs человек
  0.97; direction/reverb ≈ случайность; половина из 36 моделей на уровне
  шанса. **Линейный пробник на замороженном энкодере ≥60%** — сигнал в
  энкодере есть, не вербализуется.
- **Music Flamingo**: MF-Skills капшены инжектят ground truth из
  madmom/essentia/Chordino — NVIDIA сама обходит энкодер для низкоуровневого.
  Наше измерение (MF глуха к уровням) = норма, не баг.
- **AudioJev** (arXiv:2610.01293): typed answers + RD-SKL калибровка по
  порядку опций — правильный интерфейс судьи. **Sori-1B-MCQ** — тот же
  SystemOne-паттерн («Which clip is louder?» как типизированный вызов).
- AudioJudge: pairwise-промпт с ICL = 0.91 Spearman на уровне систем, но
  позиционный bias — RD-SKL это лечит.
- Вычислимые «уши»: masking-minimization clarity (ρ=0.838), Audiobox
  Aesthetics PQ-ось, MuQ-Eval (0.838 SRCC; ~150 клипов хватает для
  персонального вкуса), TuneJury — **~100 A/B-пар перекалибровывают судью
  под новый вкус** без ретрейна.

## Датасеты

MedleyDB (196, wet stems+mix), Mixing Secrets/Cambridge-MT (625 dry
multitrack), MUSDB18, ENST-drums (dry→wet пары!), Mix Evaluation Dataset
(180 миксов + параметры + 5k рейтингов — единственная тройка
«микс+настройки+суждения»), MixAssist (431 ход диалогов; FT Qwen-Audio даёт
лучшие советы — учит *говорить*, не *крутить*), MixParams. Пар
«сухие стемы → pro-микс с параметрами» публично почти нет — отсюда у всех
дифференцируемые консоли.

## Где мы

Наш цикл (внешний агент + метры + типизированный слух AudioJev + bounce) =
именно незанятая клетка: полный listen→critique→revise→re-render на
мультитреке через типизированного судью. Ближайшие соседи: RIME/POEMS
(агент, но судья слабый) и SPHERE (RL с детерминистическими наградами,
но без внешнего агента).
