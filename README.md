# NeuroMouse

Исследовательский прототип: обработка нейрофизиологических сигналов и ИИ-декодирование.
Текущий фокус — электрофизиология мыши (Neuropixels), а не только ЭЭГ.

Автор: Никита Кукса · Рабочая директория: `F:\NeuroMouse` · Окружение: `conda neuromouse`

## Окружение

- Python: conda-окружение `neuromouse` (Miniforge внутри папки проекта, в git не попадает).
- Сигналы: MNE. Численный анализ: NumPy, SciPy, Pandas.
- Машинное обучение: scikit-learn.
- Электрофизиология мыши: pynwb, dandi, remfile.

## Структура проекта

| Путь | Назначение |
|------|------------|
| `src/` | Рабочий код (единый пакет). Все пути и параметры — в `src/config.py`. |
| `src/eeg_pipeline.py` | Конвейер обработки ЭЭГ (Этап I). |
| `src/mouse_lfp_analysis.py` | Анализ внеклеточного LFP мыши (Этап II). |
| `src/mouse_spikes_decoder.py` | ИИ-декодер по спайкам мыши (Этап III). |
| `src/mouse_extracellular_analysis.py` | Spike sorting и популяционный анализ (Этап IV). |
| `data/` | Исходные записи (`S001R01.edf`, `*.nwb`). В git не версионируются. |
| `models/` | Обученные модели (`mouse_lr_decoder.joblib`). |
| `results/stage1/` | Выходные файлы ЭЭГ-конвейера. |
| `results/mouse_stage/` | Результаты анализа и декодера мыши. |
| `results/mouse_stage/stage4/` | Результаты spike sorting и популяционного анализа. |
| `sessions_registry.json` | Внутренний реестр проанализированных сессий. |
| `neuromouse_stage1/README.md` | Инструкция по воспроизведению ЭЭГ-конвейера. |

## Как запустить

```powershell
conda activate neuromouse

# Этап I — конвейер ЭЭГ (результаты -> results/stage1/)
python -m src.eeg_pipeline

# Этапы II–III — анализ и декодер мыши (результаты -> results/mouse_stage/)
python -m src.mouse_lfp_analysis
python -m src.mouse_spikes_decoder

# Этап IV — spike sorting и популяционный анализ (результаты -> results/mouse_stage/stage4/)
python -m src.mouse_extracellular_analysis
```

## Целостность данных

- Исходные записи не изменяются; фильтрация применяется к копии.
- Обработанные данные и признаки хранятся отдельно от исходных.
- Метки событий проверяются до обучения с учителем.

## Состояние этапов

- Этап 0 — исходное состояние: завершён.
- Этап I — конвейер ЭЭГ: завершён (проверки качества, документация формул и верификация пройдены).
- Этап II — данные мыши (Neuropixels): завершён (сессии зарегистрированы в `sessions_registry.json`).
- Этап III — первый ИИ-декодер мыши: завершён (LDA ≈ 65.3% против 50% случайного уровня).
- Этап IV — spike sorting и популяционный анализ: завершён (синтетика с ground truth: accuracy/recall/precision = 1.0; 778 юнитов Steinmetz).
- Этапы V–VIII — не начаты.

Актуальный детальный план и статусы задач — в `neuromouse_tasks.html` и `NeuroMouse_TaskManager.md`.
