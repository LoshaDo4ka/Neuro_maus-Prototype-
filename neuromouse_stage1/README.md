# NeuroMouse — Этап I: воспроизводимый конвейер ЭЭГ

Инструкция по воспроизведению за 3 шага.

## 1. Окружение

```powershell
conda activate neuromouse
python -m pip install mne numpy pandas scipy matplotlib
```

## 2. Входные данные

Убедитесь, что исходная запись лежит здесь:

`F:\NeuroMouse\data\S001R01.edf`

## 3. Запуск

Из корня проекта `F:\NeuroMouse`:

```powershell
conda activate neuromouse
python -m src.eeg_pipeline
```

Скрипт — это `src/eeg_pipeline.py`. Он создаст/перезапишет папку:

`F:\NeuroMouse\results\stage1\`

Там будут отфильтрованная запись (`.fif`), CSV с признаками, метрики качества
каналов, графики (PNG), метаданные (`run_metadata.json`), отчёт
(`stage1_report.md`) и журнал запуска (`pipeline.log`).

## Ожидаемый результат (проверка воспроизводимости)

- 64 канала, 160 Гц, 61 с.
- `features_all_channels.csv` — ровно **1856 строк**.
- Повторный запуск даёт те же числа (SHA-256 совпадают).

## Формулы и единицы измерения признаков

Признаки считаются по скользящим окнам длиной **4 с** с шагом **2 с** отдельно
для каждого канала. Для каждого окна строится односторонняя PSD Уэлча
(`scipy.signal.welch`, `nperseg` = длина окна) на отфильтрованном сигнале.

| Признак (колонка в `features_all_channels.csv`) | Формула | Единица | Код |
|---|---|---|---|
| `<band>_power_v2` | `P = ∫ S(f) df` по диапазону | В² | `band_power()`, `build_features()` в `src/eeg_pipeline.py` |
| `power_1_30_hz_v2` | `P_delta+P_theta+P_alpha+P_beta` (только 1–30 Гц) | В² | `build_features()` |
| `<band>_percent_of_1_30` | `100 · P_band / power_1_30_hz_v2` | % (0–100) | `build_features()` |
| `alpha_peak_hz` | `argmax_f S(f)`, `f ∈ [8, 13)` Гц | Гц | `build_features()` |
| `window_start_s`, `window_duration_s` | начало окна; длина окна | с | `build_features()` |

Диапазоны (Гц): δ 1–4, θ 4–8, α 8–13, β 13–30 (`FREQUENCY_BANDS` в `src/config.py`).

Отдельные файлы метрик качества (`channel_quality_metrics.csv`,
`o1_rms_by_window.csv`) используют формулу

```
RMS = sqrt( (1/N) · Σ xᵢ² )
```

и единицу **мкВ** (`rms_raw_uv`, `rms_filtered_uv`, `rms_difference_uv`,
`min_raw_uv`, `max_raw_uv`).

**Не смешивать единицы:** в `features_all_channels.csv` абсолютные мощности
записаны в В² (суффикс `_v2`), а относительные — в % (суффикс
`_percent_of_1_30`); метрики качества — в мкВ.

## Верификация этапа I

Отдельный скрипт ничего не меняет в конвейере, а проверяет уже готовые
артефакты (задачи 1.5, 1.8, 1.9, 1.10, 1.15, 1.16):

```powershell
python -m src.stage1_verification
```

Результат — `results/stage1/verification_results.json` (поле `all_checks_passed`)
и графики `psd_before_after_channels.png`, `edge_effects.png`.
Скрипт завершается кодом 1, если хотя бы одна проверка не пройдена.

## Важные ограничения

- Исходный EDF не изменяется.
- Фильтрация применяется к копии записи.
- Признаки рассчитываются по 4-секундным окнам с шагом 2 секунды для всех каналов.
- Проценты полос считаются относительно суммы мощности delta, theta, alpha и beta
  (1–30 Гц), а не относительно полной мощности сигнала.
- Это прототип технической обработки, а не диагностический инструмент и не обученная ИИ-модель.
