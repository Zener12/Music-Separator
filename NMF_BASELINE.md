# NMF Baseline — реализация и нюансы

## Подход: Supervised NMF

Обучаем словарь частотных паттернов W на чистых стемах из train-сета.
При разделении микса — фиксируем W, находим активации H, реконструируем через Wiener-маску.

---

## Структура кода

### `_fit_source(source, specs)`
Вспомогательная функция обучения одного NMF. Вынесена на уровень модуля (не вложенная) — `joblib.Parallel` не принимает вложенные функции.

На Windows `Parallel` не заработал (та же проблема что с `num_workers` в DataLoader — нет `fork`). Функция осталась как вспомогательная для последовательного цикла.

### `train_nmf_dictionaries(n_tracks)`
1. Читает спектрограммы из `./cache/` — не пересчитывает STFT
2. Денормализует + `np.exp()` → линейная амплитуда (NMF требует ≥ 0)
3. Склеивает треки по оси времени (`axis=1`) → одна большая матрица V на источник
4. `nmf.fit(V.T)` — sklearn ожидает `(samples, features)`, поэтому транспонируем
5. Возвращает `nmf.components_.T` → W shape `(freq, n_components)`

### `separate_with_nmf(mix_audio, dictionaries)`
1. STFT микса → magnitude + phase
2. `W_all = concat(W_vocals, W_drums, W_bass, W_other)` → shape `(1025, 64)`
3. NMF с `init='custom'`, `W=W_all.T` — инициализация обученными словарями
4. Срез H для каждого источника: `H[:, i*16:(i+1)*16]`
5. Wiener-маска: `V_src / (V_total + eps)` × magnitude микса
6. `spectrogram_to_audio(separated_mag, phase)` — phase approximation

### `evaluate_nmf(musdb_root, dictionaries, n_tracks)`
Структура аналогична `evaluate.py`. Ресэмплинг 44100→22050 Hz обязателен для микса и эталонов.

---

## Оптимизации

| Что | Эффект |
|-----|--------|
| Кэш вместо STFT | ~10–20x быстрее: `np.load` + денормализация vs `librosa.resample` + `librosa.stft` |
| `init='nndsvda'` | Детерминированная инициализация, сходится быстрее чем `random` |
| `n_components=16` | Баланс качества и скорости; меньше → быстрее но слабее W |
| `MAX_ITER=500` | Увеличено с 200 чтобы убрать `ConvergenceWarning` у drums |

---

## Нюансы реализации

- **Нормализация в кэше**: данные хранятся как z-score нормализованный log-magnitude (float32). Перед NMF: денормализация `spec * std + mean`, затем `np.exp()` → линейная амплитуда ≥ 0
- **`W=W_all.T`**: sklearn хранит компоненты как `(n_components, freq)`, нам нужен `(freq, n_components)` → `.T` при сохранении и при передаче в `fit_transform`
- **Val не нужен**: NMF не переобучается — нет эпох, нет early stopping, нет выбора чекпоинта. Используем все 100 train-треков
- **Windows + joblib**: `Parallel(n_jobs=4)` с loky backend не работает на Windows — откатились к последовательному `for source in SOURCES`
- **Phase approximation**: фаза источников берётся от микса, не предсказывается — стандартная практика для маскинг-методов
