# Music Source Separation

Разделение музыкального трека на 4 источника: **vocals, drums, bass, other**.

Реализованы два подхода:
- **U-Net** (CNN encoder-decoder с multiplicative masking)
- **Supervised NMF** (классический baseline)

Датасет: [MUSDB18](https://sigsep.github.io/datasets/musdb.html) (100 train + 50 test треков).

---

## Установка

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
pip install librosa musdb scikit-learn mir_eval soundfile gradio mlflow
```

Датасет положить в `./musdb18/` (формат WAV).

---

## Структура проекта

```
├── preprocessing.py     # STFT, нормализация, сегментация
├── data_loader.py       # Dataset с дисковым кэшем спектрограмм
├── model.py             # U-Net архитектура
├── train.py             # Цикл обучения
├── evaluate.py          # SDR-оценка нейросети
├── nmf_baseline.py      # Supervised NMF baseline
├── demo.py              # Разделение трека и сохранение WAV
├── export_ONNX.py       # Экспорт модели в ONNX формат
├── log_results.py       # Логирование метрик в MLflow
├── gradio_demo.py       # Веб-интерфейс для разделения треков
├── checkpoints/         # Веса модели (.pt и .onnx)
└── cache/               # Кэш спектрограмм (создаётся автоматически)
```

---

## Запуск

### Обучение нейросети
```bash
python train.py
```
Автоматически возобновляется с последнего чекпоинта если он есть.

### Оценка нейросети
```bash
python evaluate.py ./checkpoints/checkpoint_epoch_48.pt ./musdb18
```

### Демо — разделить трек и сохранить WAV
```bash
python demo.py
```
Результат сохраняется в `./results_demo/{track_name}/`.

### NMF baseline
```bash
python nmf_baseline.py
```

### Экспорт в ONNX
```bash
python export_ONNX.py
```
Сохраняет модель в `./checkpoints/model.onnx` для инференса без PyTorch.

### Логирование результатов в MLflow
```bash
python log_results.py
mlflow ui  # открыть браузер на localhost:5000
```

### Веб-интерфейс (Gradio)
```bash
python gradio_demo.py
```
Загрузи WAV-файл в браузере и получи 4 разделённых источника.

---

## Результаты (50 тестовых треков, SDR в дБ)

| Источник | U-Net | NMF baseline | Demucs v3 (SOTA) |
|----------|-------|--------------|------------------|
| Vocals   | 2.72  | 0.21         | 8.99             |
| Drums    | 3.36  | -0.27        | 8.72             |
| Bass     | 0.29  | -2.53        | 7.84             |
| Other    | 0.09  | -3.04        | 5.09             |
| **Average** | **1.62** | **-1.41** | **7.66**      |

U-Net превосходит NMF baseline на ~3 dB по всем источникам.

---

## Конфигурация обучения

| Параметр | Значение |
|----------|----------|
| Epochs | 50 |
| Batch size | 4 |
| Learning rate | 1e-3 → 7.81e-6 (ReduceLROnPlateau) |
| Loss | 0.9 × L1 + 0.1 × SDR |
| GPU | NVIDIA GTX 1650 mobile, 4 GB VRAM |
