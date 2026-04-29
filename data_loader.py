"""
data_loader.py

Отвечает за одно: дать модели данные в нужном формате батчами.

В PyTorch для этого два класса:
  Dataset  — описывает ЧТО данные и КАК получить один элемент по индексу
  DataLoader — берёт Dataset и занимается батчингом, перемешиванием, параллельной загрузкой

Тебе нужно реализовать только Dataset. DataLoader — стандартный, из PyTorch.
"""

from pathlib import Path

import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import musdb
import numpy as np

from preprocessing import (
    audio_to_spectrogram,
    log_magnitude,
    normalize,
    prepare_tensor,
    SAMPLE_RATE,
    HOP_LENGTH,
    SEGMENT_FRAMES
)

SOURCES = ['vocals', 'drums', 'bass', 'other']


class MUSDBDataset(Dataset):
    """
    Dataset для MUSDB18.

    Любой кастомный Dataset в PyTorch — это класс с тремя методами:
      __init__    — инициализация, загрузка/подготовка данных
      __len__     — сколько элементов в датасете
      __getitem__ — один элемент по индексу idx

    DataLoader вызывает __getitem__ случайно (если shuffle=True) и
    складывает результаты в батч через torch.stack.

    Из этого следует важное требование: __getitem__ всегда должен
    возвращать тензоры одинакового shape. Именно поэтому мы нарезаем треки
    на сегменты фиксированной длины.
    """

    def __init__(self, root: str, subset: str = 'train', track_start=0, track_end=None, download: bool = False):
        """
        root    — путь к папке с MUSDB18
        subset  — 'train' (100 треков) или 'test' (50 треков)
        track_start — индекс первого трека для загрузки
        track_end — индекс последнего трека для загрузки
        download — скачать автоматически (нужно ~7 GB и регистрация)

        Здесь нужно:
          1. Открыть датасет через musdb.DB
          2. Пройти по всем трекам, вычислить спектрограммы смеси и источников
          3. Нарезать на сегменты и сложить в self.segments

        Почему предвычисляем всё в __init__, а не в __getitem__?
          __getitem__ вызывается тысячи раз за эпоху. STFT — дорогая операция.
          Считаем один раз, храним результат в памяти.

          Альтернатива: считать на лету + кэшировать на диск.
          Это лучше для больших датасетов, но сложнее. Пока — в память.
        """
        self.sources = SOURCES
        self.index = []  # список (track_idx, seg_idx)
        self.cache_dir = Path('./cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.track_start = track_start
        self.track_end = track_end

        # Шаг 1: открой датасет
        self.db = musdb.DB(root=root, subsets=subset, is_wav=True)
        # is_wav=True загружает wav-стемы (быстрее чем .stem.mp4)

        # Шаг 2: вызови предобработку
        self._preprocess_all(self.db)

    def _preprocess_all(self, db):
        """
        Проходит по всем трекам в db и наполняет self.segments.

        Для каждого трека:
          - track.audio — стерео смесь, shape (samples, 2)
          - track.targets['vocals'].audio — стерео вокал, shape (samples, 2)
          - аналогично для drums, bass, other

        Как получить моно из стерео?
          .mean(axis=1) усредняет по каналам → (samples,)
          (axis=1, потому что каналы — второе измерение)

        Важно про нормализацию источников:
          Нормализуй источники теми же mean и std что и смесь.
          Почему? Потому что маска = предсказание / смесь.
          Если масштабы разные — маска не будет иметь смысл.

        В конце каждой итерации трека добавь в self.segments кортежи:
          (mix_tensor, targets_tensor)
          где targets_tensor — стек всех источников по оси 0,
          shape (n_sources, freq_bins, frames)
        """
        tracks = list(db)[self.track_start:self.track_end]
        print(f"Preprocessing {len(tracks)} tracks...")

        for track_idx, track in enumerate(tracks, start=self.track_start):
            mix_cache = self.cache_dir / f"track_{track_idx:03d}_mix.npy"

            if mix_cache.exists():
                actual_frames = np.load(mix_cache, mmap_mode='r').shape[1]
                n_segments = actual_frames // SEGMENT_FRAMES
            else:
                mono_mix = librosa.resample(
                    track.audio.mean(axis=1).astype(np.float32),
                    orig_sr=track.rate, target_sr=SAMPLE_RATE
                )
                mix_mag, _ = audio_to_spectrogram(mono_mix)
                mix_log = log_magnitude(mix_mag)
                mix_norm, mean, std = normalize(mix_log)
                np.save(mix_cache, mix_norm.astype(np.float32))
                np.save(self.cache_dir / f"track_{track_idx:03d}_stats.npy",
                        np.array([mean, std], dtype=np.float32))

                for source in self.sources:
                    mono_src = librosa.resample(
                        track.targets[source].audio.mean(axis=1).astype(np.float32),
                        orig_sr=track.rate, target_sr=SAMPLE_RATE
                    )
                    src_mag, _ = audio_to_spectrogram(mono_src)
                    src_norm = (log_magnitude(src_mag) - mean) / std
                    np.save(self.cache_dir / f"track_{track_idx:03d}_{source}.npy",
                            src_norm.astype(np.float32))

                print(f"  Cached track {track_idx + 1}: {track.name}")
                n_segments = mix_norm.shape[1] // SEGMENT_FRAMES

            for seg_idx in range(n_segments):
                self.index.append((track_idx, seg_idx))

        print(f"Готово. Всего сегментов: {len(self.index)}")

    def __len__(self):
        # Сколько элементов в датасете?
        return len(self.index)

    def __getitem__(self, i: int):
        """
        Возвращает один элемент по индексу.

        Что вернуть:
          mixture: тензор (1, freq_bins, frames) — спектрограмма смеси
          targets: тензор (n_sources, freq_bins, frames) — спектрограммы источников

        DataLoader автоматически добавит батч-измерение:
          mixture → (batch, 1, freq_bins, frames)
          targets → (batch, n_sources, freq_bins, frames)
        """
        track_idx, seg_idx = self.index[i]
        start = seg_idx * SEGMENT_FRAMES
        end = start + SEGMENT_FRAMES

        mix_spec = np.load(self.cache_dir / f"track_{track_idx:03d}_mix.npy", mmap_mode='r')
        mix_segment = mix_spec[:, start:end]
        mix_tensor = prepare_tensor(mix_segment)

        source_tensors = []
        for source in self.sources:
            src_spec = np.load(self.cache_dir / f"track_{track_idx:03d}_{source}.npy", mmap_mode='r')
            src_segment = src_spec[:, start:end]
            source_tensors.append(torch.FloatTensor(src_segment.copy()))

        tgt_tensor = torch.stack(source_tensors)
        return (mix_tensor, tgt_tensor)


def get_dataloaders(root: str, batch_size: int = 8, num_workers: int = 0):
    """
    Создаёт и возвращает train и test DataLoader.

    batch_size=4 — осторожный выбор для 4 GB VRAM.
      Если будут ошибки CUDA out of memory — уменьши до 2.

    num_workers=0 — на Windows > 0 часто вызывает deadlock из-за
      особенностей multiprocessing. Безопаснее оставить 0.

    shuffle=True только для train — зачем? Подумай, что произойдёт
    если модель всегда видит треки в одном порядке.

    Вернуть: (train_loader, test_loader)
    """
    # Создай MUSDBDataset для train и test
    # Оберни каждый в DataLoader с нужными параметрами
    train_dataset = MUSDBDataset(root=root, subset='train', track_start=0, track_end=80)
    val_dataset = MUSDBDataset(root=root, subset='train', track_start=80, track_end=100)
    test_dataset = MUSDBDataset(root=root, subset='test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return (train_loader, val_loader, test_loader)