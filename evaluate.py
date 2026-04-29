"""
evaluate.py

Оценка качества разделения по метрике SDR.

SDR (Signal-to-Distortion Ratio), измеряется в дБ:
  < 0  dB — предсказание хуже чем тишина
  0–5  dB — слышно разделение, много артефактов
  5–10 dB — приемлемо для прослушивания
  > 10 dB — хорошее качество

  Для справки: Demucs (SOTA 2023) даёт ~7–9 dB на MUSDB18.
  Наша скромная сеть должна достичь хотя бы 3–5 dB — это честный результат.

Почему не MSE (L2-норма = sqrt(x1^2 + x2^2 + ...)) или 
          MAE (L1-норма = |x1| + |x2| + ...) для оценки?
  MSE/L1 считает ошибку в числах спектрограммы.
  SDR считает ошибку в терминах "насколько слышно оригинал vs артефакты".
  Человеческое восприятие ближе к SDR.

Используем mir_eval — стандартная библиотека для MIR-задач (Music Information Retrieval).
"""

import torch
import torch.nn as nn
import numpy as np
import mir_eval
import musdb

from model import SourceSeparationNet, apply_masks
from preprocessing import (
    audio_to_spectrogram, log_magnitude, normalize, denormalize,
    spectrogram_to_audio, segment_spectrogram, prepare_tensor,
    SEGMENT_FRAMES,
)
from train import load_checkpoint
from data_loader import SOURCES


def separate_track(model: nn.Module, mix_audio: np.ndarray, device: str = 'cpu'):
    """
    Разделяет один трек на источники.

    Нельзя прогнать весь трек через модель целиком — не влезет в VRAM.
    Прогоняем сегментами и склеиваем обратно.

    Алгоритм:
      1. STFT смеси → magnitude, phase
      2. log + normalize → mix_norm
      3. Нарезать на сегменты
      4. Каждый сегмент через модель → маски → предсказания
      5. Собрать предсказания обратно в один массив
      6. Denormalize → exp (обратный log) → spectrogram_to_audio

    Важно: используй model.eval() и torch.no_grad()

    Вернуть: dict {source_name: np.ndarray(audio)}
    """
    pass


def compute_sdr(ref_audio: np.ndarray, est_audio: np.ndarray):
    """
    Считает SDR между эталоном и предсказанием.

    mir_eval.separation.bss_eval_sources ожидает:
      reference: (n_sources, samples)
      estimated: (n_sources, samples)
    У нас один источник → добавляем размерность [np.newaxis, :]

    Важно выровнять длины — после iSTFT длина может чуть отличаться.

    Вернуть: float (SDR в дБ)
    """
    pass


def evaluate_model(model: nn.Module, musdb_root: str,
                   device: str = 'cpu', n_tracks: int = None):
    """
    Прогоняет модель по тестовому сету, считает средний SDR.

    n_tracks=None → все 50 тестовых треков.
    Для быстрой проверки во время разработки: n_tracks=5.

    Вернуть: dict {source: mean_sdr, 'average': overall_mean}
    """
    pass


if __name__ == '__main__':
    import sys

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else './checkpoints/best.pt'
    musdb_root      = sys.argv[2] if len(sys.argv) > 2 else './musdb18'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ckpt = load_checkpoint(checkpoint_path, device)
    print(f"Загружен чекпоинт: эпоха {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    evaluate_model(model, musdb_root, device=device)
