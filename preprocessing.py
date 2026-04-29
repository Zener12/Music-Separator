"""
preprocessing.py

Отвечает за одну вещь: превратить сырое аудио в то, что нейросеть может "увидеть",
и уметь обратно превратить результат в аудио.

Почему мы вообще не подаём сырой сигнал в сеть?
  Сырой сигнал — это 22050 чисел на каждую секунду. Одна нота длится ~0.05 сек,
  и в сыром сигнале вы её "не увидите" — она размыта по тысячам отсчётов.
  Спектрограмма показывает, какие частоты звучат в каждый момент времени.
  Нейросеть на спектрограмме учится примерно так же, как человек "видит" звук.

Почему STFT, а не мел-спектрограмма?
  Подсказка: можно ли точно восстановить аудио из мел-спектрограммы?
"""

import numpy as np
import librosa
import torch


# Гиперпараметры — почему именно эти значения, объясняется в docstring каждой функции
SAMPLE_RATE    = 22050
N_FFT          = 2048
HOP_LENGTH     = 512
SEGMENT_SEC    = 3.0
SEGMENT_FRAMES = int(np.ceil(SEGMENT_SEC * SAMPLE_RATE / HOP_LENGTH))


def load_audio(path: str, sr: int = SAMPLE_RATE):
    """
    Загружает аудиофайл как моно-сигнал.

    Зачем моно? У нас нет задачи сохранить стерео-панораму — нам важна
    частотная структура сигнала. Моно вдвое меньше данных.

    Зачем sr=22050, а не оригинальный (обычно 44100)?
    Музыка слышна до ~20 кГц, но спектральная структура инструментов —
    до ~10 кГц. 22050 Hz покрывает до 11025 Hz (теорема Найквиста).
    Вдвое меньше данных — вдвое быстрее обучение.

    Что вернуть: np.ndarray shape (samples,)
    """

    audio = librosa.load(path, sr=sr, mono=True)

    # Загрузи файл через librosa.load
    # Какие параметры отвечают за моно и частоту дискретизации?
    return audio[0]


def audio_to_spectrogram(audio: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH):
    """
    Преобразует аудиосигнал в комплексный спектр (STFT), затем разделяет
    на амплитуду и фазу.

    n_fft — размер окна FFT.
      Определяет частотное разрешение: freq_bins = n_fft // 2 + 1
      При n_fft=2048: 1025 частотных бинов, разрешение = 22050/2048 ≈ 10 Hz

    hop_length — шаг между окнами.
      Определяет временное разрешение: чем меньше шаг, тем больше кадров.
      Обычный выбор: n_fft // 4

    Зачем возвращать фазу отдельно?
      Нейросеть будет предсказывать маску для амплитуды.
      Фазу мы сохраняем от смеси и используем при восстановлении — это
      называется "phase approximation". Не идеально, но работает на практике.

    Вернуть: (magnitude, phase), оба shape (freq_bins, time_frames)
    """

    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)

    # librosa.stft возвращает комплексный массив
    # np.abs — амплитуда, np.angle — фаза
    magnitude = np.abs(stft)
    phase = np.angle(stft)

    return magnitude, phase

def spectrogram_to_audio(magnitude: np.ndarray, phase: np.ndarray, hop_length: int = HOP_LENGTH):
    """
    Обратное преобразование: амплитуда + фаза → аудио.

    Вспомни формулу комплексного числа через полярные координаты:
      z = r * e^(iθ),  где r = амплитуда, θ = фаза
    В numpy это: magnitude * np.exp(1j * phase)

    После получения комплексного STFT — librosa.istft.
    """

    z = magnitude * np.exp(1j * phase)

    return librosa.istft(z, hop_length=hop_length)


def log_magnitude(magnitude: np.ndarray, eps: float = 1e-7):
    """
    Логарифмическое масштабирование амплитуд.

    Зачем: динамический диапазон музыки — от едва слышимого (≈0) до громкого (100+).
    Нейросети плохо работают с такими разбросами. Логарифм сжимает диапазон.

    Почему eps?
      Попробуй мысленно: что будет при log(0)?
      eps — маленькое число, защищающее от -inf.

    Вернуть: np.log(magnitude + eps)
    """
    return np.log(magnitude + eps)


def normalize(spec: np.ndarray):
    """
    Z-score нормализация: приводим распределение к ~N(0, 1).

    Формула: (x - mean) / std

    Почему не [0, 1] (MinMax)?
      MinMax зависит от конкретного трека. Один трек тихий, другой громкий —
      диапазоны разные. Z-score нормализует относительно среднего и разброса
      данного сегмента, независимо от абсолютной громкости.

    ВАЖНО: вернуть вместе с mean и std — они понадобятся в denormalize,
    чтобы вернуть предсказание модели обратно в "реальный" масштаб.

    Вернуть: (normalized_spec, mean, std)
    """

    mean = np.mean(spec)
    std = np.std(spec) + 1e-7

    return (spec - mean) / std, mean, std
    # Добавь 1e-7 к std — зачем?


def denormalize(spec_norm: np.ndarray, mean: float, std: float):
    """Обратная операция к normalize. Одна строка."""
    return spec_norm * std + mean


def segment_spectrogram(magnitude: np.ndarray, segment_frames: int = SEGMENT_FRAMES):
    """
    Нарезает спектрограмму треку на равные куски по времени.

    Зачем нарезать?
      Полный трек = ~5000 кадров. Батч из 4 таких треков не поместится в 4 GB VRAM.
      Кроме того, DataLoader требует тензоры одинакового размера в батче —
      треки разной длины нельзя сложить в батч напрямую.

    Что делать с последним неполным сегментом?
      Его лучше отбросить. Почему это безопаснее чем дополнять нулями (padding)?
      Подсказка: что произойдёт с маской на нулевом участке?

    Вернуть: список np.ndarray, каждый shape (freq_bins, segment_frames)
    """

    segments = []

    for i in range(0, magnitude.shape[1] - segment_frames + 1, segment_frames):
        segment = magnitude[:, i:i+segment_frames]
        segments.append(segment)

    return segments

    # Цикл с шагом segment_frames по оси времени (axis=1)


def prepare_tensor(magnitude_segment: np.ndarray):
    """
    Конвертирует сегмент в тензор и добавляет размерность канала.

    Conv2d ожидает (batch, channels, H, W).
    У нас спектрограмма — как grayscale изображение: 1 канал.
    unsqueeze(0) добавляет эту размерность → (1, freq_bins, frames)

    Батч DataLoader добавит потом автоматически → (batch, 1, freq_bins, frames)
    """
    return torch.FloatTensor(magnitude_segment.copy()).unsqueeze(0)