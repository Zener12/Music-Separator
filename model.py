"""
model.py

Архитектура нейросети: encoder-decoder с масками.

Общая идея:
  Вход — спектрограмма смеси (1 канал, как grayscale-изображение).
  Выход — N масок [0, 1], по одной на каждый источник.
  Маска умножается на смесь → предсказание источника.

  Это называется Multiplicative Masking.
  Почему не предсказывать спектрограмму источника напрямую?
  Подсказка: что ограничивает маска, чего не ограничивает прямое предсказание?

Архитектура U-Net (encoder-decoder со skip connections):

  Вход (1, F, T)
      ↓ EncoderBlock 1 → (16, F/2, T/2)  + сохраняем skip_1
      ↓ EncoderBlock 2 → (32, F/4, T/4)  + сохраняем skip_2
      ↓ EncoderBlock 3 → (64, F/8, T/8)  + сохраняем skip_3
      ↓ Bottleneck    → (64, F/8, T/8)
      ↑ DecoderBlock 3 ← skip_3 → (32, F/4, T/4)
      ↑ DecoderBlock 2 ← skip_2 → (16, F/2, T/2)
      ↑ DecoderBlock 1 ← skip_1 → (16, F, T)
      ↓ Conv 1x1      → (n_sources, F, T)
      ↓ Sigmoid       → маски [0, 1]

Зачем нужны skip connections?
  Encoder сжимает информацию и теряет детали.
  Skip connections передают детали напрямую в decoder.
  Без них decoder "не знает" локальную структуру спектрограммы.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from preprocessing import N_FFT, SEGMENT_FRAMES
from data_loader import SOURCES

N_SOURCES = len(SOURCES)
FREQ_BINS = N_FFT // 2 + 1  # 1025 частотных бинов


class ConvBlock(nn.Module):
    """
    Базовый блок: Conv2d → BatchNorm2d → LeakyReLU.

    Почему BatchNorm?
      Нормализует активации внутри батча. Ускоряет обучение и
      снижает чувствительность к инициализации весов.

    Почему LeakyReLU, а не обычный ReLU?
      ReLU: f(x) = max(0, x). При x < 0 градиент = 0.
      Нейрон "умирает" и больше не обновляется.
      LeakyReLU: f(x) = x если x > 0, иначе 0.2 * x.
      Маленький градиент для отрицательных значений — нейроны не умирают.

    bias=False в Conv2d — зачем, если после стоит BatchNorm?
      BatchNorm сам содержит обучаемые смещения (beta).
      bias в Conv2d был бы лишним параметром, который BatchNorm всё равно
      отменит. Убираем ради экономии.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3,
                 stride: int = 1, padding: int = 1):
        super().__init__()
        # Собери nn.Sequential из трёх слоёв: Conv2d, BatchNorm2d, LeakyReLU(0.2)
        # Не забудь bias=False и inplace=True для LeakyReLU (экономит память)
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class EncoderBlock(nn.Module):
    """
    Один шаг encoder'а: применяет ConvBlock, сохраняет результат как skip,
    затем делает downsampling (уменьшает пространственные размеры вдвое).

    Как сделать downsampling свёрткой?
      stride=2 в Conv2d — каждый второй пиксель "пропускается".
      Это лучше чем MaxPool, потому что stride=2 обучаем (параметрический).
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        # self.conv  — ConvBlock без downsampling
        self.conv = ConvBlock(in_ch, out_ch)
        # self.down  — ConvBlock со stride=2 (downsampling)
        self.down = ConvBlock(out_ch, out_ch, stride=2)

    def forward(self, x):
        # Примени self.conv → получи skip
        skip = self.conv(x)
        # Примени self.down к skip → получи x
        x = self.down(skip)
        # Верни (x, skip) — skip нужен для decoder'а
        return (x, skip)


class DecoderBlock(nn.Module):
    """
    Один шаг decoder'а: upsampling + конкатенация со skip + ConvBlock.

    ConvTranspose2d — "обратная свёртка", увеличивает размеры вдвое.
    Это обучаемый upsampling (в отличие от bilinear interpolation).

    После upsampling — конкатенируем skip по оси каналов (dim=1).
    Поэтому входных каналов для ConvBlock будет out_ch * 2.

    Проблема нечётных размеров:
      Если на входе была нечётная размерность (например, 513),
      после downsampling/upsampling размеры могут не совпасть на 1 пиксель.
      F.pad выравнивает тензор перед конкатенацией.
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_ch + out_ch, out_ch)

    def forward(self, x, skip):
        # Upsampling
        x = self.up(x)
        # Если размеры не совпадают со skip — F.pad
        if x.shape != skip.shape:
            x = x[:, :, :skip.shape[2], :skip.shape[3]]
        # Конкатенация по dim=1
        x = torch.cat([x, skip], dim=1)
        # ConvBlock
        return self.conv(x)


class SourceSeparationNet(nn.Module):
    """
    Полная модель.

    Вход:  (batch, 1, freq_bins, frames)
    Выход: (batch, n_sources, freq_bins, frames) — маски в [0, 1]

    Количество каналов специально маленькое (16→32→64) — чтобы влезло в 4 GB VRAM.
    Для GPU с 8+ GB можно попробовать 32→64→128 и сравнить качество.
    """

    def __init__(self, n_sources: int = N_SOURCES):
        super().__init__()
        self.n_sources = n_sources

        # Encoder: 3 блока, каждый удваивает каналы и уменьшает размер вдвое
        self.encoder = nn.ModuleList([
            EncoderBlock(1, 16),
            EncoderBlock(16, 32),
            EncoderBlock(32, 64)
        ])
        # Bottleneck: ConvBlock без изменения размера (самое узкое место)
        self.bottleneck = ConvBlock(64, 64)
        # Decoder: 3 блока, каждый уменьшает каналы вдвое и увеличивает размер вдвое
        self.decoder = nn.ModuleList([
            DecoderBlock(64, 32),
            DecoderBlock(32, 16),
            DecoderBlock(16, 16),
        ])
        # Final: Conv2d(16, n_sources, kernel_size=1) — 1x1 свёртка, только смешивает каналы
        self.final = nn.Conv2d(16, n_sources, kernel_size=1)

    def forward(self, x):
        # Encoder — сохрани skip connections
        x, skip1 = self.encoder[0](x)
        x, skip2 = self.encoder[1](x)
        x, skip3 = self.encoder[2](x)
        # Bottleneck
        x = self.bottleneck(x)
        # Decoder — передай skip connections в обратном порядке
        x = self.decoder[0](x, skip3)
        x = self.decoder[1](x, skip2)
        x = self.decoder[2](x, skip1)
        # Final + sigmoid
        x = self.final(x)
        x = torch.sigmoid(x)
        return x


def apply_masks(mixture: torch.Tensor, masks: torch.Tensor):
    """
    Применяет маски к спектрограмме смеси.

    mixture: (batch, 1, F, T)
    masks:   (batch, n_sources, F, T)

    Broadcasting сделает всё сам — mixture транслируется по оси n_sources.
    Результат: (batch, n_sources, F, T)
    """
    return mixture * masks


def count_parameters(model: nn.Module):
    """Считает количество обучаемых параметров модели."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Сюда напишешь тест: создай модель, прогони dummy тензор,
    # проверь что выход в [0, 1] и shape правильный.

    x = torch.randn(2, 1, FREQ_BINS, SEGMENT_FRAMES)
    print(x.shape)
    model = SourceSeparationNet()
    output = model(x)
    print(output.shape)
    print(f"min = {output.min()}; max = {output.max()}")
    print(count_parameters(model))