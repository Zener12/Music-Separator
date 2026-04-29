"""
train.py

Цикл обучения нейросети.

Как выглядит одна итерация обучения:
  1. Достать батч из DataLoader
  2. Прогнать через модель (forward pass) → предсказание
  3. Посчитать loss (насколько предсказание отличается от правды)
  4. Backward pass — вычислить градиенты
  5. Optimizer.step() — обновить веса
  6. Обнулить градиенты (zero_grad) перед следующей итерацией

Зачем обнулять градиенты?
  PyTorch по умолчанию НАКАПЛИВАЕТ градиенты.
  Если не обнулять — каждый шаг будет суммировать градиенты
  всех предыдущих шагов. Это не то, что нам нужно.

Loss-функция:
  L1 Loss (MAE) — предсказываем амплитуды, L1 устойчива к выбросам.
  SDR Loss — напрямую оптимизируем целевую метрику.
  Итого: 0.9 * L1 + 0.1 * SDR

  Почему нельзя оптимизировать только SDR?
  SDR нестабильна в начале обучения (log от близкого к нулю).
  L1 даёт стабильный сигнал пока SDR "раскачивается".
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path

from model import SourceSeparationNet, apply_masks, count_parameters
from data_loader import get_dataloaders


CONFIG = {
    'musdb_root':     './musdb18',
    'checkpoint_dir': './checkpoints', # сохраняем веса сюда
    'epochs':         50, # количество эпох обучения
    'batch_size':     4, # батчи (под 4 гб VRAM)
    'lr':             1e-3, # начальный learning rate
    'lr_patience':    5, # сколько эпох ждать улучшения val_loss, прежде чем уменьшить lr
    'l1_weight':      0.9, # вес MAE в loss-func
    'sdr_weight':     0.1, # вес SDR в loss-func
    'grad_clip':      1.0, # максимум градиента при его взрыве (чтобы веса не улетели в nan)
    'seed':           42, # воспроизводимость
}


def set_seed(seed: int):
    """Фиксирует все источники случайности для воспроизводимости."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def sdr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    """
    Дифференцируемая SDR loss. Минимизируем отрицательный SDR.

    SDR = 10 * log10( ||target||² / ||target - pred||² )

    Хотим максимизировать SDR → минимизируем -SDR.

    sum по последним двум осям (freq, time), keepdim=True — зачем?
      Чтобы не потерять батч-измерение при делении и посчитать SDR для каждого источника.

    eps защищает от деления на ноль для тихих источников.

    Вернуть: скаляр (среднее по батчу и источникам)
    """
    target_power = torch.sum(target**2, dim=[2,3], keepdim=True) + eps
    diff_power = torch.sum((target-pred)**2, dim=[2,3], keepdim=True) + eps
    divRes = torch.div(target_power, diff_power)
    return (-10 * torch.log10(divRes)).mean() # выводим среднее, т.к. нужен скаляр для backpropagation


class Trainer:
    def __init__(self, config: dict):
        self.config = config

        # Определи устройство: cuda если доступно, иначе cpu
        # Выведи информацию об устройстве (и о VRAM если GPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        set_seed(config['seed'])

        self.model = SourceSeparationNet().to(self.device)

        print(f"Параметров: {count_parameters(self.model):,}")

        # Оптимайзер: Adam с lr из config
        # Почему Adam, а не SGD?
        #   Adam адаптирует lr для каждого параметра. Быстрее сходится
        #   на задачах с разреженными градиентами (как спектрограммы).
        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])

        # Планировщик lr: ReduceLROnPlateau
        # Если val_loss не улучшается lr_patience эпох — умножает lr на factor=0.5
        # mode='min' — следим за уменьшением loss
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, 
                                                              mode='min',
                                                              patience=self.config['lr_patience'], 
                                                              factor=0.5)

        self.l1_fn = nn.L1Loss()

        Path(config['checkpoint_dir']).mkdir(exist_ok=True)

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor):
        """Взвешенная сумма L1 и SDR loss."""
        return self.config['l1_weight'] * self.l1_fn(pred, target) + self.config['sdr_weight'] * sdr_loss(pred, target)

    def _forward(self, mixture: torch.Tensor, targets: torch.Tensor):
        """
        Один forward pass: mixture → маски → предсказания → loss.

        Не забудь перенести тензоры на self.device.
        """
        mix_device = mixture.to(self.device)
        tgt_device = targets.to(self.device)

        masks = self.model(mix_device)
        
        predictions = apply_masks(mix_device, masks)
        return self._compute_loss(predictions, tgt_device)

    def train_epoch(self, loader: DataLoader):
        """
        Одна эпоха обучения.

        Шаблон одной итерации:
          optimizer.zero_grad()
          loss = self._forward(...)
          loss.backward()
          clip_grad_norm_  ← зачем? предотвращает взрывной градиент
          optimizer.step()

        Gradient clipping: обрезает норму градиента до grad_clip.
        Без него при больших градиентах веса могут "улететь" в nan.

        Вернуть: средний loss за эпоху
        """
        self.model.train()
        # Цикл по батчам, логируй каждые 20 батчей
        i = 0
        total_loss = 0
        for mixture, targets in loader:
            i += 1
            self.optimizer.zero_grad()
            loss = self._forward(mixture, targets)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.optimizer.step()
            if i%20==0: # Прогресс-чек по 20 батчей
                print(f"Batch: {i}/{len(loader)}    |   Loss: {loss}")

        return total_loss/len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader):
        """
        Одна эпоха валидации. @torch.no_grad() — зачем?
          Отключает вычисление графа вычислений.
          Меньше памяти + быстрее (не нужны градиенты для оценки).

        Вернуть: средний loss за эпоху
        """
        self.model.eval()
        total_loss = 0
        for mixture, targets in loader:
            loss = self._forward(mixture, targets)
            total_loss += loss.item()
        return total_loss/len(loader)

    def save_checkpoint(self, epoch: int, val_loss: float):
        """
        Сохраняет состояние модели и оптимайзера.

        Зачем сохранять optimizer.state_dict?
          Чтобы можно было продолжить обучение с того же места.
          Adam хранит "моменты" для каждого параметра — без них
          первые итерации после загрузки будут нестабильными.
        """
        checkpoints = {
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        path = Path(self.config['checkpoint_dir']) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoints, path)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, start_epoch: int = 1):
        """Основной цикл: эпохи → train → eval → scheduler → checkpoint."""
        best_val_loss = float('inf')

        for epoch in range(start_epoch, self.config['epochs']+1):
            print(f"\n=== Epoch {epoch}/{self.config['epochs']} ===")
            print("Training...")
            train_loss = self.train_epoch(train_loader)
            print("Evaluating...")
            val_loss = self.eval_epoch(val_loader)
            lr = self.optimizer.param_groups[0]['lr']
            saved = ""
            self.scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint(epoch, val_loss)
                saved = " ← checkpoint saved"
            print(f"Epoch {epoch}/{self.config['epochs']} | Train: {train_loss:.4f} | Val: {val_loss:.4f} | LR: {lr:.2e}{saved}")


def load_checkpoint(path: str, device: str = 'cpu'):
    """Загружает модель из файла checkpoint."""
    checkpoint = torch.load(path, map_location=device)
    model = SourceSeparationNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, checkpoint


if __name__ == '__main__':
    train_loader, val_loader, test_loader = get_dataloaders(
        root=CONFIG['musdb_root'],
        batch_size=CONFIG['batch_size'],
    )
    trainer = Trainer(CONFIG)

    start_epoch = 1
    checkpoints = sorted(Path(CONFIG['checkpoint_dir']).glob('*.pt'))
    if checkpoints:
        latest = checkpoints[-1]
        _, ckpt = load_checkpoint(str(latest), device=str(trainer.device))
        trainer.model.load_state_dict(ckpt['model_state_dict'])
        trainer.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        print(f"Возобновление с эпохи {start_epoch}, val_loss={ckpt['val_loss']:.4f}")

    trainer.train(train_loader, val_loader, start_epoch=start_epoch)
    trainer.train(train_loader, val_loader)