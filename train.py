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
    'checkpoint_dir': './checkpoints',
    'epochs':         50,
    'batch_size':     4,
    'lr':             1e-3,
    'lr_patience':    5,
    'l1_weight':      0.9,
    'sdr_weight':     0.1,
    'grad_clip':      1.0,
    'seed':           42,
}


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def sdr_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8):
    target_power = torch.sum(target**2, dim=[2,3], keepdim=True) + eps
    diff_power = torch.sum((target-pred)**2, dim=[2,3], keepdim=True) + eps
    return (-10 * torch.log10(torch.div(target_power, diff_power))).mean()


class Trainer:
    def __init__(self, config: dict):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        set_seed(config['seed'])

        self.model = SourceSeparationNet().to(self.device)
        print(f"Параметров: {count_parameters(self.model):,}")

        self.optimizer = optim.Adam(self.model.parameters(), lr=config['lr'])
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer=self.optimizer,
            mode='min',
            patience=self.config['lr_patience'],
            factor=0.5
        )
        self.l1_fn = nn.L1Loss()

        Path(config['checkpoint_dir']).mkdir(exist_ok=True)

    def _compute_loss(self, pred: torch.Tensor, target: torch.Tensor):
        return self.config['l1_weight'] * self.l1_fn(pred, target) + self.config['sdr_weight'] * sdr_loss(pred, target)

    def _forward(self, mixture: torch.Tensor, targets: torch.Tensor):
        mix_device = mixture.to(self.device)
        tgt_device = targets.to(self.device)
        masks = self.model(mix_device)
        predictions = apply_masks(mix_device, masks)
        return self._compute_loss(predictions, tgt_device)

    def train_epoch(self, loader: DataLoader):
        self.model.train()
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
            if i % 20 == 0:
                print(f"Batch: {i}/{len(loader)}    |   Loss: {loss}")
        return total_loss / len(loader)

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader):
        self.model.eval()
        total_loss = 0
        for mixture, targets in loader:
            loss = self._forward(mixture, targets)
            total_loss += loss.item()
        return total_loss / len(loader)

    def save_checkpoint(self, epoch: int, val_loss: float):
        checkpoints = {
            'epoch': epoch,
            'val_loss': val_loss,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()
        }
        path = Path(self.config['checkpoint_dir']) / f"checkpoint_epoch_{epoch}.pt"
        torch.save(checkpoints, path)

    def train(self, train_loader: DataLoader, val_loader: DataLoader, start_epoch: int = 1):
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
    checkpoint = torch.load(path, map_location=device)
    model = SourceSeparationNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
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
