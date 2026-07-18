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
    def __init__(self, root: str, subset: str = 'train', track_start=0, track_end=None, download: bool = False):
        self.sources = SOURCES
        self.index = []
        self.cache_dir = Path('./cache')
        self.cache_dir.mkdir(exist_ok=True)
        self.track_start = track_start
        self.track_end = track_end

        self.db = musdb.DB(root=root, subsets=subset, is_wav=True)
        self._preprocess_all(self.db)

    def _preprocess_all(self, db):
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
        return len(self.index)

    def __getitem__(self, i: int):
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
    train_dataset = MUSDBDataset(root=root, subset='train', track_start=0, track_end=80)
    val_dataset = MUSDBDataset(root=root, subset='train', track_start=80, track_end=100)
    test_dataset = MUSDBDataset(root=root, subset='test')
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return (train_loader, val_loader, test_loader)
