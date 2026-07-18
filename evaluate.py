import librosa
import torch
import torch.nn as nn
import numpy as np
import mir_eval
import musdb

from model import SourceSeparationNet, apply_masks
from preprocessing import (
    audio_to_spectrogram, log_magnitude, normalize, denormalize,
    spectrogram_to_audio, segment_spectrogram, prepare_tensor,
    SEGMENT_FRAMES, SAMPLE_RATE
)
from train import load_checkpoint
from data_loader import SOURCES


def separate_track(model: nn.Module, mix_audio: np.ndarray, device: str = 'cpu'):
    magnitude, phase = audio_to_spectrogram(mix_audio)
    log_mag = log_magnitude(magnitude)
    norm_mag, mean, std = normalize(log_mag)
    mean, std = float(mean), float(std)
    segments = segment_spectrogram(norm_mag)

    preds = {source: [] for source in SOURCES}
    model.eval()
    with torch.no_grad():
        for seg in segments:
            seg_tensor = prepare_tensor(seg).unsqueeze(0).to(device)
            masks = model(seg_tensor)
            prediction = apply_masks(seg_tensor, masks).cpu().numpy()
            for s_idx, source in enumerate(SOURCES):
                preds[source].append(prediction[0, s_idx, :, :])

    phase = phase[:, :len(segments)*SEGMENT_FRAMES]
    result = {}
    for source in SOURCES:
        full_spec = np.concatenate(preds[source], axis=1)
        denorm = denormalize(full_spec, mean, std)
        mag = np.exp(denorm)
        re_audio = spectrogram_to_audio(mag, phase)
        result[source] = re_audio

    return result


def compute_sdr(ref_audio: np.ndarray, est_audio: np.ndarray):
    min_len = min(len(ref_audio), len(est_audio))
    ref_audio, est_audio = ref_audio[np.newaxis, :min_len], est_audio[np.newaxis, :min_len]
    sdr, _, _, _ = mir_eval.separation.bss_eval_sources(ref_audio, est_audio)
    return float(sdr[0])


def evaluate_model(model: nn.Module, musdb_root: str,
                   device: str = 'cpu', n_tracks: int = None): # type: ignore
    test_db = musdb.DB(root=musdb_root, subsets='test', is_wav=True)
    tracks = list(test_db.tracks)

    if n_tracks:
        tracks = tracks[:n_tracks]
    all_sdrs = {source: [] for source in SOURCES}

    for i, track in enumerate(tracks):
        print(f'Track {i+1}/{len(tracks)}: {track.name}')
        mono = librosa.resample(track.audio.mean(axis=1), orig_sr=track.rate, target_sr=SAMPLE_RATE)
        preds = separate_track(model, mono, device=device)
        for source in SOURCES:
            ref = librosa.resample(track.targets[source].audio.mean(axis=1).astype(np.float32),
                                   orig_sr=track.rate,
                                   target_sr=SAMPLE_RATE)
            sdr = compute_sdr(ref, preds[source])
            all_sdrs[source].append(sdr)

    mean_sdr = {source: np.mean(all_sdrs[source]) for source in SOURCES}
    mean_sdr['average'] = np.mean(list(mean_sdr.values()))

    for source, sdr in mean_sdr.items():
        print(f"  {source:10s}: {sdr:.2f} dB")

    return mean_sdr


if __name__ == '__main__':
    import sys

    checkpoint_path = sys.argv[1] if len(sys.argv) > 1 else './checkpoints/best.pt'
    musdb_root      = sys.argv[2] if len(sys.argv) > 2 else './musdb18'

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, ckpt = load_checkpoint(checkpoint_path, device)
    print(f"Загружен чекпоинт: эпоха {ckpt['epoch']}, val_loss={ckpt['val_loss']:.4f}")

    evaluate_model(model, musdb_root, device=device)
