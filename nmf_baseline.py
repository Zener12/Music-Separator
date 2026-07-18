import librosa
import numpy as np
import musdb
import mir_eval
from sklearn.decomposition import NMF
from pathlib import Path

from evaluate import compute_sdr
from preprocessing import audio_to_spectrogram, denormalize, spectrogram_to_audio, N_FFT, HOP_LENGTH
from data_loader import SOURCES

N_COMPONENTS = 16
MAX_ITER     = 500
SAMPLE_RATE  = 22050


def train_nmf_dictionaries(n_tracks: int = None):
    cache_dir = Path('./cache')
    total = len(list(cache_dir.glob('*_mix.npy')))
    n = n_tracks if n_tracks else total
    all_specs = {source: [] for source in SOURCES}
    print(f"Загрузка спектрограмм из кэша ({n} треков)...")
    for i in range(n):
        print(f"  [{i+1}/{n}] track_{i:03d}")
        mean, std = np.load(cache_dir / f'track_{i:03d}_stats.npy')
        for source in SOURCES:
            norm_spec = np.load(cache_dir / f'track_{i:03d}_{source}.npy')
            denorm_spec = denormalize(norm_spec, mean, std)
            all_specs[source].append(np.exp(denorm_spec)[:, ::4])

    dictionaries = {}
    for source in SOURCES:
        print(f"Обучение NMF: {source}...")
        V = np.concatenate(all_specs[source], axis=1)
        nmf = NMF(n_components=N_COMPONENTS, init='nndsvda', max_iter=MAX_ITER)
        nmf.fit(V.T)
        dictionaries[source] = nmf.components_.T

    print("Словари обучены.")
    return dictionaries


def separate_with_nmf(mix_audio: np.ndarray, dictionaries: dict):
    magnitude, phase = audio_to_spectrogram(mix_audio, N_FFT, HOP_LENGTH)
    magnitude = magnitude.astype(np.float32)
    W_all = np.concatenate([dictionaries[source] for source in SOURCES], axis=1)
    nmf = NMF(n_components=W_all.shape[1], init='custom', max_iter=MAX_ITER)
    nmf.components_ = W_all.T
    nmf.n_components_ = W_all.shape[1]
    activations = nmf.transform(magnitude.T)

    result = {}
    for s_idx, source in enumerate(SOURCES):
        start = s_idx * N_COMPONENTS
        H_src = activations[:, start:start + N_COMPONENTS].T
        W_src = dictionaries[source]
        V_src = W_src @ H_src
        V_total = W_all @ activations.T
        mask = V_src / (V_total + 1e-8)
        separated_mag = mask * magnitude
        audio = spectrogram_to_audio(separated_mag, phase)
        result[source] = audio

    return result


def evaluate_nmf(musdb_root: str, dictionaries: dict, n_tracks: int = None):
    test_db = musdb.DB(musdb_root, subsets='test', is_wav=True)
    tracks = list(test_db) # type: ignore

    if n_tracks:
        tracks = tracks[:n_tracks]

    all_sdrs = {source: [] for source in SOURCES}
    for i, track in enumerate(tracks):
        print(f'Track: {i+1}/{len(tracks)}: {track.name}')
        mono = track.audio.mean(axis=1)
        resample = librosa.resample(mono, orig_sr=track.rate, target_sr=SAMPLE_RATE)
        preds = separate_with_nmf(resample, dictionaries)
        for source in SOURCES:
            s_res = librosa.resample(track.targets[source].audio.mean(axis=1),
                                     orig_sr=track.rate,
                                     target_sr=SAMPLE_RATE)
            sdr = compute_sdr(s_res, preds[source])
            all_sdrs[source].append(sdr)

    mean_sdr = {source: np.mean(all_sdrs[source]) for source in SOURCES}
    mean_sdr['average'] = np.mean(list(mean_sdr.values()))

    for source, sdr in mean_sdr.items():
        print(f"  {source:10s}: {sdr:.2f} dB")

    return mean_sdr


if __name__ == '__main__':
    musdb_root = './musdb18'
    dicts = train_nmf_dictionaries()
    Path('checkpoints').mkdir(exist_ok=True)
    np.save('checkpoints/nmf_dictionaries.npy', dicts, allow_pickle=True)
    evaluate_nmf(musdb_root, dicts)
