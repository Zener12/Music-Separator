import numpy as np
import librosa
import torch


SAMPLE_RATE    = 22050
N_FFT          = 2048
HOP_LENGTH     = 512
SEGMENT_SEC    = 3.0
SEGMENT_FRAMES = int(np.ceil(SEGMENT_SEC * SAMPLE_RATE / HOP_LENGTH))


def load_audio(path: str, sr: int = SAMPLE_RATE):
    audio = librosa.load(path, sr=sr, mono=True)
    return audio[0]


def audio_to_spectrogram(audio: np.ndarray, n_fft: int = N_FFT, hop_length: int = HOP_LENGTH):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    return magnitude, phase


def spectrogram_to_audio(magnitude: np.ndarray, phase: np.ndarray, hop_length: int = HOP_LENGTH):
    z = magnitude * np.exp(1j * phase)
    return librosa.istft(z, hop_length=hop_length)


def log_magnitude(magnitude: np.ndarray, eps: float = 1e-7):
    return np.log(magnitude + eps)


def normalize(spec: np.ndarray):
    mean = np.mean(spec)
    std = np.std(spec) + 1e-7
    return (spec - mean) / std, mean, std


def denormalize(spec_norm: np.ndarray, mean: float, std: float):
    return spec_norm * std + mean


def segment_spectrogram(magnitude: np.ndarray, segment_frames: int = SEGMENT_FRAMES):
    segments = []
    for i in range(0, magnitude.shape[1] - segment_frames + 1, segment_frames):
        segment = magnitude[:, i:i+segment_frames]
        segments.append(segment)
    return segments


def prepare_tensor(magnitude_segment: np.ndarray):
    return torch.FloatTensor(magnitude_segment.copy()).unsqueeze(0)
