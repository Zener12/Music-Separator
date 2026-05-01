import soundfile as sf
import musdb
import librosa
import numpy as np
from train import load_checkpoint
from evaluate import separate_track
from preprocessing import SAMPLE_RATE

from pathlib import Path

musdb_root = './musdb18'
checkpoint_path = './checkpoints/checkpoint_epoch_48.pt'
results_path = './results_demo'

device = 'cuda'
model, _ = load_checkpoint(checkpoint_path, device)
model.to(device)

db = musdb.DB(musdb_root, subsets='test', is_wav=True)
track = db.tracks[1]
print(f"Track: {track.name}")

mono = librosa.resample(track.audio.mean(axis=1).astype(np.float32),
                        orig_sr=track.rate,
                        target_sr=SAMPLE_RATE)

results = separate_track(model, mono, device=device)

track_dir = Path(results_path) / track.name
track_dir.mkdir(parents=True, exist_ok=True)

sf.write(track_dir / 'mix.wav', mono, SAMPLE_RATE)
for source, audio in results.items():
    sf.write(track_dir / f"{source}.wav", audio, SAMPLE_RATE)
    print(f'Saved: {track_dir / source}.wav')