import gradio as gr
import librosa
import torch
from data_loader import SOURCES, SAMPLE_RATE
from evaluate import separate_track, load_checkpoint

checkpoint_path = './checkpoints/checkpoint_epoch_48.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model, _ = load_checkpoint(checkpoint_path, device)

def separate(audio_path):
    mix, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    result = separate_track(model, mix, device)
    return [(SAMPLE_RATE, result[s]) for s in SOURCES]

gr.Interface(
    fn=separate,
    inputs = gr.Audio(type="filepath"),
    outputs = [gr.Audio(label=s) for s in SOURCES]
).launch()