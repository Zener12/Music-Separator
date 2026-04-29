import musdb
import matplotlib.pyplot as plt
import torch

mus = musdb.DB(root='./musdb18', subsets='train', is_wav=True, download=False)
print(f"Треков: {len(mus)}")

track = mus[0]
# print(f"Название: {track.name}")
# print(f"Частота: {track.rate} Гц")
# print(f"Форма аудио смеси: {track.audio.shape}")
# print(f"Форма вокала:       {track.targets['vocals'].audio.shape}")
# print(f"Форма ударных:      {track.targets['drums'].audio.shape}")
# print(f"Форма баса:        {track.targets['bass'].audio.shape}")

# # Визуализация аудио сигнала
# plt.figure(figsize=(12, 4))
# plt.plot(track.audio)
# plt.title('Аудио сигнал смеси')
# plt.xlabel('Время')
# plt.ylabel('Амплитуда')
# plt.show()
# print(track.stems)