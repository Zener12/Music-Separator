import musdb

mus = musdb.DB(root='./musdb18', subsets='train', is_wav=True, download=False)
print(f"Треков: {len(mus)}")

track = mus[0]
print(f"Название: {track.name}")
print(f"Частота: {track.rate} Гц")
print(f"Форма аудио смеси: {track.audio.shape}")
print(f"Форма вокала:       {track.targets['vocals'].audio.shape}")
mus[0]