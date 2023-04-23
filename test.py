import numpy as np
import mne
import librosa
import matplotlib.pyplot as plt
import soundfile as sf

filename = 'data\stimuli\stimuli classical p1_chopin-n10-op12-bertoglio.mp3'

y, sr= librosa.load(filename)

librosa.feature.melspectrogram(y=y, sr=sr)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,

                                    fmax=8000)

fig, ax = plt.subplots()

S_dB = librosa.power_to_db(S, ref=np.max)

img = librosa.display.specshow(S_dB, x_axis='time',

                         y_axis='mel', sr=sr,

                         fmax=8000, ax=ax)

fig.colorbar(img, ax=ax, format='%+2.0f dB')

ax.set(title='Mel-frequency spectrogram')
plt.savefig('data/spectrograms/stimulus1.png')

R = librosa.feature.inverse.mel_to_audio(S)

sf.write('stim1_reconstructed.wav', R, sr)