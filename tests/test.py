import numpy as np
import mne
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import math
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_hub as hub

module = hub.KerasLayer('https://tfhub.dev/google/soundstream/mel/decoder/music/1')


filename = 'data\stimuli\stimuli hvha1.wav'

y, sr= librosa.load(filename)


fig, ax = plt.subplots()
img = librosa.display.waveshow(y, sr=sr, ax=ax)
plt.show()

librosa.feature.melspectrogram(y=y, sr=sr)

S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128,
                                    fmax=8000)


# S_dB = librosa.power_to_db(S, ref=np.max)

# img = librosa.display.specshow(S_dB, x_axis='time',

#                         y_axis='mel', sr=sr,

#                         fmax=8000, ax=ax)

# fig.colorbar(img, ax=ax, format='%+2.0f dB')

# ax.set(title='Mel-frequency spectrogram')
# plt.savefig('data/spectrograms/stimulus1.png')

R = librosa.feature.inverse.mel_to_audio(S)

sf.write('data/reconstructed/stim1_reconstructed.wav', R, sr)