import librosa
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import math

dir = 'data/ds002721-prep/stimulus/mp3/Soundtrack360/'
dst = 'data/ds002721-prep/mel_spectrogram/'

for filename in os.listdir(dir):
    if filename[-3:] == 'wav':
        y, sr = librosa.load(dir + filename)
        fig, ax = plt.subplots()
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        img = librosa.display.specshow(S_dB, x_axis='time',
                                        y_axis='mel', sr=sr,
                                        fmax=8000, ax=ax)
        cb = fig.colorbar(img, ax=ax, format='%+2.0f dB')
        plt.axis('off')
        cb.remove()
        plt.savefig(dst + Path(filename).stem + '.png', bbox_inches='tight', pad_inches=0.0)
        plt.clf()
        plt.close()
        print(filename)