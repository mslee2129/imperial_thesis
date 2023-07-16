import librosa
import numpy as np
import os
from pathlib import Path
import math
import matplotlib.pyplot as plt

dir = 'data/nmed-t/stimulus/segment/'
dst = 'data/nmed-t/mel_spectrogram/'

for filename in os.listdir(dir):
    for i in range(20):
        fname = filename[:6] + '_sub' + str(i) + filename[6:]
        fname = fname[:-4]
        print(fname)
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
        plt.savefig(dst + fname + '.png', bbox_inches='tight', pad_inches=0.0)
        plt.clf()
        plt.close()
    
