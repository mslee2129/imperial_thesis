import librosa
import numpy as np
import os
from pathlib import Path
import math
import matplotlib.pyplot as plt
from PIL import Image

dir = 'data/nmed-t/stimulus/segment/'
dst = 'data/nmed-t/mel_spectrogram/'

for filename in os.listdir(dir):
    for i in range(20):
        fname = filename[:6] + '_sub' + str(i) + filename[6:]
        fname = fname[:-4]
        print(fname)
        y, sr = librosa.load(dir + filename)
        S = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=173)
        S_dB = librosa.power_to_db(S, ref=np.max)
        im = Image.fromarray(S_dB+80).convert('L')
        im.save(dst+fname+'.tiff')
    
