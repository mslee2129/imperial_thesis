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
        S = librosa.feature.melspectrogram(y=y, sr=sr)
        S_dB = librosa.power_to_db(S, ref=np.max)
        im = Image.fromarray(S_dB+80).convert('L')
        width, height = im.size
        img = Image.new(im.mode, (height, height) (255))
        img.paste(im, ((height - width) // 2, 0))
        img.save(dst+fname+'.tiff')
    
