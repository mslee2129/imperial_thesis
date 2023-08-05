import librosa
import soundfile as sf
import numpy as np
from PIL import Image
import matplotlib
import os
from pathlib import Path

dir = 'checkpoints/ccgan_base/web/images/'
dst = 'ccgan_res/ccgan_base/audio/'

for filename in os.listdir(dir):
    if 'B' in filename:
        sample = Image.open(dir + filename).convert('L')
        sample.thumbnail((128, 128), Image.ANTIALIAS)
        img_arr = np.array(sample, dtype='float64')
        img_arr = img_arr - 80.0
        P = librosa.db_to_power(img_arr)
        wav = librosa.feature.inverse.mel_to_audio(P*1000.0, hop_length=173)
        name = Path(filename).stem
        sf.write(dst+name+'.wav', np.ravel(wav), samplerate=22050)