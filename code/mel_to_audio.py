from vocoder.vocoder import WaveGlowVocoder
import librosa
import torch
import os
from PIL import Image
import numpy as np
import soundfile as sf

dir = 'pix2pix_res/pix2pix_AtoB/test_latest/images/'

WV = WaveGlowVocoder()

for filename in os.listdir(dir):
    if 'B' in filename:
        mel_img = Image.open(dir+filename)
        mel_arr = np.array(mel_img)
        mel_arr = mel_arr - 80.0
        P = librosa.db_to_power(mel_arr)
        wav = librosa.feature.inverse.mel_to_audio(P*1000.0, hop_length=173)

        sf.write(dir+'', np.ravel(wav), samplerate=22050)
        