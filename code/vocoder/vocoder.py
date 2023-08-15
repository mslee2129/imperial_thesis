import librosa
import soundfile as sf
import numpy as np
from PIL import Image
import matplotlib
import os
from pathlib import Path
from pymcd.mcd import Calculate_MCD

dir = 'checkpoints/'
dst_real = 'results/audio/cgan/real/'
dst_fake = 'results/audio/cgan/fake/'

for d in os.listdir(dir):
    print(d)
    if 'cgan_resnet_9_batch_8' in d:
        d = os.path.join(dir, d, 'web/images/')
        print(d)
        for filename in os.listdir(d):
            if filename == 'epoch100_real_B.png' or filename == 'epoch100_fake_B.png':
                sample = Image.open(d + filename).convert('L')
                sample.thumbnail((128, 128), Image.ANTIALIAS)
                img_arr = np.array(sample, dtype='float64')
                img_arr = img_arr - 80.0
                P = librosa.db_to_power(img_arr)
                wav = librosa.feature.inverse.mel_to_audio(P*1000.0, hop_length=173)
                name = d.split('/')[1] + Path(filename).stem
                dst = dst_real if 'real' in filename else dst_fake
                sf.write(dst+name+'.wav', np.ravel(wav), samplerate=22050)

