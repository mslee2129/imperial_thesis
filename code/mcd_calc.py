'''
Run to compute MCD metric and compute waveform. Change directory below
'''

import librosa
from pymcd.mcd import Calculate_MCD
import os
from pathlib import Path
import matplotlib.pyplot as plt

dir_real = 'results/audio/cgan-25/real/'
dir_fake = 'results/audio/cgan-25/fake/'
dst_real = 'results/eval/cgan-25/waveform/real/'
dst_fake = 'results/eval/cgan-25/waveform/fake/'

mcd_dict = {}

for dir in os.listdir(dir_real):
    mcd = 0
    count = 0
    d = os.path.join(dir_real, dir)
    for f in os.listdir(d):
        n = Path(f).stem
        print(n)
        fake_n = n[:-6] + 'fake_B'
        mcd_toolbox = Calculate_MCD(MCD_mode='plain')
        mcd_value = mcd_toolbox.calculate_mcd(d+'/'+n+'.wav', dir_fake+dir+'/'+fake_n+'.wav')
        print(mcd_value)
        mcd += mcd_value
        count += 1
        print(count)
    mcd_dict[dir] = mcd/count

print(mcd_dict)
    
for dir in os.listdir(dir_real):
    d = os.path.join(dir_real, dir)
    dst = os.path.join(dst_real, dir)
    for f in os.listdir(d):
        n = Path(f).stem
        y, sr = librosa.load(d+'/'+f)
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set(title=n)
        ax.label_outer()
        fig.savefig(dst+'/'+n+'.png')
        plt.close()

for dir in os.listdir(dir_fake):
    d = os.path.join(dir_fake, dir)
    dst = os.path.join(dst_fake, dir)
    for f in os.listdir(d):
        n = Path(f).stem
        y, sr = librosa.load(d+'/'+f)
        fig, ax = plt.subplots()
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set(title=n)
        ax.label_outer()
        fig.savefig(dst+'/'+n+'.png')
        plt.close()