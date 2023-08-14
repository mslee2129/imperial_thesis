import librosa
from pymcd.mcd import Calculate_MCD
import os
from pathlib import Path
import matplotlib.pyplot as plt

dir_real = 'results/audio/cnn/real/'
dir_fake = 'results/audio/cnn/fake/'
dst_real = 'results/eval/cnn/waveform/real/'
dst_fake = 'results/eval/cnn/waveform/fake/'

mcd_dict = {}

# for f in os.listdir(dir_real):
#     n = Path(f).stem
#     print(n)
#     mcd_toolbox = Calculate_MCD(MCD_mode='plain')
#     mcd_value = mcd_toolbox.calculate_mcd(dir_real+n+'.wav', dir_fake+n+'.wav')
#     print(mcd_value)
#     mcd_dict[n[:-8]] = mcd_value

for f in os.listdir(dir_real):
    n = Path(f).stem
    y, sr = librosa.load(dir_real+f)
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title=n)
    ax.label_outer()
    fig.savefig(dst_real+n+'.png')

for f in os.listdir(dir_fake):
    n = Path(f).stem
    y, sr = librosa.load(dir_fake+f)
    fig, ax = plt.subplots()
    librosa.display.waveshow(y, sr=sr, ax=ax)
    ax.set(title=n)
    ax.label_outer()
    fig.savefig(dst_fake+n+'.png')