import numpy as np
import mne
import librosa
import matplotlib.pyplot as plt
import soundfile as sf
import math
from PIL import Image

fname = 'data/nmed-t/eeg_spectrogram/stim1/stim1_sub0_0.png'

img = Image.open(fname)
print(img.size)
# img.load()
# data = np.asarray(img)
# print(data)
# print(data.shape)

f2 = 'data/nmed-t/mel_spectrogram/stim1_0.png'

img = Image.open(f2)
print(img.size)
# img.load()
# data = np.asarray(img)
# print(data)
# print(data.shape)