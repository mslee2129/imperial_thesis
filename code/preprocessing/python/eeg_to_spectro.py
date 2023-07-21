import scipy.io as sio
import matplotlib.pyplot as plt
import os
import mne
import numpy as np
import matplotlib
from PIL import Image


dir = 'data/nmed-t/eeg/'
dst = 'data/nmed-t/eeg_spectrogram/'

def read_mat(filename):
    mat = sio.loadmat(dir+filename)
    return mat

n = 1
for fname in os.listdir(dir):
    mat = read_mat(fname)
    key = list(mat.keys())[3]
    eeg = mat[key]
    sfreq = 125
    for i in range(eeg.shape[2]):
        print('stim' + str(n) + '_sub' + str(i))
        sub = eeg[:,:,i]
        for j in range(0, sub.shape[1], 125):
            if n < 10:
                name = dst + 'stim0' + str(n) + '_sub' + str(i) + '_' + str(int(j/125)) + '.tiff'
            else:
                name = dst + 'stim' + str(n) + '_sub' + str(i) + '_' + str(int(j/125)) + '.tiff'

            sample = sub[:,j:j+125]
            psd, freqs = mne.time_frequency.psd_array_welch(sample, sfreq, fmin=1, fmax=63.5, n_fft=256, n_per_seg=256)
            psd = np.pad(psd, ((0, 3), (0, 0)), mode='edge')
            cmap = plt.get_cmap('gray')
            norm = matplotlib.colors.Normalize()
            im = Image.fromarray(np.uint8(cmap(norm(psd))*255))
            im.save(name)
    n += 1
