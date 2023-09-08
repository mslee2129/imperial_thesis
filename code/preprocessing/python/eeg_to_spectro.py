import scipy.io as sio
import matplotlib.pyplot as plt
import os
import mne
import numpy as np
import matplotlib
from PIL import Image
import pathlib

dir = 'data/nmed-t/eeg/'
dst = 'data/nmed-t/eeg_spect_25/'

def read_mat(filename):
    mat = sio.loadmat(dir+filename)
    return mat


for fname in os.listdir(dir):
    if pathlib.Path(fname).suffix == '.mat':
        mat = read_mat(fname)
        n = fname[5]
        key = list(mat.keys())[3]
        eeg = mat[key]
        sfreq = 125
        for i in range(eeg.shape[2]):
            print('stim' + str(n) + '_sub' + str(i))
            sub = eeg[:,:,i]
            for j in range(25, sub.shape[1], 125):
                if int(n) > 0:
                    name = dst + 'stim0' + n + '_sub' + str(i) + '_' + str(int(j/125)) + '.tiff'
                else:
                    name = dst + 'stim10' + '_sub' + str(i) + '_' + str(int(j/125)) + '.tiff'

                sample = sub[:,j:j+125]
                psd, freqs = mne.time_frequency.psd_array_welch(sample, sfreq, fmin=1, fmax=63.5, n_fft=256, n_per_seg=256, verbose=False)
                psd = np.pad(psd, ((0, 3), (0, 0)), mode='edge')
                cmap = plt.get_cmap('gray')
                norm = matplotlib.colors.Normalize()
                im = Image.fromarray(np.uint8(cmap(norm(psd))*255))
                im.save(name)
