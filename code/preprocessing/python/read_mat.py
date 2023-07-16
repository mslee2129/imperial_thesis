import scipy.io as sio
import matplotlib.pyplot as plt
import os

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
    for i in range(eeg.shape[2]):
        print('stim' + str(n) + '_sub' + str(i))
        sub = eeg[:,:,i]
        for j in range(0, sub.shape[1], 125):
            if n < 10:
                name = dst + 'stim0' + str(n) + '_sub' + str(i) + '_' + str(int(j/125)) + '.png'
            else:
                name = dst + 'stim' + str(n) + '_sub' + str(i) + '_' + str(int(j/125)) + '.png'

            sample = sub[:,j:j+125]
            fig, ax = plt.subplots()
            img = ax.imshow(sample, cmap='magma', interpolation='nearest', aspect='auto')
            plt.axis('off')
            plt.savefig(name, bbox_inches='tight', pad_inches=0.0)
            plt.clf()
            plt.close()
    n += 1
