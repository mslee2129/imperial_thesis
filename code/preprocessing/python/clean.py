import os
dir = 'data/nmed-t/eeg_spectrogram'
for filename in os.listdir(dir):
    dst = os.path.join(dir, filename)
    os.remove(dst)