import os
import random
import shutil


eegdir = 'data/nmed-t/eeg_spectrogram'
meldir = 'data/nmed-t/mel_spectrogram'
trainA = 'data/nmed-t-prep/A/train'
valA = 'data/nmed-t-prep/A/val'
testA = 'data/nmed-t-prep/A/test'
trainB = 'data/nmed-t-prep/B/train'
valB = 'data/nmed-t-prep/B/val'
testB = 'data/nmed-t-prep/B/test'

tr = 0.8
ts = 0.1
va = 0.1

ext = '.png'

A_list = [filename for filename in os.listdir(eegdir) if os.path.splitext(filename)[-1] in ext]
B_list = [filename for filename in os.listdir(meldir) if os.path.splitext(filename)[-1] in ext]

random.seed(42)

random.shuffle(A_list)

train_size = int(len(A_list) * tr)
val_size = int(len(A_list) * va)
test_size = int(len(A_list) * ts)

for i, f in enumerate(A_list):
    if i < train_size:
        destA = trainA
        destB = trainB
    elif i < train_size + val_size:
        destA = valA
        destB = valB
    else: 
        destA = testA
        destB = testB
    shutil.copy(os.path.join(eegdir, f), os.path.join(destA, f))
    shutil.copy(os.path.join(meldir, f), os.path.join(destB, f))