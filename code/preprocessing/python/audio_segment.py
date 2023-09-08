'''
Code for audio segmentation
'''

import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pathlib import Path

path = 'data/nmed-t/stimulus/original'
dst = 'data/nmed-t/stimulus/segment'

for filename in os.listdir(path):
    mp3 = os.path.join(path, filename)
    # print(dir)
    sound = AudioSegment.from_mp3(mp3)
    # fname = Path(filename).stem + '.wav'
    for sec in range(0, len(sound), 1000): # segment into 1 sec until 12 secs
        fname = Path(filename).stem + '_' + str(int(sec/1000)) + '.wav'
        print(fname)
        seg = sound[sec:sec+1000]
        dir = os.path.join(dst, fname)
        seg.export(dir, format='wav')
