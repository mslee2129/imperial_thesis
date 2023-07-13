import librosa
import numpy as np
import os
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pathlib import Path

path = 'data\ds002721-master\stimulus\set1\mp3\Soundtrack360_mp3'

for filename in os.listdir(path):
    dir = os.path.join(path, filename)
    # print(dir)
    sound = AudioSegment.from_mp3(dir)
    # print(len(sound))
    for sec in range(0, 11001, 1000): # segment into 1 sec until 12 secs
        fname = Path(filename).stem + '_' + str(int(sec/1000)) + '.wav'
        seg = sound[sec:sec+1000]
        dst = os.path.join(path, fname)
        seg.export(dst, format='wav')

