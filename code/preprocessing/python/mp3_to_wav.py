import os
from os import path
from pydub import AudioSegment
from pathlib import Path

dir = 'data\ds002721-master\stimulus\set1\mp3\Soundtrack360_mp3'
dst = 'data\ds002721-master\stimulus\set1\mp3\Soundtrack360_wav'

for filename in os.listdir(dir):
    f = os.path.join(dir, filename)
    sound = AudioSegment.from_mp3(f)
    dst_dir = os.path.join(dst, Path(filename).stem)
    print(dst_dir)
    # sound.export(dst_dir, format="wav")