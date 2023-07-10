import os
dir = 'data\ds002721-master\stimulus\set1\mp3\Soundtrack360_mp3'
for filename in os.listdir(dir):
    if filename[-3:] != 'mp3':
        dst = os.path.join(dir, filename)
        os.remove(dst)