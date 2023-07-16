import math
import torch
from PIL import Image
import torchvision.transforms as transforms
from waveglow_vocoder import WaveGlowVocoder
import librosa

dir = 'data/ds002721-prep/mel_spectrogram/'
file = 'stimulus1.png'
audio = 'data/ds002721-prep/stimulus/set1/mp3/Soundtrack360_wav/001_0.wav'
filename = 'data/ds002721-prep/stimulus/mp3/Soundtrack360_mp3/001.mp3'

y, sr = y,sr = librosa.load(filename, sr=22050)
y_tensor = torch.from_numpy(y).to(device='cuda', dtype=torch.float32)

WV = WaveGlowVocoder()
mel = WV.wav2mel(y_tensor)

wav = WV.mel2wav(mel)

librosa.output.write_wav(dir+'rec1.wav', wav, sr)