import librosa
import numpy as np
import os
from pathlib import Path
import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import math

dir = 'data\ds002721-prep\stimulus\set1\mp3\Soundtrack360_wav'
dst = 'data\ds002721-prep\mel_spectrogram'

module = hub.KerasLayer('https://tfhub.dev/google/soundstream/mel/decoder/music/1')

# for filename in os.listdir(dir):
#     if filename[-3:] != 'mp3':

# filename = 'data/ds002721-prep/stimulus/mp3/Soundtrack360_mp3/001.mp3'
# y, sr= librosa.load(filename)
# fig, ax = plt.subplots()
# librosa.feature.melspectrogram(y=y, sr=sr)
# S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128, fmax=8000)
# S_dB = librosa.power_to_db(S, ref=np.max)
# img = librosa.display.specshow(S_dB, x_axis='time',
#                         y_axis='mel', sr=sr,
#                         fmax=8000, ax=ax)
# fig.colorbar(img, ax=ax, format='%+2.0f dB')
# ax.set(title='Mel-frequency spectrogram')
# plt.savefig(dst + '/stimulus1.png')

SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 320
WIN_LENGTH = 640
N_MEL_CHANNELS = 128
MEL_FMIN = 0.0
MEL_FMAX = int(SAMPLE_RATE // 2)
CLIP_VALUE_MIN = 1e-5
CLIP_VALUE_MAX = 1e8

MEL_BASIS = tf.signal.linear_to_mel_weight_matrix(
    num_mel_bins=N_MEL_CHANNELS,
    num_spectrogram_bins=N_FFT // 2 + 1,
    sample_rate=SAMPLE_RATE,
    lower_edge_hertz=MEL_FMIN,
    upper_edge_hertz=MEL_FMAX)

def calculate_spectrogram(samples):
  """Calculate mel spectrogram using the parameters the model expects."""
  fft = tf.signal.stft(
      samples,
      frame_length=WIN_LENGTH,
      frame_step=HOP_LENGTH,
      fft_length=N_FFT,
      window_fn=tf.signal.hann_window,
      pad_end=True)
  fft_modulus = tf.abs(fft)

  output = tf.matmul(fft_modulus, MEL_BASIS)

  output = tf.clip_by_value(
      output,
      clip_value_min=CLIP_VALUE_MIN,
      clip_value_max=CLIP_VALUE_MAX)
  output = tf.math.log(output)
  return output

# Load a music sample from the GTZAN dataset.
# gtzan = dst + '/stimulus1.png'
# Convert an example from int to float.
# samples = tf.cast(next(iter(gtzan))['audio'] / 32768, dtype=tf.float32)
# Add batch dimension.
# samples = tf.expand_dims(samples, axis=0)
# Compute a mel-spectrogram.
# spectrogram = calculate_spectrogram(dst + '/stimulus1.png')
# Reconstruct the audio from a mel-spectrogram using a SoundStream decoder.
reconstructed_samples = module(dst + '/stimulus1.png')




