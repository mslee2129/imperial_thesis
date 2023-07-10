import mne
import os
from pathlib import Path
from datetime import datetime, timedelta
from scipy.signal import spectrogram
from matplotlib.pyplot import specgram
import matplotlib.pyplot as plt
import numpy as np


epoch_duration = 1.0
overlap = 0.5
n_fft = 256
window = 'hann'

dir = "C:/Users/ZEPHYRUS/Desktop/Imperial/Thesis/individual_project/data/ds002721-prep/sub-01/eeg/"
pid = 'sub-01_task-run2_ica.set'

def read_eeg(dname, fname):
  eeg_file = os.path.join(dname, fname)
  raw = mne.io.read_raw_eeglab(eeg_file, verbose=True)
  # raw.load_data()
  # print(raw.info)
  return raw

raw = read_eeg(dir, pid)

# raw_plot = raw.plot(block=True)


annot = mne.read_annotations(dir + 'sub-01_task-run2_ica.set')
# events = mne.events_from_annotations(raw)
# print(annot)
# print(events)



annot = annot.rename({'boundary': 'bad'})
raw.set_annotations(annot)

annot_df = raw.annotations.to_data_frame()
print(annot_df)
# onset = annot_df['onset']
# print(annot_df)
# raw = raw.copy().crop(tmin=tmin, tmax=tmax, include_tmax=True)

# onset_list = list(annot_df.loc[annot_df['description'] == '788'].onset)


epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, reject_by_annotation=True, overlap=0.5)

data_array = epochs.get_data()

print(data_array[0].shape)
sample = data_array[0]

fig, ax = plt.subplots(figsize=(10, 10))  # Adjust the figure size as needed

# Plot the EEG data
img = ax.imshow(sample, cmap='Greys', interpolation='nearest', aspect='auto')

# Customize the plot
ax.set_ylabel('Channels', labelpad=16)
ax.set_xlabel('Samples', labelpad=16)

# Make the image square
fig.tight_layout()

# Show the plot
plt.show()
fig .savefig('code/preprocessing/samples/raw_eeg_spectrogram.png')
