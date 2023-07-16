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


dir = "C:/Users/ZEPHYRUS/Desktop/Imperial/Thesis/individual_project/data/ds002721-prep/"

def read_eeg(dname, fname):
  eeg_file = os.path.join(dname, fname)
  raw = mne.io.read_raw_eeglab(eeg_file, verbose=True)
  return raw


# for d in os.listdir(dir):
#   if 'sub' in d:
#     fpath = dir + d + '/eeg/'
#     for filename in os.listdir(fpath):
#       if 'ica.set' in filename:
#         # print(filename)
#         raw = read_eeg(fpath, filename)
#         annot = mne.read_annotations(fpath + filename)
#         annot = annot.rename({'boundary': 'bad'})
#         raw.set_annotations(annot)
#         annot_df = raw.annotations.to_data_frame()
#         onset_list = list(annot_df.loc[annot_df['description'] == '788'].onset)
#         print(annot_df)
#         # annot_df['duration'] = 

#         # if onset of next bad is < 12 sec after onset of 788:
        
#         # tmax = onset + (onset of next bad - onset)
#         # crop raw from onset to tmax

#         # else:
#         # crop raw from onset to onset + 12

#         epochs = mne.make_fixed_length_epochs(raw, duration=epoch_duration, reject_by_annotation=True, overlap=0.5)
#         data_array = epochs.get_data()
#         for data in data_array: 
#           print(data)
#           # fig, ax = plt.subplots()
#           # img = ax.imshow(data, cmap='magma', interpolation='nearest', aspect='auto')
#           # plt.savefig()
#           break
#     break




raw = read_eeg(dir, 'sub-03/eeg/sub-03_task-run3_ica.set')
annot = mne.read_annotations(dir + 'sub-03/eeg/sub-03_task-run4_ica.set')
annot_df = raw.annotations.to_data_frame()
onset_list = list(annot_df.loc[annot_df['description'] == '788'].onset)
print(onset_list)
raw.plot(block=True)