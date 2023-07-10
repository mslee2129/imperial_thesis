import mne
import numpy as np
import os.path as op
from autoreject import get_rejection_threshold
import pandas as pd


dir = 'data\ds002721-master\sub-01\eeg'
pid = '\sub-01_task-run2'
# eeg_file = op.join(dir, 'sub-01_task-run2_eeg.edf')

def read_edf(dname, fname):
  eeg_file = op.join(dname, fname)
  raw = mne.io.read_raw_edf(eeg_file, verbose=True)
  raw.load_data()
  print(raw.info)
  return raw

raw = read_edf(dir, 'sub-01_task-run2_eeg.edf')

# raw.plot(block=True)

ch_names = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'O2']
ch_types = ['eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg','eeg']
info = mne.create_info(ch_names=ch_names, sfreq=512, ch_types=ch_types)
raw.info = info
# print(raw.info)
def set_1020_montage(raw):
  montage_kind = 'standard_1020'
  raw.set_montage(montage_kind, match_case=False)
  return raw

raw = set_1020_montage(raw)

# raw.plot_sensors(show_names=True, block=True)

# raw.plot_psd(fmax=256)

# filtering
low_cut = 0.1
high_cut = 30
def filter(raw, low_cut, high_cut):
  raw_filt = raw.copy().filter(low_cut, high_cut)
  return raw_filt

raw_filt = filter(raw, low_cut, high_cut)
# raw_filt.plot_psd(fmax=256)
# raw_filt.plot_psd(fmax=10)

# raw.plot(start=5, duration=13)
raw_filt.plot(start=5, duration=13, block=True)

# save filtered as raw as fif
raw_filt.save(dir + pid + '-filt-raw.fif', overwrite=True)
raw.save(dir + pid +'-raw.fif', overwrite=True)

# artifact removal
raw_ica = raw_filt.copy()
# Break raw data into 1 s epochs
tstep = 1.0
events_ica = mne.make_fixed_length_events(raw_ica, duration=tstep)
epochs_ica = mne.Epochs(raw_ica, events_ica,
                        tmin=0.0, tmax=tstep,
                        baseline=None,
                        preload=True)

reject = get_rejection_threshold(epochs_ica)
reject

# ICA parameters
random_state = 42   # ensures ICA is reproducable each time it's run
ica_n_components = .99     # Specify n_components as a decimal to set % explained variance

# Fit ICA
ica = mne.preprocessing.ICA(n_components=ica_n_components,
                            random_state=random_state,
                            )
ica.fit(epochs_ica,
        reject=reject,
        tstep=tstep)

ica.plot_components()

ica.plot_properties(epochs_ica, picks=range(0, ica.n_components_), psd_args={'fmax': high_cut})

ica_z_thresh = 1.96 
eog_indices, eog_scores = ica.find_bads_eog(raw_ica, 
                                            ch_name=['Fp1', 'F8'], 
                                            threshold=ica_z_thresh)
ica.exclude = eog_indices

ica.plot_scores(eog_scores)