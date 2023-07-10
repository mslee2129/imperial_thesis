import mne
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# raw = 'data\ds002721-master\sub-01\eegsub-01_task-run2-filt-raw.fif'

# dir = 'data\ds002721-master\sub-01\eeg'
dir = 'data\sub-01_task-genMusic01_eeg.edf'
# raw = mne.io.read_raw_edf(dir + '\sub-01_task-run2-filt-raw.fif', verbose=True)
raw = mne.io.read_raw_edf(dir, verbose=True)
print(raw.info)
# raw.copy().pick_types(eeg=False, stim=True).plot(block=True)
data = raw.get_data()
data = pd.DataFrame(data)
print(data)
music = data.loc[39]
music = music[music>0] * 20

print(music)
# raw.plot(block=True)
