import mne
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt

# raw = 'data\ds002721-master\sub-01\eegsub-01_task-run2-filt-raw.fif'

dir = 'data\ds002721-master\sub-01\eeg'
raw = mne.io.read_raw_fif(dir + '\sub-01_task-run2-filt-raw.fif', verbose=True)

# load events tsv file (all events)
events_df = pd.read_csv(dir + '\sub-01_task-run2_events.tsv', sep='\t')
# print(events_df)
# print(events.trial_type.unique())
events = np.array(events_df)
# events = np.ndarray(events)
events[:,2] = events[:,2].astype(int)
# print(events)

event_to_id = {'Music played': 788}
for i in events_df.trial_type.unique():
    if i >= 301 and i <= 660:
        event_to_id['Stimulus'+str(i-300)] = i
    elif i == 257:
        event_to_id['artifact:EOG'] = i
    elif i == 259:
        event_to_id['artifact:EMG/Muscle'] = i
    elif i == 263:
        event_to_id['artifact:50/60 Hz mains interference'] = i


event_mapping = {'788.0': 788}
for i in events_df.trial_type.unique():
    if i >= 301 and i <= 660:
        event_mapping[str(float(i))] = i
    elif i == 257:
        event_mapping[str(float(i))] = i
    elif i == 259:
        event_mapping[str(float(i))] = i
    elif i == 263:
        event_mapping[str(float(i))] = i


# print('Event mapping: ', event_mapping)

annot = mne.Annotations(
    onset=events[:,0],
    duration=events[:,1],
    description=events[:,2]
)

# print(annot)

raw.set_annotations(annot)

events_from_annot, event_dict = mne.events_from_annotations(raw, event_id=event_mapping)

# print(events_from_annot)
# print(event_dict)

# plot events
fig, ax = plt.subplots(figsize=[15, 5])

mne.viz.plot_events(events_from_annot, raw.info['sfreq'],  
                    event_id=event_to_id,                    
                    axes=ax)
plt.show()

# epoching
