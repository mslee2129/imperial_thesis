[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

for i = 1:31
    % Form the file paths with the updated integer
    for j = 2:5
        prepPath = sprintf('C:\\Users\\ZEPHYRUS\\Desktop\\Imperial\\Thesis\\individual_project\\data\\ds002721-prep\\sub-%02d\\eeg', i);
        eegFilePath = sprintf('C:\\Users\\ZEPHYRUS\\Desktop\\Imperial\\Thesis\\individual_project\\data\\ds002721-raw\\sub-%02d\\eeg\\sub-%02d_task-run%d_eeg.edf', i, i, j);
        eventFilePath = sprintf('C:\\Users\\ZEPHYRUS\\Desktop\\Imperial\\Thesis\\individual_project\\data\\ds002721-prep\\sub-%02d\\eeg\\sub-%02d_task-run%d_events.tsv', i, i, j);
        filename = sprintf('sub-%02d_task-run%d.set', i, j);
        disp(eegFilePath);
        disp(eventFilePath);
        disp(prepPath);
        % Load EEG data
        EEG = pop_biosig(eegFilePath);
        [ALLEEG, EEG, CURRENTSET] = pop_newset(ALLEEG, EEG, 0, 'gui', 'off');
    
        % Import events
        EEG = pop_importevent(EEG, 'event', eventFilePath, 'fields', {'latency', 'duration', 'type'}, 'timeunit', 1);
        [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

        EEG=pop_chanedit(EEG, []);
        [ALLEEG, EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);

        EEG = pop_saveset( EEG, 'filename', filename,'filepath', prepPath);
        [ALLEEG EEG] = eeg_store(ALLEEG, EEG, CURRENTSET);
    
        % Perform further processing or analysis on the EEG data
    
        % Clear variables to prepare for the next iteration
        clear EEG CURRENTSET;
    end
end
