[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
for i=1:31
    path = sprintf('C:\\Users\\ZEPHYRUS\\Desktop\\Imperial\\Thesis\\individual_project\\data\\ds002721-prep\\sub-%02d\\eeg\\', i);
    for j=2:5
        file = sprintf('sub-%02d_task-run%d.set', i, j);
        EEG = pop_loadset('filename', file ,'filepath', path);
        [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, (i-1)*4+(j-2),'study',0); 
    end
end