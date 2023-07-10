[EEG ALLEEG CURRENTSET ALLCOM] = eeglab;

for i = 1:1
    % Form the file paths with the updated integer
    files = {};
    newFiles = {};
    for j = 2:5
        prepPath = sprintf('C:\\Users\\ZEPHYRUS\\Desktop\\Imperial\\Thesis\\individual_project\\data\\ds002721-prep\\sub-%02d\\eeg', i);
        eegFilePath = sprintf('C:\\Users\\ZEPHYRUS\\Desktop\\Imperial\\Thesis\\individual_project\\data\\ds002721-prep\\sub-%02d\\eeg\\sub-%02d_task-run%d_prep.set', i, i, j);
        filename = sprintf('sub-%02d_task-run%d_rej.set', i, j);

        % load data
        files{end+1} = eegFilePath;
        newFiles{end+1} = filename;
    end
    disp(files);
    % load subject
    [ALLEEG EEG] = pop_loadset('filename', files);
    
    % load files
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 4,'retrieve', [1:4] ,'study',0); 
    chanlocs = EEG.chanlocs;

    % remove bad channels and bad portions of data
    EEG = pop_clean_rawdata(EEG, 'FlatlineCriterion',4,'ChannelCriterion',0.85,'LineNoiseCriterion',4,'Highpass','off',...
        'BurstCriterion',20,'WindowCriterion',0.25,'BurstRejection','on','Distance','Euclidian','WindowCriterionTolerances',[-Inf 7] );

    for index = 1:length(EEG)
	    pop_saveset(EEG(index), 'filename', newFiles{index}, 'filepath', prepPath);
    end

    eeglab('redraw');    
end
