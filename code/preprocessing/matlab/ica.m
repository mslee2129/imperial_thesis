[EEG ALLEEG CURRENTSET ALLCOM] = eeglab;

for i = 1:1
    % Form the file paths with the updated integer
    files = {};
    newFiles = {};
    for j = 2:5
        prepPath = sprintf('C:\\Users\\ZEPHYRUS\\Desktop\\Imperial\\Thesis\\individual_project\\data\\ds002721-prep\\sub-%02d\\eeg', i);
        eegFilePath = sprintf('C:\\Users\\ZEPHYRUS\\Desktop\\Imperial\\Thesis\\individual_project\\data\\ds002721-prep\\sub-%02d\\eeg\\sub-%02d_task-run%d_interp.set', i, i, j);
        filename = sprintf('sub-%02d_task-run%d_ica.set', i, j);

        % load data
        files{end+1} = eegFilePath;
        newFiles{end+1} = filename;
    end

    % load subject
    [ALLEEG EEG] = pop_loadset('filename', files);

    % % load files
    [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 4,'retrieve', [1:4] ,'study',0); 
    
    % Run ICA and IC Label
    [ALLEEG, EEG] = pop_runica(ALLEEG, 'icatype', 'picard', 'maxiter',500);
    [ALLEEG, EEG] = pop_iclabel(ALLEEG, 'default');
    % reject components that are likely to be eye/muslce artifacts

    for index = 1:length(ALLEEG)
	    EEG = pop_icflag(ALLEEG(index), [NaN NaN;0.85 1;0.85 1;NaN NaN;NaN NaN;NaN NaN;NaN NaN]);
        EEG = pop_subcomp(EEG, [], 0);
        pop_saveset(EEG, 'filename', newFiles{index}, 'filepath', prepPath);
    end
    
    eeglab('redraw');
end
