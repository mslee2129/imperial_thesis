[EEG ALLEEG CURRENTSET ALLCOM] = eeglab;

for i = 1:1
    % Form the file paths with the updated integer
    files = {};
    newFiles = {};
    origFiles = {};
    for j = 2:5
        prepPath = sprintf('C:\\Users\\ZEPHYRUS\\Desktop\\Imperial\\Thesis\\individual_project\\data\\ds002721-prep\\sub-%02d\\eeg', i);
        origFilePath = sprintf('C:\\Users\\ZEPHYRUS\\Desktop\\Imperial\\Thesis\\individual_project\\data\\ds002721-prep\\sub-%02d\\eeg\\sub-%02d_task-run%d_prep.set', i, i, j);
        eegFilePath = sprintf('C:\\Users\\ZEPHYRUS\\Desktop\\Imperial\\Thesis\\individual_project\\data\\ds002721-prep\\sub-%02d\\eeg\\sub-%02d_task-run%d_rej.set', i, i, j);
        filename = sprintf('sub-%02d_task-run%d_interp.set', i, j);

        % load data
        % origFiles{end+1} = origFilePath;
        % files{end+1} = eegFilePath;
        % newFiles{end+1} = filename;

        EEG = pop_loadset('filename', eegFilePath);
        TMP = pop_loadset('filename', origFilePath);
        chanlocs = TMP.chanlocs;
        EEG = pop_interp(EEG, chanlocs);
        pop_saveset(EEG, 'filename', filename, 'filepath', prepPath);
        eeglab('redraw');
    end
    % disp(files);
    % % load subject
    % 
    % % load files
    % [ALLEEG EEG CURRENTSET] = pop_newset(ALLEEG, EEG, 4,'retrieve', [1:4] ,'study',0); 
    % chanlocs = EEG.chanlocs;
    
    % Run ICA and IC Label
    % [ALLEEG, EEG] = pop_runica(EEG, 'icatype', 'picard', 'maxiter',500);
    % [ALLEEG, EEG] = pop_iclabel(EEG, 'default');
    % % reject components that are likely to be eye/muslce artifacts
    % EEG = pop_icflag(EEG, [NaN NaN;0.9 1;0.9 1;NaN NaN;NaN NaN;NaN NaN;NaN NaN]);
    % EEG = pop_subcomp(EEG, [], 0);


    % % Interpolate removed channels
    % for index = 1:length(ALLEEG)
	%     EEG = pop_interp(ALLEEG(index), chanlocs, 'spherical');
    % end
    % 
end
