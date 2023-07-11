sampling_rate = 128;

[ALLEEG EEG CURRENTSET ALLCOM] = eeglab;
for i = 1:1
    for j = 2:5
        prepPath = sprintf('C:\\Users\\ZEPHYRUS\\Desktop\\Imperial\\Thesis\\individual_project\\data\\ds002721-prep\\sub-%02d\\eeg', i);
        eegFilePath = sprintf('C:\\Users\\ZEPHYRUS\\Desktop\\Imperial\\Thesis\\individual_project\\data\\ds002721-prep\\sub-%02d\\eeg\\sub-%02d_task-run%d.set', i, i, j);
        filename = sprintf('sub-%02d_task-run%d_prep.set', i, j);

        EEG = pop_loadset('filename', eegFilePath);

        EEG = pop_resample(EEG, sampling_rate);

        EEG = pop_eegfiltnew(EEG, 'locutoff', 0.5);

        EEG = pop_cleanline(EEG, 'Bandwidth',2,'ChanCompIndices',[1:19], ...
        'SignalType','Channels','ComputeSpectralPower',true,'LineFrequencies',[60 120] , ...
        'NormalizeSpectrum',false,'LineAlpha',0.01,'PaddingFactor',2,'PlotFigures',false, ...
        'ScanForLines',true,'SmoothingFactor',100,'VerbosityLevel',1,'SlidingWinLength', ...
        EEG.pnts/EEG.srate,'SlidingWinStep',EEG.pnts/EEG.srate);

        pop_saveset(EEG, 'filename', filename, 'filepath', prepPath);

        eeglab('redraw');
    end
end
