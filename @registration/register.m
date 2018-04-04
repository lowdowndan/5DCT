%% register method

function register(aRegistration)

refScan = aRegistration.refScan;
inputFolder = fullfile(aRegistration.study.folder,'nii');
outputFolder = aRegistration.folder;
chkmkdir(outputFolder);


alpha = aRegistration.parameters.alpha;
samples = aRegistration.parameters.samples;

nJobs = 3;
assert(strcmp(aRegistration.algorithm,'deeds'),'Algorithm not recongized.');
regCmd = ['~/Dropbox/4DCT/parallelDeedsV2.sh "' inputFolder '" "' outputFolder '" ' num2str(refScan) ' ' num2str(nJobs)]
system(regCmd);
