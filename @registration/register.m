%% register method

function register(aRegistration)

refScan = aRegistration.refScan;
inputFolder = fullfile(aRegistration.study.folder,'nii');
outputFolder = aRegistration.folder;
chkmkdir(outputFolder);


alpha = aRegistration.parameters.alpha;
samples = aRegistration.parameters.samples;

% Set number of jobs
if(ispref('fiveD','nJobs'))
aRegistration.nJobs = getpref('fiveD','nJobs');
else
aRegistration.nJobs = 2;
end

nJobs = aRegistration.nJobs;
assert(strcmp(aRegistration.algorithm,'deeds'),'Algorithm not recongized.');
regCmd = ['parallelDeedsV2.sh "' inputFolder '" "' outputFolder '" ' num2str(refScan) ' ' num2str(nJobs)];
tic
[status,result] = system(regCmd);
toc
