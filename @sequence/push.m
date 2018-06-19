%% Push DICOMs to MIM

function push(aSequence)

%% Verify that scans have been generated
nExtra = 3;
% MIP, ref, error
assert(numel(aSequence.dicomFolders) == numel(aSequence.reconstructionPoints) + nExtra, 'DICOM folders not found.  Run generate_scans method to create DICOMs for exporting.');

%% MIM info

ip = '10.2.139.20';
port = '104';

%% Push
for iFolder = 1:length(aSequence.dicomFolders)
    
    pushCmd = ['storescu --scan-directories ' ip ' ' port ' "' fullfile(aSequence.dicomFolders{iFolder}) '/"'];
    system(pushCmd);
end
