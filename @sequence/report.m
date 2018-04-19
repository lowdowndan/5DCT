%% report
%
% Collate reports from all parent objects for this sequence using
% GhostSript

function report(aSequence)

studyReport = fullfile(aSequence.model.study.folder,'report','report.pdf');
breathReport = fullfile(aSequence.breath.folder,'report','report.pdf');
modelReport = fullfile(aSequence.model.folder,'report','report.pdf');
registrationReport = fullfile(aSequence.model.registration.folder,'report','report.pdf');

chkmkdir(fullfile(aSequence.folder,'report'));
outFile = fullfile(aSequence.folder,'report','report.pdf');
mergeCmd =['gs -q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite -sOutputFile="' outFile '" "' studyReport '" "' breathReport '" "' registrationReport '" "' modelReport '"'];
system(mergeCmd);
