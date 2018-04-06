%% get_scan
%
function aScan = get_scan(aStudy, scanNo)

%assert(scanNo >= aStudy.nScans, 'Invalid scan number.');

load(fullfile(aStudy.folder,sprintf('%02d.mat',scanNo)));
end

