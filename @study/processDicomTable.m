function [dicomTable, scanIDs, sliceCounts] = processDicomTable(aStudy, dicomTable)

%% Convert acquisition time to seconds from midnight
for iRow = 1:length(dicomTable)
    dicomTable{iRow,3} = time2sec(dicomTable{iRow,3});
end

%% Sort headers according to acquisition time.
dicomTable = sortrows(dicomTable,3);

%% Remove any files that don't have an ID (topogram)
sliceIDs = dicomTable(:,9);
emptyInds = cellfun(@isempty,sliceIDs,'uni',1);
sliceIDs(emptyInds) = [];

%% Remove duplicates
[~, uniqueInds] = unique(sliceIDs,'stable');
dicomTable = dicomTable(uniqueInds,:);

% Get list of series instance UIDs. 
scanIDs = unique(dicomTable(:,2),'stable');
sliceCounts = cellfun(@(x) nnz(strcmp(x,dicomTable(:,2))), scanIDs);


%% Check for topograms, remove
topogram = find(sliceCounts == 1);
if ~isempty(topogram)

    for iTopogram = 1:length(topogram)
    topogramInds = strcmp(dicomTable(:,2),scanIDs(topogram(iTopogram)));
    dicomTable(topogramInds,:) = [];
    warning('Topogram %d ignored.', iTopogram)
    end
    
    sliceCounts(topogram) = [];
    scanIDs(topogram) = [];
end

%% Unequal scan lengths?
if any(diff(sliceCounts))
    warning('Scans have unequal numbers of slices.');
end



%% Consider only first (aStudy.nScans) scans
if aStudy.nScans < size(scanIDs,1); 
warning(sprintf('Dicoms for %d scans found, but the number of scans was set to %d.  Synchronizing only the first %d scans.', size(scanIDs,1), aStudy.nScans, aStudy.nScans));

unusedScans = cellfun(@(x) any(strcmp(x, scanIDs(aStudy.nScans + 1:end))), dicomTable(:,2));
dicomTable(unusedScans,:) = [];
scanIDs(aStudy.nScans + 1:end) = [];
end

