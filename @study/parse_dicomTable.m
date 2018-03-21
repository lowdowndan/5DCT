function [dicomTable, scanIDs, sliceCounts] = parse_dicomTable(aStudy, dicomTable)

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

end