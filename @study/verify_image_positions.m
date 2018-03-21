%% verify_image_positions

function dicomTable = verify_image_positions(aStudy, dicomTable)


% Check status of study object.  If image positions have been verified
% already, return.

% Get list of series instance UIDs. 
scanIDs = unique(dicomTable(:,2),'stable');
sliceCounts = cellfun(@(x) nnz(strcmp(x,dicomTable(:,2))), scanIDs);

%keyboard

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

%% Consider only first (aStudy.nScansValid) scans
if aStudy.nScans < size(scanIDs,1)
    
warning(sprintf('Dicoms for %d scans found, but the number of scans was set to %d.  Synchronizing only the first %d scans.', size(scanIDs,1), aStudy.nScans, aStudy.nScans));

unusedScans = cellfun(@(x) any(strcmp(x, scanIDs(aStudy.nScans + 1:end))), dicomTable(:,2));
%dicomTable(unusedScans,:) = [];
scanIDs(aStudy.nScans + 1:end) = [];
sliceCounts = cellfun(@(x) nnz(strcmp(x,dicomTable(:,2))), scanIDs);

end

%% Check that all scans have contiguous slices.  If any have gaps, mark as invalid
nScansTotal = length(scanIDs);
validScans = true(nScansTotal,1);
tolerance = 0.001;

for iScan = 1 : nScansTotal

seriesInds = strcmp(scanIDs(iScan),dicomTable(:,2));
zPositions = cell2mat(dicomTable(seriesInds,4));

% Take only Z coordinate from image position patient vector
zPositions = zPositions(3:3:end);
zIncrement = abs(diff(zPositions));
%zIncrement = zIncrement - zIncrement(1);

% Check for duplicate slices at the same z-position
while(any(zIncrement == 0))
    
    duplicateInd = find(zIncrement == 0, 1, 'last');
    duplicateIndTable = duplicateInd + find(seriesInds,1,'first');
    
    % Remove
    zPositions(duplicateInd + 1) = [];
    dicomTable(duplicateIndTable,:) = [];
    warning('Removing duplicate slice:  Scan %02d, Slice %03d', iScan,duplicateInd + 1);
    sliceCounts(iScan) = sliceCounts(iScan) - 1;
    zIncrement = abs(diff(zPositions));
    
end

if range(zIncrement) > tolerance
	validScans(iScan) = false;
end

end

% Remove any scans with gaps
if(~all(validScans))

	% Warn user
	invalidScanNumbers = find(~validScans);
	for iScan = 1:length(invalidScanNumbers)
	warning(sprintf('Scan %02d has missing slices and cannot be included in this study.', invalidScanNumbers(iScan)));
	end
end

% Is reference valid
assert(validScans(1),'Reference scan has missing slices.');

%% Find shared range
nScansValid = sum(validScans);

% Get all z positions
allZpositions = cell(nScansValid,1);

for iScan = 1:nScansValid
seriesInds = strcmp(scanIDs(iScan),dicomTable(:,2));
allZpositions{iScan} = cell2mat(dicomTable(seriesInds,4));
% Take only Z coordinate from image position patient vector
allZpositions{iScan} = allZpositions{iScan}(3:3:end);
allZpositions{iScan} = sort(allZpositions{iScan},'ascend');
end

% Find the minimum maximum, and maximum minimum present in all scans
sharedMin = max(cellfun(@min,allZpositions(validScans)));
sharedMax = min(cellfun(@max,allZpositions(validScans)));

% Get the z positions of the reference that are within the common range
commonRefZpositions = allZpositions{1}(allZpositions{1} > sharedMin & allZpositions{1} < sharedMax);

%% Load reference image
seriesInds = strcmp(scanIDs(1),dicomTable(:,2));
filenames = dicomTable(seriesInds,[1 4]);
filenames(:,2) = cellfun(@(x) x(3), filenames(:,2), 'uni', 0);
filenames = sortrows(filenames,2);
filenames = filenames(:,1);

for iFile = 1:length(filenames)

	if(iFile == 1)
		header = dicominfo(filenames{iFile});
		slope = header.RescaleSlope;
		intercept = header.RescaleIntercept;

		img = zeros(header.Rows,header.Columns,length(filenames),'single');
	end

	img(:,:,iFile) = dicomread(filenames{iFile});
end

img = (slope * img) + intercept;

% Make DRR
img = squeeze(sum(img,1));
img = mat2gray(img);
img = img';

%% Plot scan lengths and prompt user for scans to exclude

% Dialog box parameters

% Exclude
text = 'Enter the numbers of any scans you would like to exclude, separated by spaces.  Leave blank to include all scans.';
dlgTitle = 'Scan selection';

% Confirm
question = 'Confirm scan selection?';
respYes = 'Yes';
respNo = 'No';

userResponse = respYes; 

scanIterate = [1:nScansValid];
scanIterate = scanIterate(validScans);
newValidScans = validScans;
userConfirmed = false;
sliceFig = figure;

while(~userConfirmed)

clf(sliceFig);
aStudy.plot_scan_positions(sliceFig, allZpositions, commonRefZpositions, sharedMin,sharedMax,validScans,tolerance,img);
drawnow;
pause(.1);
figure(sliceFig);

% Prompt for scans to exclude
excludeScans = inputdlg(text,dlgTitle,[1 25]);

% Validate user input
if(strcmp(excludeScans,''))

    
    newSharedMin = max(cellfun(@min,allZpositions(newValidScans)));
	newSharedMax = min(cellfun(@max,allZpositions(newValidScans)));
		
	% Get the z positions of the reference that are within the common range
	newCommonRefZpositions = allZpositions{1}(allZpositions{1} > newSharedMin & allZpositions{1} < newSharedMax);
    
    
	% Prompt for confirmation
	userResponse = questdlg(question, dlgTitle, respYes, respNo,respNo);
	if(strcmp(userResponse,respYes))
		userConfirmed = true;
	end

else
	excludeScans = cell2mat(excludeScans);
	excludeScans = str2num(excludeScans);

	if(isempty(excludeScans))

		% Prompt for confirmation
		userResponse = questdlg(question, dlgTitle, respYes, respNo,respNo);
		if(strcmp(userResponse,respYes))
			userConfirmed = true;
		end

	elseif( ismember(1,excludeScans))
		warning('Scan 1, the reference, cannot be exluded.');

	elseif ~all(ismember(excludeScans,scanIterate))
		warning('Invalid entry.');

	else
		% Mark scans as invalid
		newValidScans(excludeScans) = 0;

		clf(sliceFig);

		newSharedMin = max(cellfun(@min,allZpositions(newValidScans)));
		newSharedMax = min(cellfun(@max,allZpositions(newValidScans)));
		
		% Get the z positions of the reference that are within the common range
		newCommonRefZpositions = allZpositions{1}(allZpositions{1} > newSharedMin & allZpositions{1} < newSharedMax);
		aStudy.plot_scan_positions(sliceFig, allZpositions, newCommonRefZpositions, newSharedMin,newSharedMax,newValidScans,tolerance,img);
		drawnow;
		pause(.1);
		figure(sliceFig);

		% Prompt for confirmation
		userResponse = questdlg(question, dlgTitle, respYes, respNo,respNo);
		if(strcmp(userResponse,respYes))
			userConfirmed = true;
		end
	end

% end validation if statement
end
% end while loop
end

% Update plot
validScans = newValidScans;
%aStudy.plot_scan_positions(sliceFig, allZpositions, commonRefZpositions, sharedMin,sharedMax,validScans,tolerance,img);

sharedMin = newSharedMin;
sharedMax = newSharedMax;

% Save plot
documentFolder = fullfile(aStudy.folder,'documents');
f = getframe(sliceFig);
imwrite(f.cdata,fullfile(documentFolder,'scanSelection.png'),'png');
close(sliceFig);

%% Renumber scans
aStudy.nScans = sum(validScans);
aStudy.startScan(~validScans) = [];
aStudy.stopScan(~validScans) = [];
scanIDs(~validScans) = [];
sliceCounts(~validScans) = [];
aStudy.scanRange = [sharedMin sharedMax];



aStudy.scanIDs = scanIDs;
aStudy.sliceCounts = sliceCounts;
