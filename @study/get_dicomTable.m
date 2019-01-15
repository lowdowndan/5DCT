%% Support functions

% Build table of dicom tags
function dicomTable = get_dicomTable(aStudy)

dicomFolder = aStudy.dicomFolder;

%% Verify dcmtk is added to path
pathTest = system('dcmdump > /dev/null');

if(pathTest == 127)
    error('dcmdump, one of the dcmtk binaries, is not in the system path.');
end

% Get list of subfolders within scan directory. Remove '.' and '..' from list
scanFolderList = dir(dicomFolder);
scanFolderList(logical(cell2mat(cellfun(@(x) ismember(x,{'.','..'}), {scanFolderList.name}, 'UniformOutput', false)))) = [];
subfolders = {scanFolderList([scanFolderList.isdir]).name};
numSubfolders = length(subfolders);

% Get list of files in the scanDirectory
scanFolderFiles = {scanFolderList(~[scanFolderList.isdir]).name};

% Command string for dcmdump (requires dcmtk)
dcmtkString = 'dcmdump -M +P "0020,000E" +P "0008,0032" +P "0020,0032" +P "0018,0050" +P "0028,0030" +P "0028,0010" +P "0028,0011" +P "0028,1053" +P "0028,1052" +P "0008,0018" ';


% Set number of columns
numColumns = 9;

% Load the headers for dicom files in the scan directory
if (~isempty(scanFolderFiles))
	numFolderFiles = length(scanFolderFiles);
	isNotDicom = ones(numFolderFiles,1,'single');
	headerBar = waitbar(0,'Loading dicom headers...');

	% Preallocate header table
	folderRows = cell(numFolderFiles,numColumns);	

	% Load Headers
	for iFile = 1:numFolderFiles

		%Read dicom metadata
		filename = fullfile(dicomFolder,scanFolderFiles{iFile});
		dcmtkCmd = [dcmtkString '"' filename '"'];
		[isNotDicom(iFile),dcmdumpResult] = system(dcmtkCmd);

		% Parse
		if isNotDicom(iFile) == 0
		folderRows(iFile,:) = study.parse_dump(filename,dcmdumpResult);
		end

		try
			waitbar(iFile/numFolderFiles,headerBar);
		end
	end

	if any(isNotDicom)
	folderRows(logical(isNotDicom),:) = [];
    end
    
    % Remove empty rows (due to topograms or other non axial images)
    folderRows(~cellfun(@ischar,folderRows(:,1),'uni',1),:) = [];

	try
	close(headerBar);
    end
else
    folderRows = cell(0,numColumns);

end

% Get list of files in subfolders
if numSubfolders > 0
	subfolderCounts = zeros(numSubfolders,1);
	subfolderLists = cell(numSubfolders,1);
	
		for iFolder = 1:length(subfolders)
			subfolderLists{iFolder} = dir(fullfile(dicomFolder,subfolders{iFolder}));
			subfolderLists{iFolder} = setdiff({subfolderLists{iFolder}.name}, {'.','..'});
			subfolderCounts(iFolder) = length(subfolderLists{iFolder});
		end
	
	


	numSubfolderFiles = sum(subfolderCounts);
	isNotDicom = ones(numSubfolderFiles,1,'single');
	headerBar = waitbar(0,'Searching subfolders for dicom files and loading headers...');

	% Preallocate header table
	subfolderRows = cell(numSubfolderFiles,numColumns);

	% Load Headers
	for iFolder = 1:numSubfolders

		if subfolderCounts(iFolder) > 0
		for jFile = 1:subfolderCounts(iFolder)

			% Calculate index of this file in the output table
			if iFolder == 1
			fileInd = jFile;
			else
			fileInd = sum(subfolderCounts(1:iFolder-1)) + jFile;
			end

			% Read dicom metadata
			filename = fullfile(dicomFolder,subfolders{iFolder},subfolderLists{iFolder}{jFile});
			dcmtkCmd = [dcmtkString '"' filename '"'];
			[isNotDicom(fileInd),dcmdumpResult] = system(dcmtkCmd);

			% Parse
			if isNotDicom(fileInd) == 0

			subfolderRows(fileInd,:) = study.parse_dump(filename,dcmdumpResult); 

			end

		end
		end

		try
			waitbar(iFolder/numSubfolders,headerBar);
		end
	end


	if any(isNotDicom)
	subfolderRows(logical(isNotDicom),:) = [];
	end

	try
	close(headerBar);
    end
else
    subfolderRows = cell(0,numColumns);
end
	


dicomTable = cat(1,folderRows,subfolderRows);

%% Remove ignored entries (empty rows)
emptyMask = cellfun(@isempty,dicomTable(:,1));
dicomTable(emptyMask,:) = [];

%% Convert acquisition time to seconds from midnight
for iRow = 1:length(dicomTable)
    dicomTable{iRow,3} = study.time2sec(dicomTable{iRow,3});
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

end


