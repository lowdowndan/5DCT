function mask = load_mask(aStudy,maskFolder)
% mask = aScan.loadMask(maskFolder) reads in a segmentation saved as a collection of DICOMS in maskFolder, assuming the same geometry as the scan object, and resampled the segmentation to 1x1x1.



%% Verify dcmtk is added to path
pathTest = system('dcmdump > /dev/null');

if(pathTest == 127)
    error('dcmdump, one of the dcmtk binaries, is not in the system path.');
end


%% Get list of dicom files

% Get list of subfolders within scan directory. Remove '.' and '..' from list
scanFolderList = dir(maskFolder);
scanFolderList(logical(cell2mat(cellfun(@(x) ismember(x,{'.','..'}), {scanFolderList.name}, 'UniformOutput', false)))) = [];
subfolders = {scanFolderList([scanFolderList.isdir]).name};
numSubfolders = length(subfolders);

% Get list of files in the scanDirectory
scanFolderFiles = {scanFolderList(~[scanFolderList.isdir]).name};

% Command string for dcmdump (requires dcmtk)
dcmtkString = 'dcmdump -M +P "0020,000E" +P "0008,0032" +P "0020,0032" +P "0018,0050" +P "0028,0030" +P "0028,0010" +P "0028,0011" +P "0028,1053" +P "0028,1052" +P "0008,0018" ';


% Set number of columns
numColumns = 9;







%% process


%% Remove duplicate slices

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



[img,headers] = dicomfolder(maskFolder);
% %%TODO: fix sorting slices
% files = dir(maskFolder);
% files = setdiff({files.name},{'.','..'});
% 
% img = dicomread(fullfile(maskFolder,files{1}));
% header = dicominfo(fullfile(maskFolder,files{1}));
% img = zeros([size(img) length(files)], 'single');


if iscell(headers)
header = headers{1};
elseif istruct(headers)
header = headers(1);
else
error('Invalid DICOM headers');
end

elementSpacing = zeros(1,3);
elementSpacing(1:2) = header.PixelSpacing;
elementSpacing(3) = header.SliceThickness;

% for iSlice = 1 :length(files);
%     
%     if iSlice == 1
%     	header = dicominfo(fullfile(maskFolder, files{iSlice}));
%         slope = header.RescaleSlope;
%         intercept = header.RescaleIntercept;
%     end
% 
%      img(:,:,iSlice) = dicomread(fullfile(maskFolder, files{iSlice}));
%     
% end

mask = resampleImage(img, 1 / elementSpacing(1), 1 / elementSpacing(2), 1 / elementSpacing(3), aScan.direction);

%mask = (mask * slope) + intercept;

mask(mask > 0.5) = 1;
mask(mask < 1) = 0;

if ~aScan.direction
    mask = flipdim(mask,3);
end
