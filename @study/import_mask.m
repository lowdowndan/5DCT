function import_mask(aStudy,maskFolder, scanNo)

%% Get list of dicom files
% Get list of subfolders within scan directory. Remove '.' and '..' from list
scanFolderList = dir(maskFolder);
scanFolderList(logical(cell2mat(cellfun(@(x) ismember(x,{'.','..'}), {scanFolderList.name}, 'UniformOutput', false)))) = [];

% Get list of files in the scanDirectory
scanFolderFiles = {scanFolderList(~[scanFolderList.isdir]).name};

%% Check if the files are DICOMS
numFolderFiles = length(scanFolderFiles);
isNotDicom = false(numFolderFiles,1);

slices = cell(numFolderFiles,1);
headers = cell(numFolderFiles,1);

for iFile = 1:numFolderFiles
    
    try
        slices{iFile} = dicomread(fullfile(maskFolder,scanFolderFiles{iFile}));
        headers{iFile} = dicominfo(fullfile(maskFolder,scanFolderFiles{iFile}));
    catch ME 
        if(strcmp(ME.identifier,'images:dicominfo:notDICOM'))
            isNotDicom(iFile) = 1;
            warning('Ignoring file %s; Does not appear to be a DICOM file.\n',scanFolderFiles{iFile});
        end
    end

end

slices(isNotDicom) = [];
headers(isNotDicom) = [];

% Get slice location, then sort headers and slices by ascending z position
zPositions = [cellfun(@(x) x.('ImagePositionPatient')(3), headers, 'uni',1)];
[zPositions,zInds] = sort(zPositions,'ascend');

slices = slices(zInds);
headers = headers(zInds);


%% Remove duplicate slices

% Take only Z coordinate from image position patient vector
zIncrement = abs(diff(zPositions));

% Check for duplicate slices at the same z-position
while(any(zIncrement == 0))
    
    duplicateInd = find(zIncrement == 0, 1, 'last');
    
    % Remove
    zPositions(duplicateInd + 1) = [];
    slices(duplicateInd + 1) = [];
    headers(duplicateInd + 1) = [];
    warning('Removing duplicate slice:  Slice %03d', duplicateInd + 1);
    zIncrement = abs(diff(zPositions));
    
end

%% Verify that mask is in the correct frame of reference
assert(strcmp(headers{1}.FrameOfReferenceUID, aStudy.acquisitionInfo.FrameOfReferenceUID),'Mask is not in the same reference frame as the reference image (different FrameOfReferenceUID tag).  Cannot import.');

%% Align/resample to reference grid and resolution

imagePositionPatient = headers{1}.ImagePositionPatient;
elementSpacing = headers{1}.PixelSpacing;
seriesUID = headers{1}.SeriesInstanceUID;
dim = double([headers{1}.Rows; headers{1}.Columns; length(headers)]);

%% Get scan number
if(~exist('scanNo','var'))
    
    % Which scan is this?
    scanNo = nan;
    if(any(strcmp(seriesUID, aStudy.scanIDs)))
	scanNo = find(strcmp(seriesUID,aStudy.scanIDs),1,'first');
    elseif (any(strcmp(seriesUID, aStudy.importedScanIDs)))
	scanNo = find(strcmp(seriesUID,aStudy.importedScanIDs),1,'first');
    else
	error('Mask does not match any series that has been imported into this study.');
    end
end

% Load mask image
mask = single(reshape([slices{:}],[dim(1), dim(2), length(headers)]));
mask = (mask * headers{1}.RescaleSlope) + headers{1}.RescaleIntercept;

% Original grid
xx = imagePositionPatient(1) : elementSpacing(1) : imagePositionPatient(1) + ((dim(1) - 1) * elementSpacing(1));
yy = imagePositionPatient(2) : elementSpacing(2) : imagePositionPatient(2) + ((dim(2) - 1) * elementSpacing(2));
zz = zPositions;

[XX,YY,ZZ] = meshgrid(xx,yy,zz);

%% New grid
xi = aStudy.imagePositionPatient(1) : 1 : aStudy.imagePositionPatient(1) + ((aStudy.dim(1) - 1) * 1);
yi = aStudy.imagePositionPatient(2) : 1 : aStudy.imagePositionPatient(2) + ((aStudy.dim(2) - 1) * 1);
zi = aStudy.zPositions;

[XI,YI,ZI] = meshgrid(xi,yi,zi);

%% Warn if there is a shift in x, y or z.

% X
if (~isequal(aStudy.imagePositionPatient(1), imagePositionPatient(1)))
    shift = aStudy.imagePositionPatient(1) - imagePositionPatient(1);
    warning('Mask is shifted from the reference scan by %0.4f mm in the X direction.  Interpolating to correct.', shift);
end

% Y
if (~isequal(aStudy.imagePositionPatient(2), imagePositionPatient(2)))
    shift = aStudy.imagePositionPatient(2) - imagePositionPatient(2);
    warning('Mask is shifted from the reference scan by %0.4f mm in the Y direction.  Interpolating to correct.', shift);
end

% Z
if (~isequal(aStudy.zPositions(1), zPositions(1)))
    shift = aStudy.zPositions(1) - zPositions(1);
    warning('Mask is shifted from the reference scan by %0.4f mm in the Z direction.  Interpolating to correct.', shift);
end

%% Interpolate image
mask = interp3(XX,YY,ZZ,mask,XI,YI,ZI,'nearest',0);

%% Save
save(fullfile(aStudy.folder,sprintf('mask_%02d.mat',scanNo)),'mask');
